from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers import TrainerCallback
from trl import SFTTrainer


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


def _embedding_weights(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    base_model = _unwrap_model(model)
    input_embeddings = base_model.get_input_embeddings()
    weights = [("input_embeddings.weight", input_embeddings.weight)]
    output_embeddings = base_model.get_output_embeddings()
    if output_embeddings is not None and output_embeddings.weight is not input_embeddings.weight:
        weights.append(("output_embeddings.weight", output_embeddings.weight))
    return weights


class _RowsOnlyInvariantCallback(TrainerCallback):
    """Verify that non-pause embedding rows stay bit-identical during training."""

    def __init__(
        self,
        pause_token_id: int | None = None,
        pause_token_ids: list[int] | tuple[int, ...] | None = None,
        chunk_rows: int = 2048,
        check_interval_steps: int = 50,
    ) -> None:
        if pause_token_ids is None:
            if pause_token_id is None:
                raise ValueError("pause_token_id or pause_token_ids is required")
            pause_token_ids = [int(pause_token_id)]
        self.pause_token_ids = {int(token_id) for token_id in pause_token_ids}
        self.chunk_rows = max(1, int(chunk_rows))
        self.check_interval_steps = max(1, int(check_interval_steps))
        self.snapshots: list[tuple[str, torch.Tensor]] | None = None
        self.last_checked_step = 0

    def on_train_begin(self, args, state, control, model=None, **kwargs):  # type: ignore[override]
        if model is None or self.snapshots is not None:
            return control
        self.snapshots = [
            (name, weight.detach().cpu().clone())
            for name, weight in _embedding_weights(model)
        ]
        return control

    def _check(self, model: torch.nn.Module, step_label: str) -> None:
        current_weights = dict(_embedding_weights(model))
        for name, before in self.snapshots:
            current = current_weights.get(name)
            if current is None:
                raise ValueError(f"Rows-only invariant failed: missing {name} at {step_label}")
            if tuple(current.shape) != tuple(before.shape):
                raise ValueError(
                    f"Rows-only invariant failed: {name} shape changed "
                    f"from {tuple(before.shape)} to {tuple(current.shape)}"
                )

            for start in range(0, before.shape[0], self.chunk_rows):
                end = min(start + self.chunk_rows, before.shape[0])
                now = current.detach()[start:end].cpu()
                changed = now.ne(before[start:end])
                for pause_token_id in self.pause_token_ids:
                    if start <= pause_token_id < end:
                        changed[pause_token_id - start] = False
                if changed.any():
                    bad_rows = (
                        changed.reshape(changed.shape[0], -1)
                        .any(dim=1)
                        .nonzero(as_tuple=True)[0][:10]
                        .add(start)
                        .tolist()
                    )
                    raise ValueError(
                        f"Rows-only invariant failed at {step_label}: "
                        f"non-pause rows changed in {name}; sample rows={bad_rows}. "
                        "Check gradient masks and ensure weight_decay remains 0.0."
                    )

    def on_step_end(self, args, state, control, model=None, **kwargs):  # type: ignore[override]
        if model is None or self.snapshots is None:
            return control
        step = int(getattr(state, "global_step", 0))
        if step < 1:
            return control
        if self.last_checked_step and step - self.last_checked_step < self.check_interval_steps:
            return control
        self._check(model, f"optimizer step {step}")
        self.last_checked_step = step
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):  # type: ignore[override]
        if model is None or self.snapshots is None:
            return control
        step = int(getattr(state, "global_step", 0))
        self._check(model, f"train end step {step}")
        return control


class PauseKLSFTTrainer(SFTTrainer):
    """SFTTrainer variant for transparent pause-token emission.

    This trainer consumes the same formatted examples as the existing Stage2
    SFT path. The only difference is the loss:
    - CE is applied only where the target token is the pause token.
    - Post-pause continuation logits are KL-matched to the same frozen model
      run on the pause-stripped sequence.
    - A small suppression term discourages pause emission at non-pause targets.
    """

    def __init__(self, *args: Any, pause_kl: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self.pause_kl_cfg = dict(pause_kl or {})
        super().__init__(*args, **kwargs)
        tokenizer = getattr(self, "tokenizer", None) or getattr(self, "processing_class", None)
        if tokenizer is None:
            raise ValueError("PauseKLSFTTrainer requires a tokenizer/processing_class")

        raw_pause_tokens = self.pause_kl_cfg.get("pause_tokens")
        configured_n_pause_tokens = int(self.pause_kl_cfg.get("n_pause_tokens", 0) or 0)
        if raw_pause_tokens:
            pause_chain_tokens = [str(token) for token in raw_pause_tokens]
            if (
                configured_n_pause_tokens > 0
                and len(pause_chain_tokens) not in {1, configured_n_pause_tokens}
            ):
                raise ValueError(
                    "pause_kl.pause_tokens length must be 1 or match "
                    f"pause_kl.n_pause_tokens={configured_n_pause_tokens}; "
                    f"got {len(pause_chain_tokens)} tokens."
                )
            if len(pause_chain_tokens) == 1 and configured_n_pause_tokens > 1:
                pause_chain_tokens = pause_chain_tokens * configured_n_pause_tokens
        else:
            pause_token = str(self.pause_kl_cfg.get("pause_token", "<|pause|>"))
            pause_chain_tokens = [pause_token] * max(1, configured_n_pause_tokens)
        if not pause_chain_tokens:
            raise ValueError("PauseKLSFTTrainer requires at least one pause token")
        self.pause_tokens = pause_chain_tokens
        self.pause_token = self.pause_tokens[0]
        pause_chain_token_ids: list[int] = []
        for pause_token in self.pause_tokens:
            pause_token_id = tokenizer.convert_tokens_to_ids(pause_token)
            if pause_token_id is None:
                raise ValueError(f"Unknown pause token for PauseKLSFTTrainer: {pause_token!r}")
            pause_token_id = int(pause_token_id)
            if pause_token_id < 0 or pause_token_id == getattr(tokenizer, "unk_token_id", None):
                raise ValueError(f"Unknown pause token for PauseKLSFTTrainer: {pause_token!r}")
            pause_chain_token_ids.append(pause_token_id)
        self.pause_chain_token_ids = tuple(pause_chain_token_ids)
        self.pause_token_ids = tuple(dict.fromkeys(pause_chain_token_ids))
        self.pause_token_id = self.pause_token_ids[0]
        self.pause_token_id_set = set(self.pause_token_ids)

        self.pad_token_id = int(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else getattr(tokenizer, "eos_token_id", 0)
        )
        self.enabled = bool(self.pause_kl_cfg.get("enabled", True))
        self.continuation_weight = float(self.pause_kl_cfg.get("continuation_weight", 1.0))
        self.pre_weight = float(self.pause_kl_cfg.get("pre_weight", 0.1))
        self.suppression_weight = float(self.pause_kl_cfg.get("suppression_weight", 1.0))
        self.emit_weight = float(self.pause_kl_cfg.get("emit_weight", 0.3))
        self.emit_margin_weight = float(self.pause_kl_cfg.get("emit_margin_weight", 0.0))
        self.stop_weight = float(self.pause_kl_cfg.get("stop_weight", 0.0))
        self.suppression_loss_type = str(self.pause_kl_cfg.get("suppression_loss_type", "unlikelihood"))
        self.emit_margin = float(self.pause_kl_cfg.get("emit_margin", 3.0))
        self.suppression_margin = float(self.pause_kl_cfg.get("suppression_margin", 5.0))
        self.pause_head_cfg = dict(self.pause_kl_cfg.get("pause_head") or {})
        self.pause_head_enabled = bool(self.pause_head_cfg.get("enabled", False))
        self.pause_head: torch.nn.Module | None = None
        if self.pause_head_enabled:
            raise NotImplementedError(
                "pause_head.enabled=true is disabled until generation-time application "
                "and checkpoint loading are implemented. Use rows-only Stage2.1 first."
            )
        self.temperature = float(self.pause_kl_cfg.get("temperature", 1.0))
        self.max_kl_tokens_per_example = int(self.pause_kl_cfg.get("max_kl_tokens_per_example", 256))
        self.suppression_chunk_size = int(self.pause_kl_cfg.get("suppression_chunk_size", 1024))
        self.require_pause_before_continuation_kl = bool(
            self.pause_kl_cfg.get("require_pause_before_continuation_kl", True)
        )
        self.assert_rows_only = bool(self.pause_kl_cfg.get("assert_rows_only", True))
        self.post_step_invariant_check = bool(
            self.pause_kl_cfg.get("post_step_invariant_check", True)
        )
        self.invariant_check_interval_steps = int(
            self.pause_kl_cfg.get("invariant_check_interval_steps", 50)
        )
        self.teacher_eval_mode = bool(self.pause_kl_cfg.get("teacher_eval_mode", True))
        self._last_pause_kl_log_step = -1
        self._pause_row_initial = self._snapshot_pause_rows()
        if self.assert_rows_only:
            self._assert_rows_only_training()
            if self.post_step_invariant_check:
                self.add_callback(
                    _RowsOnlyInvariantCallback(
                        pause_token_ids=list(self.pause_token_ids),
                        check_interval_steps=self.invariant_check_interval_steps,
                    )
                )

    def _hidden_size(self) -> int:
        config = getattr(_unwrap_model(self.model), "config", None)
        for name in ("hidden_size", "n_embd", "d_model"):
            value = getattr(config, name, None)
            if value:
                return int(value)
        embedding = _unwrap_model(self.model).get_input_embeddings()
        return int(embedding.weight.shape[1])

    def _attach_pause_head(self) -> None:
        hidden_size = self._hidden_size()
        bottleneck = int(self.pause_head_cfg.get("hidden_size", 64))
        dropout = float(self.pause_head_cfg.get("dropout", 0.0))
        layers: list[torch.nn.Module] = [
            torch.nn.Linear(hidden_size, bottleneck),
            torch.nn.SiLU(),
        ]
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(bottleneck, len(self.pause_token_ids)))
        head = torch.nn.Sequential(*layers)
        device = next(_unwrap_model(self.model).parameters()).device
        head.to(device=device)
        _unwrap_model(self.model).add_module("_stage21_pause_head", head)
        self.pause_head = head

    def _apply_pause_head(self, outputs: Any, logits: torch.Tensor) -> torch.Tensor:
        if not self.pause_head_enabled:
            return logits
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise ValueError("pause_head.enabled=true requires model outputs with hidden_states")
        pause_delta = self.pause_head(hidden_states[-1]).to(dtype=logits.dtype)
        out = logits.clone()
        pause_ids = self._pause_ids_tensor(logits.device)
        out[:, :, pause_ids] = out[:, :, pause_ids] + pause_delta
        return out

    def _model_kwargs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        allowed = {"input_ids", "attention_mask", "position_ids"}
        return {key: value for key, value in inputs.items() if key in allowed}

    def _assert_rows_only_training(self) -> None:
        weight_decay = float(getattr(self.args, "weight_decay", 0.0) or 0.0)
        if weight_decay != 0.0:
            raise ValueError(
                "PauseKLSFTTrainer rows-only KL teacher requires weight_decay == 0.0. "
                f"Got weight_decay={weight_decay}."
            )

        allowed = {id(weight) for _, weight in _embedding_weights(self.model)}
        if self.pause_head_enabled and self.pause_head is not None:
            allowed.update(id(parameter) for parameter in self.pause_head.parameters())
        bad = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and id(parameter) not in allowed
        ]
        if bad:
            raise ValueError(
                "PauseKLSFTTrainer requires rows-only/format-only training. "
                f"Unexpected trainable parameters: {bad[:20]}"
            )

    def _zero(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.sum() * 0.0

    def _snapshot_pause_rows(self) -> dict[str, torch.Tensor]:
        rows: dict[str, torch.Tensor] = {}
        for name, weight in _embedding_weights(self.model):
            rows[name] = weight.detach()[list(self.pause_token_ids)].float().cpu().clone()
        return rows

    def _pause_row_diagnostics(self, model: torch.nn.Module) -> dict[str, float]:
        diagnostics: dict[str, float] = {}
        for name, weight in _embedding_weights(model):
            safe_name = name.replace(".weight", "").replace(".", "_")
            row = weight.detach()[list(self.pause_token_ids)].float()
            diagnostics[f"pause_row/{safe_name}_norm_mean"] = float(row.norm(dim=-1).mean().cpu())
            initial = self._pause_row_initial.get(name)
            if initial is not None:
                initial = initial.to(device=row.device, dtype=row.dtype)
                cosine = F.cosine_similarity(row, initial, dim=-1)
                diagnostics[f"pause_row/{safe_name}_cosine_to_init_mean"] = float(cosine.mean().cpu())
        return diagnostics

    def _pause_logit_diagnostics(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, float]:
        shift_logits = logits[:, :-1, :]
        shift_targets = input_ids[:, 1:]
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels.ne(-100)
        pause_ids = self._pause_ids_tensor(logits.device)
        pause_target_mask = torch.isin(shift_targets, pause_ids)
        pause_mask = valid_mask & pause_target_mask
        non_pause_mask = valid_mask & ~pause_target_mask
        diagnostics: dict[str, float] = {}
        if pause_mask.any():
            pause_logits = shift_logits[pause_mask].float()
            pause_targets = shift_targets[pause_mask]
            target_log_probs = pause_logits.gather(1, pause_targets.view(-1, 1)).squeeze(1) - torch.logsumexp(
                pause_logits,
                dim=-1,
            )
            pause_argmax = pause_logits.argmax(dim=-1).eq(pause_targets).float()
            diagnostics["pause_emit/target_prob_mean"] = float(target_log_probs.exp().mean().cpu())
            diagnostics["pause_emit/target_argmax_rate"] = float(pause_argmax.mean().cpu())
        if non_pause_mask.any():
            flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
            flat_indices = non_pause_mask.reshape(-1).nonzero(as_tuple=False).flatten()
            prob_sum = self._zero(logits)
            argmax_sum = self._zero(logits)
            count = 0
            for chunk_indices in flat_indices.split(max(1, self.suppression_chunk_size)):
                chunk_logits = flat_logits.index_select(0, chunk_indices).float()
                pause_logits = chunk_logits.index_select(1, pause_ids)
                pause_mass = pause_logits.logsumexp(dim=-1) - torch.logsumexp(chunk_logits, dim=-1)
                prob_sum = prob_sum + pause_mass.exp().sum()
                argmax_sum = argmax_sum + torch.isin(chunk_logits.argmax(dim=-1), pause_ids).float().sum()
                count += int(chunk_logits.shape[0])
            diagnostics["pause_emit/non_pause_prob_mean"] = float((prob_sum / max(count, 1)).cpu())
            diagnostics["pause_emit/non_pause_argmax_rate"] = float((argmax_sum / max(count, 1)).cpu())
        return diagnostics

    def _pause_ids_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(list(self.pause_token_ids), device=device, dtype=torch.long)

    def _should_log_loss_parts(self) -> bool:
        step = int(getattr(self.state, "global_step", 0))
        logging_steps = int(getattr(self.args, "logging_steps", 25) or 25)
        return step != self._last_pause_kl_log_step and step % logging_steps == 0

    def _pause_stripped_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[int, int]]]:
        input_rows = input_ids.detach().cpu().tolist()
        valid_lengths = attention_mask.detach().sum(dim=1).cpu().tolist()
        stripped_rows: list[list[int]] = []
        mappings: list[dict[int, int]] = []
        max_len = 0
        for row_ids, valid_len_raw in zip(input_rows, valid_lengths):
            valid_len = int(valid_len_raw)
            kept: list[int] = []
            row_mapping: dict[int, int] = {}
            teacher_idx = 0
            for src_idx in range(valid_len):
                token_id = int(row_ids[src_idx])
                if token_id in self.pause_token_id_set:
                    continue
                kept.append(token_id)
                row_mapping[src_idx] = teacher_idx
                teacher_idx += 1
            if not kept:
                kept = [self.pad_token_id]
            stripped_rows.append(kept)
            mappings.append(row_mapping)
            max_len = max(max_len, len(kept))

        teacher_ids = input_ids.new_full((input_ids.shape[0], max_len), self.pad_token_id)
        teacher_mask = attention_mask.new_zeros((input_ids.shape[0], max_len))
        for row_idx, stripped_ids in enumerate(stripped_rows):
            length = len(stripped_ids)
            teacher_ids[row_idx, :length] = input_ids.new_tensor(stripped_ids)
            teacher_mask[row_idx, :length] = 1
        return teacher_ids, teacher_mask, mappings

    def _select_kl_pairs(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        mappings: list[dict[int, int]],
    ) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
        post_pairs: list[tuple[int, int, int]] = []
        pre_pairs: list[tuple[int, int, int]] = []
        input_rows = input_ids.detach().cpu().tolist()
        label_rows = labels.detach().cpu().tolist()
        valid_lengths = attention_mask.detach().sum(dim=1).cpu().tolist()
        seq_len = int(input_ids.shape[1])
        for batch_idx in range(int(input_ids.shape[0])):
            valid_len = int(valid_lengths[batch_idx])
            input_row = input_rows[batch_idx]
            label_row = label_rows[batch_idx]
            pause_seen = 0
            row_post: list[tuple[int, int, int]] = []
            row_pre: list[tuple[int, int, int]] = []
            for target_pos in range(1, min(seq_len, valid_len)):
                if int(label_row[target_pos]) == -100:
                    continue
                token_id = int(input_row[target_pos])
                if token_id in self.pause_token_id_set:
                    pause_seen += 1
                    continue
                teacher_target_pos = mappings[batch_idx].get(target_pos)
                if teacher_target_pos is None or teacher_target_pos <= 0:
                    continue
                pair = (batch_idx, target_pos - 1, teacher_target_pos - 1)
                if pause_seen > 0:
                    row_post.append(pair)
                elif not self.require_pause_before_continuation_kl:
                    row_post.append(pair)
                else:
                    row_pre.append(pair)

            row_post = self._cap_pairs(row_post)
            row_pre = self._cap_pairs(row_pre)
            post_pairs.extend(row_post)
            pre_pairs.extend(row_pre)
        return post_pairs, pre_pairs

    def _cap_pairs(self, pairs: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        if self.max_kl_tokens_per_example <= 0 or len(pairs) <= self.max_kl_tokens_per_example:
            return pairs
        keep = torch.linspace(
            0,
            len(pairs) - 1,
            self.max_kl_tokens_per_example,
            dtype=torch.long,
        ).tolist()
        return [pairs[int(idx)] for idx in keep]

    def _gather_pair_logits(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        pairs: list[tuple[int, int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.tensor([item[0] for item in pairs], device=student_logits.device)
        student_idx = torch.tensor([item[1] for item in pairs], device=student_logits.device)
        teacher_idx = torch.tensor([item[2] for item in pairs], device=student_logits.device)
        return student_logits[batch_idx, student_idx], teacher_logits[batch_idx, teacher_idx]

    def _kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        pairs: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        if not pairs:
            return self._zero(student_logits)
        student_sel, teacher_sel = self._gather_pair_logits(student_logits, teacher_logits, pairs)
        student_sel = student_sel.float() / self.temperature
        teacher_sel = teacher_sel.float() / self.temperature

        student_sel = student_sel.clone()
        teacher_sel = teacher_sel.clone()
        pause_ids = self._pause_ids_tensor(student_sel.device)
        student_sel[:, pause_ids] = torch.finfo(student_sel.dtype).min
        teacher_sel[:, pause_ids] = torch.finfo(teacher_sel.dtype).min

        student_log_probs = F.log_softmax(student_sel, dim=-1)
        teacher_probs = F.softmax(teacher_sel, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (
            self.temperature**2
        )

    def _pause_losses(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shift_logits = logits[:, :-1, :]
        shift_targets = input_ids[:, 1:]
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels.ne(-100)
        pause_ids = self._pause_ids_tensor(logits.device)
        pause_target_mask = torch.isin(shift_targets, pause_ids)
        pause_mask = valid_mask & pause_target_mask
        non_pause_mask = valid_mask & ~pause_target_mask

        if pause_mask.any():
            emit = F.cross_entropy(
                shift_logits[pause_mask].float(),
                shift_targets[pause_mask],
                reduction="mean",
            )
            emit_margin = self._emit_margin_loss(shift_logits[pause_mask], shift_targets[pause_mask], pause_ids)
        else:
            emit = self._zero(logits)
            emit_margin = self._zero(logits)

        if non_pause_mask.any():
            flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
            flat_indices = non_pause_mask.reshape(-1).nonzero(as_tuple=False).flatten()
            suppression = self._suppression_loss(flat_logits, flat_indices, pause_ids)
        else:
            suppression = self._zero(logits)

        stop_mask = self._stop_after_pause_chain_mask(input_ids[:, :-1], valid_mask)
        if stop_mask.any():
            stop_loss = self._pause_margin_suppression(shift_logits[stop_mask], pause_ids)
        else:
            stop_loss = self._zero(logits)
        return emit, suppression, emit_margin, stop_loss

    def _non_pause_max(self, logits: torch.Tensor, pause_ids: torch.Tensor) -> torch.Tensor:
        masked = logits.float().clone()
        masked[:, pause_ids] = torch.finfo(masked.dtype).min
        return masked.max(dim=-1).values

    def _competitor_max_except_target(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        masked = logits.float().clone()
        masked.scatter_(1, targets.view(-1, 1), torch.finfo(masked.dtype).min)
        return masked.max(dim=-1).values

    def _emit_margin_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pause_ids: torch.Tensor,
    ) -> torch.Tensor:
        target_logits = logits.float().gather(1, targets.view(-1, 1)).squeeze(1)
        del pause_ids
        competitor_max = self._competitor_max_except_target(logits, targets)
        return F.softplus(competitor_max - target_logits + self.emit_margin).mean()

    def _pause_margin_suppression(self, logits: torch.Tensor, pause_ids: torch.Tensor) -> torch.Tensor:
        pause_logits = logits.float().index_select(1, pause_ids)
        non_pause_max = self._non_pause_max(logits, pause_ids).unsqueeze(1)
        return F.softplus(pause_logits - non_pause_max + self.suppression_margin).mean()

    def _suppression_loss(
        self,
        flat_logits: torch.Tensor,
        flat_indices: torch.Tensor,
        pause_ids: torch.Tensor,
    ) -> torch.Tensor:
        loss_sum = self._zero(flat_logits)
        count = 0
        for chunk_indices in flat_indices.split(max(1, self.suppression_chunk_size)):
            chunk_logits = flat_logits.index_select(0, chunk_indices).float()
            if self.suppression_loss_type == "margin":
                chunk_loss = self._pause_margin_suppression(chunk_logits, pause_ids)
                loss_sum = loss_sum + chunk_loss * int(chunk_indices.numel())
            else:
                pause_logits = chunk_logits.index_select(1, pause_ids)
                pause_log_mass = pause_logits.logsumexp(dim=-1) - torch.logsumexp(
                    chunk_logits,
                    dim=-1,
                )
                pause_probs = pause_log_mass.exp().clamp(max=1.0 - 1e-6)
                loss_sum = loss_sum + (-torch.log1p(-pause_probs)).sum()
            count += int(chunk_indices.numel())
        return loss_sum / max(count, 1)

    def _stop_after_pause_chain_mask(
        self,
        contexts: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        chain_ids = getattr(self, "pause_chain_token_ids", self.pause_token_ids)
        width = len(chain_ids)
        if width <= 0:
            return mask
        if contexts.shape[1] < width:
            return mask
        expected = torch.tensor(chain_ids, device=contexts.device, dtype=contexts.dtype)
        for end_pos in range(width - 1, contexts.shape[1]):
            window = contexts[:, end_pos - width + 1 : end_pos + 1]
            mask[:, end_pos] = window.eq(expected.view(1, -1)).all(dim=1)
        return mask & valid_mask

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        if not self.enabled:
            try:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            except TypeError:
                return super().compute_loss(model, inputs, return_outputs=return_outputs)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        labels = inputs["labels"]

        model_kwargs = self._model_kwargs(inputs)
        if self.pause_head_enabled:
            model_kwargs["output_hidden_states"] = True
        outputs = model(**model_kwargs, use_cache=False)
        student_logits = self._apply_pause_head(outputs, outputs.logits)

        emit_loss, suppression_loss, emit_margin_loss, stop_loss = self._pause_losses(student_logits, input_ids, labels)

        teacher_ids, teacher_mask, mappings = self._pause_stripped_batch(input_ids, attention_mask)
        was_training = model.training
        if self.teacher_eval_mode:
            model.eval()
        with torch.no_grad():
            teacher_outputs = model(input_ids=teacher_ids, attention_mask=teacher_mask, use_cache=False)
            teacher_logits = teacher_outputs.logits.detach()
        if self.teacher_eval_mode and was_training:
            model.train()

        post_pairs, pre_pairs = self._select_kl_pairs(input_ids, labels, attention_mask, mappings)
        continuation_kl = (
            self._kl_loss(student_logits, teacher_logits, post_pairs)
            if self.continuation_weight
            else self._zero(student_logits)
        )
        pre_kl = (
            self._kl_loss(student_logits, teacher_logits, pre_pairs)
            if self.pre_weight
            else self._zero(student_logits)
        )

        loss = (
            self.emit_weight * emit_loss
            + self.emit_margin_weight * emit_margin_loss
            + self.continuation_weight * continuation_kl
            + self.pre_weight * pre_kl
            + self.suppression_weight * suppression_loss
            + self.stop_weight * stop_loss
        )
        diagnostics = None
        if self._should_log_loss_parts():
            with torch.no_grad():
                diagnostics = {
                    **self._pause_logit_diagnostics(student_logits, input_ids, labels),
                    **self._pause_row_diagnostics(model),
                    "pause_kl/post_pairs": float(len(post_pairs)),
                    "pause_kl/pre_pairs": float(len(pre_pairs)),
                }
        self._maybe_log_loss_parts(
            model,
            emit_loss,
            continuation_kl,
            pre_kl,
            suppression_loss,
            emit_margin_loss,
            stop_loss,
            diagnostics,
        )

        if return_outputs:
            return loss, outputs
        return loss

    def _maybe_log_loss_parts(
        self,
        model: torch.nn.Module,
        emit_loss: torch.Tensor,
        continuation_kl: torch.Tensor,
        pre_kl: torch.Tensor,
        suppression_loss: torch.Tensor,
        emit_margin_loss: torch.Tensor,
        stop_loss: torch.Tensor,
        diagnostics: dict[str, float] | None = None,
    ) -> None:
        step = int(getattr(self.state, "global_step", 0))
        if not self._should_log_loss_parts():
            return
        self._last_pause_kl_log_step = step
        prefix = "pause_kl/train" if model.training else "pause_kl/eval"
        payload = {
            f"{prefix}/emit": float(emit_loss.detach().float().cpu()),
            f"{prefix}/continuation": float(continuation_kl.detach().float().cpu()),
            f"{prefix}/pre": float(pre_kl.detach().float().cpu()),
            f"{prefix}/suppression": float(suppression_loss.detach().float().cpu()),
            f"{prefix}/emit_margin": float(emit_margin_loss.detach().float().cpu()),
            f"{prefix}/stop_after_chain": float(stop_loss.detach().float().cpu()),
        }
        if diagnostics:
            payload.update({f"{prefix}/{key}": value for key, value in diagnostics.items()})
        self.log(payload)
