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
    """Verify that non-pause embedding rows stay bit-identical after step 1."""

    def __init__(self, pause_token_id: int, chunk_rows: int = 2048) -> None:
        self.pause_token_id = int(pause_token_id)
        self.chunk_rows = max(1, int(chunk_rows))
        self.snapshots: list[tuple[str, torch.Tensor]] | None = None
        self.checked = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):  # type: ignore[override]
        if model is None or self.snapshots is not None:
            return control
        self.snapshots = [
            (name, weight.detach().cpu().clone())
            for name, weight in _embedding_weights(model)
        ]
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):  # type: ignore[override]
        if self.checked or model is None or self.snapshots is None:
            return control
        if int(getattr(state, "global_step", 0)) < 1:
            return control

        current_weights = dict(_embedding_weights(model))
        for name, before in self.snapshots:
            current = current_weights.get(name)
            if current is None:
                raise ValueError(f"Rows-only invariant failed: missing {name} after step 1")
            if tuple(current.shape) != tuple(before.shape):
                raise ValueError(
                    f"Rows-only invariant failed: {name} shape changed "
                    f"from {tuple(before.shape)} to {tuple(current.shape)}"
                )

            for start in range(0, before.shape[0], self.chunk_rows):
                end = min(start + self.chunk_rows, before.shape[0])
                now = current.detach()[start:end].cpu()
                changed = now.ne(before[start:end])
                if start <= self.pause_token_id < end:
                    changed[self.pause_token_id - start] = False
                if changed.any():
                    bad_rows = (
                        changed.reshape(changed.shape[0], -1)
                        .any(dim=1)
                        .nonzero(as_tuple=True)[0][:10]
                        .add(start)
                        .tolist()
                    )
                    raise ValueError(
                        "Rows-only invariant failed after optimizer step 1: "
                        f"non-pause rows changed in {name}; sample rows={bad_rows}. "
                        "Check gradient masks and ensure weight_decay remains 0.0."
                    )

        self.checked = True
        self.snapshots = None
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

        self.pause_token = str(self.pause_kl_cfg.get("pause_token", "<|pause|>"))
        pause_token_id = tokenizer.convert_tokens_to_ids(self.pause_token)
        if pause_token_id is None:
            raise ValueError(f"Unknown pause token for PauseKLSFTTrainer: {self.pause_token!r}")
        self.pause_token_id = int(pause_token_id)
        if self.pause_token_id < 0 or self.pause_token_id == getattr(tokenizer, "unk_token_id", None):
            raise ValueError(f"Unknown pause token for PauseKLSFTTrainer: {self.pause_token!r}")

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
        self.teacher_eval_mode = bool(self.pause_kl_cfg.get("teacher_eval_mode", True))
        self._last_pause_kl_log_step = -1
        if self.assert_rows_only:
            self._assert_rows_only_training()
            if self.post_step_invariant_check:
                self.add_callback(_RowsOnlyInvariantCallback(self.pause_token_id))

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
                if token_id == self.pause_token_id:
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
                if token_id == self.pause_token_id:
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
        student_sel[:, self.pause_token_id] = torch.finfo(student_sel.dtype).min
        teacher_sel[:, self.pause_token_id] = torch.finfo(teacher_sel.dtype).min

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift_logits = logits[:, :-1, :]
        shift_targets = input_ids[:, 1:]
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels.ne(-100)
        pause_mask = valid_mask & shift_targets.eq(self.pause_token_id)
        non_pause_mask = valid_mask & ~shift_targets.eq(self.pause_token_id)

        if pause_mask.any():
            emit = F.cross_entropy(
                shift_logits[pause_mask].float(),
                shift_targets[pause_mask],
                reduction="mean",
            )
        else:
            emit = self._zero(logits)

        if non_pause_mask.any():
            flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
            flat_indices = non_pause_mask.reshape(-1).nonzero(as_tuple=False).flatten()
            loss_sum = self._zero(logits)
            count = 0
            for chunk_indices in flat_indices.split(max(1, self.suppression_chunk_size)):
                chunk_logits = flat_logits.index_select(0, chunk_indices).float()
                pause_log_probs = chunk_logits[:, self.pause_token_id] - torch.logsumexp(
                    chunk_logits,
                    dim=-1,
                )
                pause_probs = pause_log_probs.exp().clamp(max=1.0 - 1e-6)
                loss_sum = loss_sum + (-torch.log1p(-pause_probs)).sum()
                count += int(chunk_indices.numel())
            suppression = loss_sum / max(count, 1)
        else:
            suppression = self._zero(logits)
        return emit, suppression

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

        outputs = model(**self._model_kwargs(inputs), use_cache=False)
        student_logits = outputs.logits

        emit_loss, suppression_loss = self._pause_losses(student_logits, input_ids, labels)

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
            + self.continuation_weight * continuation_kl
            + self.pre_weight * pre_kl
            + self.suppression_weight * suppression_loss
        )
        self._maybe_log_loss_parts(model, emit_loss, continuation_kl, pre_kl, suppression_loss)

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
    ) -> None:
        step = int(getattr(self.state, "global_step", 0))
        logging_steps = int(getattr(self.args, "logging_steps", 25) or 25)
        if step == self._last_pause_kl_log_step or step % logging_steps != 0:
            return
        self._last_pause_kl_log_step = step
        prefix = "pause_kl/train" if model.training else "pause_kl/eval"
        self.log(
            {
                f"{prefix}/emit": float(emit_loss.detach().float().cpu()),
                f"{prefix}/continuation": float(continuation_kl.detach().float().cpu()),
                f"{prefix}/pre": float(pre_kl.detach().float().cpu()),
                f"{prefix}/suppression": float(suppression_loss.detach().float().cpu()),
            }
        )
