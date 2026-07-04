from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from trl import SFTTrainer

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
        self.pause_token_id = int(tokenizer.convert_tokens_to_ids(self.pause_token))
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
        self.require_pause_before_continuation_kl = bool(
            self.pause_kl_cfg.get("require_pause_before_continuation_kl", True)
        )

    def _model_kwargs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        allowed = {"input_ids", "attention_mask", "position_ids"}
        return {key: value for key, value in inputs.items() if key in allowed}

    def _zero(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.sum() * 0.0

    def _pause_stripped_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[int, int]]]:
        stripped_rows: list[torch.Tensor] = []
        mappings: list[dict[int, int]] = []
        max_len = 0
        for row_ids, row_mask in zip(input_ids, attention_mask):
            valid_len = int(row_mask.sum().item())
            kept: list[torch.Tensor] = []
            row_mapping: dict[int, int] = {}
            teacher_idx = 0
            for src_idx in range(valid_len):
                token_id = int(row_ids[src_idx].item())
                if token_id == self.pause_token_id:
                    continue
                kept.append(row_ids[src_idx])
                row_mapping[src_idx] = teacher_idx
                teacher_idx += 1
            if not kept:
                kept = [row_ids.new_tensor(self.pad_token_id)]
            stripped = torch.stack(kept)
            stripped_rows.append(stripped)
            mappings.append(row_mapping)
            max_len = max(max_len, int(stripped.numel()))

        teacher_ids = input_ids.new_full((input_ids.shape[0], max_len), self.pad_token_id)
        teacher_mask = attention_mask.new_zeros((input_ids.shape[0], max_len))
        for row_idx, stripped in enumerate(stripped_rows):
            length = int(stripped.numel())
            teacher_ids[row_idx, :length] = stripped
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
        seq_len = int(input_ids.shape[1])
        for batch_idx in range(int(input_ids.shape[0])):
            valid_len = int(attention_mask[batch_idx].sum().item())
            pause_seen = 0
            row_post: list[tuple[int, int, int]] = []
            row_pre: list[tuple[int, int, int]] = []
            for target_pos in range(1, min(seq_len, valid_len)):
                if int(labels[batch_idx, target_pos].item()) == -100:
                    continue
                token_id = int(input_ids[batch_idx, target_pos].item())
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

            if self.max_kl_tokens_per_example > 0 and len(row_post) > self.max_kl_tokens_per_example:
                keep = torch.linspace(
                    0,
                    len(row_post) - 1,
                    self.max_kl_tokens_per_example,
                    dtype=torch.long,
                ).tolist()
                row_post = [row_post[int(idx)] for idx in keep]
            post_pairs.extend(row_post)
            pre_pairs.extend(row_pre)
        return post_pairs, pre_pairs

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
            pause_log_probs = F.log_softmax(shift_logits[non_pause_mask].float(), dim=-1)[
                :, self.pause_token_id
            ]
            pause_probs = pause_log_probs.exp().clamp(max=1.0 - 1e-6)
            suppression = -torch.log1p(-pause_probs).mean()
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
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        labels = inputs["labels"]

        outputs = model(**self._model_kwargs(inputs), use_cache=False)
        student_logits = outputs.logits

        emit_loss, suppression_loss = self._pause_losses(student_logits, input_ids, labels)

        teacher_ids, teacher_mask, mappings = self._pause_stripped_batch(input_ids, attention_mask)
        with torch.no_grad():
            teacher_outputs = model(input_ids=teacher_ids, attention_mask=teacher_mask, use_cache=False)
            teacher_logits = teacher_outputs.logits.detach()

        post_pairs, pre_pairs = self._select_kl_pairs(input_ids, labels, attention_mask, mappings)
        continuation_kl = self._kl_loss(student_logits, teacher_logits, post_pairs)
        pre_kl = self._kl_loss(student_logits, teacher_logits, pre_pairs)

        loss = (
            self.emit_weight * emit_loss
            + self.continuation_weight * continuation_kl
            + self.pre_weight * pre_kl
            + self.suppression_weight * suppression_loss
        )

        if return_outputs:
            return loss, outputs
        return loss
