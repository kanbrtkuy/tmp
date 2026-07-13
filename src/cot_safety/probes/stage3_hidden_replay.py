"""Memory-bounded exact-token HF replay for formal Stage 3 extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


class Stage3HiddenReplayError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExactReplayItem:
    token_ids: tuple[int, ...]
    target_positions: tuple[int, ...]
    target_valid: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise Stage3HiddenReplayError("exact replay input cannot be empty")
        if len(self.target_positions) != len(self.target_valid):
            raise Stage3HiddenReplayError("target position/validity length mismatch")
        for position, valid in zip(self.target_positions, self.target_valid):
            if valid and not 0 <= int(position) < len(self.token_ids):
                raise Stage3HiddenReplayError(
                    f"valid target position {position} is outside sequence length {len(self.token_ids)}"
                )


@dataclass(frozen=True)
class ExactReplayBatch:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    position_ids: np.ndarray
    target_positions: np.ndarray
    target_valid: np.ndarray
    sequence_lengths: np.ndarray


def build_exact_replay_batch(
    items: Sequence[ExactReplayItem], *, pad_token_id: int
) -> ExactReplayBatch:
    """Right-pad exact IDs while preserving each row's absolute positions."""

    if not items:
        raise Stage3HiddenReplayError("cannot build an empty exact replay batch")
    n_targets = len(items[0].target_positions)
    if any(len(item.target_positions) != n_targets for item in items):
        raise Stage3HiddenReplayError("replay rows disagree on target count")
    lengths = np.asarray([len(item.token_ids) for item in items], dtype=np.int64)
    width = int(lengths.max())
    input_ids = np.full((len(items), width), int(pad_token_id), dtype=np.int64)
    attention = np.zeros((len(items), width), dtype=np.int64)
    position_ids = np.zeros((len(items), width), dtype=np.int64)
    targets = np.zeros((len(items), n_targets), dtype=np.int64)
    target_valid = np.zeros((len(items), n_targets), dtype=bool)
    for row_index, item in enumerate(items):
        length = len(item.token_ids)
        input_ids[row_index, :length] = np.asarray(item.token_ids, dtype=np.int64)
        attention[row_index, :length] = 1
        position_ids[row_index, :length] = np.arange(length, dtype=np.int64)
        for target_index, (position, valid) in enumerate(
            zip(item.target_positions, item.target_valid)
        ):
            if valid:
                targets[row_index, target_index] = int(position)
                target_valid[row_index, target_index] = True
    return ExactReplayBatch(
        input_ids=input_ids,
        attention_mask=attention,
        position_ids=position_ids,
        target_positions=targets,
        target_valid=target_valid,
        sequence_lengths=lengths,
    )


def _resolve_llama_backbone(model: Any) -> tuple[Any, Sequence[Any], Any]:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    norm = getattr(backbone, "norm", None)
    if backbone is None or layers is None or norm is None:
        raise Stage3HiddenReplayError(
            "formal hook replay requires a Llama-compatible model.model.layers/norm backbone"
        )
    return backbone, layers, norm


def validate_hook_layer_ids(model: Any, layer_ids: Sequence[int]) -> int:
    """Validate Hugging Face hidden-state indices for hook-only capture."""

    _, layers, _ = _resolve_llama_backbone(model)
    n_layers = len(layers)
    normalized = [int(item) for item in layer_ids]
    if len(set(normalized)) != len(normalized):
        raise Stage3HiddenReplayError("hook layer IDs must be unique")
    if any(layer <= 0 or layer > n_layers for layer in normalized):
        raise Stage3HiddenReplayError(
            f"hook layer IDs must lie in [1,{n_layers}]: {normalized}"
        )
    return n_layers


def is_cuda_oom(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return "outofmemory" in name or "cuda out of memory" in message


def capture_exact_hidden_batch(
    model: Any,
    batch: ExactReplayBatch,
    *,
    layer_ids: Sequence[int],
    device: str,
) -> np.ndarray:
    """Capture only requested positions, without retaining all-layer states.

    Primary hidden-state index ``l`` is the output of decoder block ``l-1``.
    Terminal index ``n_layers`` is captured after the final norm, matching the
    last element returned by ``output_hidden_states=True``.
    """

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise Stage3HiddenReplayError("torch is required for exact HF replay") from exc

    n_layers = validate_hook_layer_ids(model, layer_ids)
    backbone, decoder_layers, final_norm = _resolve_llama_backbone(model)
    input_ids = torch.as_tensor(batch.input_ids, dtype=torch.long, device=device)
    attention_mask = torch.as_tensor(
        batch.attention_mask, dtype=torch.long, device=device
    )
    position_ids = torch.as_tensor(batch.position_ids, dtype=torch.long, device=device)
    target_positions = torch.as_tensor(
        batch.target_positions, dtype=torch.long, device=device
    )
    row_ids = torch.arange(input_ids.shape[0], device=device)[:, None]
    captured: dict[int, Any] = {}
    handles = []

    def hook_for(layer_id: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
                raise Stage3HiddenReplayError(
                    f"unexpected hook output for hidden-state index {layer_id}"
                )
            captured[layer_id] = tensor[row_ids, target_positions]

        return capture

    for layer_id in [int(item) for item in layer_ids]:
        module = final_norm if layer_id == n_layers else decoder_layers[layer_id - 1]
        handles.append(module.register_forward_hook(hook_for(layer_id)))
    try:
        with torch.inference_mode():
            backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
        missing = [int(layer) for layer in layer_ids if int(layer) not in captured]
        if missing:
            raise Stage3HiddenReplayError(f"hidden hooks did not fire: {missing}")
        stacked = torch.stack(
            [captured[int(layer)] for layer in layer_ids], dim=1
        ).float()
        valid = torch.as_tensor(batch.target_valid, dtype=torch.bool, device=device)
        stacked = stacked.masked_fill(~valid[:, None, :, None], 0.0)
        return stacked.cpu().numpy().astype(np.float16)
    finally:
        for handle in handles:
            handle.remove()


def replay_with_oom_policy(
    model: Any,
    items: Sequence[ExactReplayItem],
    *,
    layer_ids: Sequence[int],
    pad_token_id: int,
    device: str,
    batch_size: int,
    min_batch_size: int = 1,
    oom_policy: str = "halve",
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Replay in stable order, halving only the failing CUDA batch if allowed."""

    if int(batch_size) <= 0 or int(min_batch_size) <= 0:
        raise Stage3HiddenReplayError("batch sizes must be positive")
    if int(min_batch_size) > int(batch_size):
        raise Stage3HiddenReplayError("min_batch_size cannot exceed batch_size")
    if oom_policy not in {"halve", "fail"}:
        raise Stage3HiddenReplayError("oom_policy must be 'halve' or 'fail'")
    if not items:
        return np.empty((0, len(layer_ids), 0, 0), dtype=np.float16), {
            "configured_batch_size": int(batch_size),
            "minimum_effective_batch_size": None,
            "cuda_oom_retries": 0,
        }
    outputs: list[np.ndarray] = []
    cursor = 0
    current = int(batch_size)
    minimum_used = int(batch_size)
    oom_retries = 0
    while cursor < len(items):
        take = min(current, len(items) - cursor)
        planned = build_exact_replay_batch(
            items[cursor : cursor + take], pad_token_id=pad_token_id
        )
        try:
            values = capture_exact_hidden_batch(
                model, planned, layer_ids=layer_ids, device=device
            )
        except Exception as exc:  # CUDA OOM type differs across torch releases.
            if not is_cuda_oom(exc):
                raise
            oom_retries += 1
            if oom_policy == "fail" or take <= int(min_batch_size):
                raise Stage3HiddenReplayError(
                    f"CUDA OOM at batch size {take}; formal replay stopped fail-closed"
                ) from exc
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            current = max(int(min_batch_size), take // 2)
            continue
        outputs.append(values)
        minimum_used = min(minimum_used, take)
        cursor += take
    return np.concatenate(outputs, axis=0), {
        "configured_batch_size": int(batch_size),
        "minimum_effective_batch_size": int(minimum_used),
        "cuda_oom_retries": int(oom_retries),
        "oom_policy": oom_policy,
        "right_padding": True,
        "explicit_position_ids": True,
        "decode_retokenize": False,
        "capture_backend": "decoder_hooks_requested_positions_only",
    }
