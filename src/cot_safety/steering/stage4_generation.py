"""Formal Stage-4 minimal-prefix counterfactual generation primitives.

This module is intentionally independent of the archival GPRS generator.  It
implements the 2026-07-14 estimand:

* A0/A1 are sampled naturally with counter-based common random numbers;
* A2--A5 teacher-replay *exact A1 token ids* only through the final target;
* a decoder-block output hook edits exactly three named positions before
  subsequent decoder blocks write their K/V cache; and
* generation resumes immediately from the returned, intervention-bearing
  cache.  No pause token is forced, suppressed, repaired, or regenerated.

The public helpers are kept small enough to test with a synthetic model.  The
CLI which loads real Hugging Face models lives in
``scripts/run_stage4_formal_generation_hf.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "stage4_formal_minimal_prefix_generation_v1"
COUNTER_SAMPLER_VERSION = "sha256_u53_top_p_v1"


class Stage4GenerationError(ValueError):
    """A fail-closed generation or provenance error."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def require_sha256(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise Stage4GenerationError(f"{field}_must_be_64_lowercase_hex_sha256")
    return normalized


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def content_sha256(prompt_token_ids: Sequence[int], output_token_ids: Sequence[int]) -> str:
    """Bind a generated response to the exact prompt and output token ids."""

    return sha256_text(
        canonical_json(
            {
                "prompt_token_ids": [int(item) for item in prompt_token_ids],
                "output_token_ids": [int(item) for item in output_token_ids],
            }
        )
    )


def counter_uniform(
    *,
    run_id: str,
    prompt_id: str,
    rollout_seed: int,
    absolute_output_position: int,
) -> float:
    """Return a deterministic U(0,1) variate without mutable RNG state.

    Fifty-three digest bits are mapped to the open interval, making the result
    exactly representable as the midpoint of a binary64 bin.  Arm is
    deliberately absent from the key: A0--A5 share the same uniform at the
    same absolute output position.
    """

    if int(absolute_output_position) < 0:
        raise Stage4GenerationError("absolute_output_position_must_be_nonnegative")
    key = canonical_json(
        {
            "sampler": COUNTER_SAMPLER_VERSION,
            "run_id": str(run_id),
            "prompt_id": str(prompt_id),
            "rollout_seed": int(rollout_seed),
            "absolute_output_position": int(absolute_output_position),
        }
    )
    integer = int(sha256_text(key)[:14], 16) >> 3  # exactly 53 bits
    return (integer + 0.5) / float(1 << 53)


def sample_top_p_from_uniform(logits: Any, *, uniform: float, temperature: float, top_p: float) -> int:
    """Sample one token from temperature/top-p logits using a supplied U(0,1)."""

    import torch

    if logits.ndim != 1:
        raise Stage4GenerationError(f"expected_rank_one_logits:{tuple(logits.shape)}")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise Stage4GenerationError(f"invalid_temperature:{temperature}")
    if not math.isfinite(float(top_p)) or not 0.0 < float(top_p) <= 1.0:
        raise Stage4GenerationError(f"invalid_top_p:{top_p}")
    if not 0.0 < float(uniform) < 1.0:
        raise Stage4GenerationError(f"uniform_not_in_open_interval:{uniform}")
    # Float32 makes the sampling rule independent of BF16 model output dtype.
    values = logits.detach().float() / float(temperature)
    if not bool(torch.isfinite(values).all()):
        raise Stage4GenerationError("nonfinite_logits")
    sorted_logits, sorted_indices = torch.sort(values, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Keep the first token which crosses top_p (the standard nucleus rule).
    keep = (cumulative - sorted_probs) < float(top_p)
    kept_count = int(keep.sum().item())
    if kept_count <= 0:
        raise Stage4GenerationError("empty_top_p_nucleus")
    kept_probs = sorted_probs[:kept_count]
    kept_indices = sorted_indices[:kept_count]
    total = kept_probs.sum()
    if not bool(torch.isfinite(total)) or float(total.item()) <= 0.0:
        raise Stage4GenerationError("empty_or_nonfinite_top_p_distribution")
    kept_probs = kept_probs / total
    kept_cdf = torch.cumsum(kept_probs, dim=-1)
    # Avoid selecting a zero-mass/out-of-range tail because of float32 CDF
    # roundoff when a counter uniform is extremely close to one.
    kept_cdf[-1] = 1.0
    needle = torch.tensor(float(uniform), device=kept_cdf.device, dtype=kept_cdf.dtype)
    selected = int(torch.searchsorted(kept_cdf, needle, right=False).item())
    selected = min(selected, kept_count - 1)
    return int(kept_indices[selected].item())


def stable_rollout_seed(
    global_seed: int,
    *,
    run_id: str,
    phase: str,
    source: str,
    prompt_id: str,
    draw_index: int,
) -> int:
    payload = canonical_json(
        {
            "global_seed": int(global_seed),
            "run_id": str(run_id),
            "phase": str(phase),
            "source": str(source),
            "prompt_id": str(prompt_id),
            "draw_index": int(draw_index),
        }
    )
    return int(sha256_text(payload)[:8], 16) & 0x7FFFFFFF


def stable_shard(group_id: str, num_shards: int) -> int:
    if int(num_shards) <= 0:
        raise Stage4GenerationError("num_shards_must_be_positive")
    return int(sha256_text(str(group_id))[:16], 16) % int(num_shards)


@dataclass(frozen=True)
class SamplingSpec:
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 2048

    def validate(self) -> None:
        if float(self.temperature) != 0.6 or float(self.top_p) != 0.95:
            raise Stage4GenerationError("formal_sampling_must_be_temperature_0.6_top_p_0.95")
        if int(self.max_new_tokens) != 2048:
            raise Stage4GenerationError("formal_sampling_must_use_2048_max_new_tokens")


@dataclass(frozen=True)
class CounterKey:
    run_id: str
    prompt_id: str
    rollout_seed: int

    def uniform(self, absolute_output_position: int) -> float:
        return counter_uniform(
            run_id=self.run_id,
            prompt_id=self.prompt_id,
            rollout_seed=self.rollout_seed,
            absolute_output_position=absolute_output_position,
        )


@dataclass(frozen=True)
class TargetPlan:
    """Exact A1 target positions, relative to prompt+output token ids."""

    positions: dict[str, int]
    token_ids: dict[str, int]
    output_offsets: dict[str, int]
    structural_valid: bool
    missing: tuple[str, ...]
    info: dict[str, Any]

    def for_names(self, names: Sequence[str]) -> tuple[list[int], list[int]]:
        missing = [str(name) for name in names if str(name) not in self.positions]
        if missing:
            raise Stage4GenerationError(f"missing_a1_targets:{missing}")
        positions = [int(self.positions[str(name)]) for name in names]
        token_ids = [int(self.token_ids[str(name)]) for name in names]
        if len(set(positions)) != len(positions):
            raise Stage4GenerationError(f"duplicate_target_positions:{positions}")
        return positions, token_ids


FORMAL_TARGET_NAMES = (
    "cot_2",
    "cot_3",
    "cot_4",
    "pause_0",
    "pause_1",
    "pause_2",
    "post_pause_1",
    "post_pause_2",
    "post_pause_3",
)


def _ordinary_token(
    tokenizer: Any,
    token_id: int,
    *,
    pause_token_id: int,
    forbidden_ids: set[int],
) -> bool:
    token_id = int(token_id)
    if token_id == int(pause_token_id) or token_id in forbidden_ids:
        return False
    if token_id in {int(item) for item in (getattr(tokenizer, "all_special_ids", None) or [])}:
        return False
    piece = tokenizer.decode([token_id], skip_special_tokens=False)
    return bool(str(piece).strip())


def resolve_a1_target_plan(
    tokenizer: Any,
    *,
    prompt_token_ids: Sequence[int],
    output_token_ids: Sequence[int],
    pause_token_id: int,
    assistant_ids: Sequence[int],
    think_ids: Sequence[int],
    end_think_ids: Sequence[int],
) -> TargetPlan:
    """Resolve formal targets from exact A1 ids, including ordinary post-pause tokens."""

    from cot_safety.probes.stage3_replay import resolve_formal_positions
    from cot_safety.steering.targeting import resolve_steering_positions

    prompt_ids = [int(item) for item in prompt_token_ids]
    output_ids = [int(item) for item in output_token_ids]
    full_ids = prompt_ids + output_ids
    _formal_positions, formal_info = resolve_formal_positions(
        tokenizer,
        prompt_token_ids=prompt_ids,
        output_token_ids=output_ids,
        pause_token_id=int(pause_token_id),
        assistant_ids=assistant_ids,
        think_ids=think_ids,
        end_think_ids=end_think_ids,
    )
    resolved = resolve_steering_positions(
        tokenizer,
        full_ids,
        assistant_ids=assistant_ids,
        pause_ids=[int(pause_token_id)],
        think_ids=think_ids,
        end_think_ids=end_think_ids,
        n_pause_tokens=3,
        allow_open_ended_think=True,
        pre_pause_window=3,
        post_pause_window=0,
    )
    positions = {
        name: int(resolved.positions[name])
        for name in ("cot_2", "cot_3", "cot_4", "pause_0", "pause_1", "pause_2")
        if name in resolved.positions
    }
    reasoning_end = int(resolved.info.get("reasoning_end", len(full_ids)))
    pause_2 = positions.get("pause_2")
    ordinary_post: list[int] = []
    if pause_2 is not None:
        forbidden_ids = {int(item) for item in end_think_ids}
        for absolute_pos in range(int(pause_2) + 1, min(reasoning_end, len(full_ids))):
            if _ordinary_token(
                tokenizer,
                full_ids[absolute_pos],
                pause_token_id=int(pause_token_id),
                forbidden_ids=forbidden_ids,
            ):
                ordinary_post.append(int(absolute_pos))
                if len(ordinary_post) == 3:
                    break
    for ordinal, absolute_pos in enumerate(ordinary_post, start=1):
        positions[f"post_pause_{ordinal}"] = int(absolute_pos)

    # Every intervention target must be generated, never part of the prompt.
    positions = {name: pos for name, pos in positions.items() if int(pos) >= len(prompt_ids)}
    missing = tuple(name for name in FORMAL_TARGET_NAMES if name not in positions)
    token_ids = {name: int(full_ids[pos]) for name, pos in positions.items()}
    output_offsets = {name: int(pos) - len(prompt_ids) for name, pos in positions.items()}
    # Base structural validity is the exact-three/location Stage2 contract.
    # Missing later A4 tokens is arm-specific and must not invalidate A2/A3/A5.
    structural_valid = bool(formal_info.get("structural_valid"))
    info = {
        "stage3_structural": formal_info,
        "targeting": resolved.info,
        "first_three_ordinary_post_pause_positions": ordinary_post,
        "all_formal_targets_present": not missing,
    }
    return TargetPlan(
        positions=positions,
        token_ids=token_ids,
        output_offsets=output_offsets,
        structural_valid=structural_valid,
        missing=missing,
        info=info,
    )


def get_decoder_layers(model: Any) -> Any:
    """Resolve decoder blocks without accepting an LM head as a layer list."""

    candidates = (
        ("model", "layers"),
        ("model", "model", "layers"),
        ("base_model", "model", "model", "layers"),
        ("transformer", "h"),
    )
    for path in candidates:
        current = model
        try:
            for attr in path:
                current = getattr(current, attr)
        except AttributeError:
            continue
        if hasattr(current, "__len__") and len(current) > 0:
            return current
    raise Stage4GenerationError("unable_to_resolve_decoder_layers")


def hidden_index_to_block_index(hidden_state_index: int, *, num_decoder_blocks: int) -> int:
    layer = int(hidden_state_index)
    if not 1 <= layer < int(num_decoder_blocks):
        raise Stage4GenerationError(
            f"nonsteerable_hidden_state_index:{layer}:required=1..{int(num_decoder_blocks) - 1}"
        )
    return layer - 1


def _replace_hidden(output: Any, hidden: Any) -> Any:
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if isinstance(output, list):
        return [hidden] + list(output[1:])
    # Decoder blocks normally return tuples, but a tiny test block may return a
    # tensor directly.
    return hidden


@contextmanager
def exact_matched_relative_hook(
    model: Any,
    *,
    hidden_state_index: int,
    padded_target_mask: Any,
    unit_direction: Any,
    rho: float,
    target_names_by_row: Sequence[Sequence[str]],
    target_token_ids_by_row: Sequence[Sequence[int]],
):
    """Edit exactly three positions at decoder block ``hidden_index - 1``.

    The hook is active only for the first full-prefix call.  It returns edited
    block output, so every later decoder block consumes the edited states and
    writes intervention-bearing K/V entries.  Cached one-token calls cannot be
    touched because their sequence shape differs from the frozen prefix mask.
    """

    import torch

    if not math.isfinite(float(rho)) or not 0.0 < float(rho) <= 0.10:
        raise Stage4GenerationError(f"formal_nonzero_rho_out_of_range:{rho}")
    if padded_target_mask.ndim != 2:
        raise Stage4GenerationError("target_mask_must_be_rank_two")
    counts = [int(item) for item in padded_target_mask.sum(dim=1).detach().cpu().tolist()]
    if any(count != 3 for count in counts):
        raise Stage4GenerationError(f"exactly_three_targets_required_per_row:{counts}")
    if len(target_names_by_row) != len(counts) or len(target_token_ids_by_row) != len(counts):
        raise Stage4GenerationError("target_audit_batch_length_mismatch")
    if any(len(tuple(names)) != 3 for names in target_names_by_row):
        raise Stage4GenerationError("exactly_three_target_names_required_per_row")
    if any(len(tuple(ids)) != 3 for ids in target_token_ids_by_row):
        raise Stage4GenerationError("exactly_three_target_token_ids_required_per_row")

    layers = get_decoder_layers(model)
    block_index = hidden_index_to_block_index(
        int(hidden_state_index), num_decoder_blocks=len(layers)
    )
    direction = unit_direction.detach().float().reshape(-1)
    direction_norm = direction.norm()
    if not bool(torch.isfinite(direction_norm)) or float(direction_norm.item()) <= 0.0:
        raise Stage4GenerationError("direction_has_invalid_norm")
    direction = direction / direction_norm
    stats: dict[str, Any] = {
        "hidden_state_index": int(hidden_state_index),
        "decoder_block_index": int(block_index),
        "rho": float(rho),
        "hook_registered_before_prefix_forward": True,
        "hook_calls": [],
        "num_applied_calls": 0,
        "per_row": [
            {
                "target_names": [str(item) for item in names],
                "touched_token_ids": [int(item) for item in ids],
                "actual_relative_norms": [],
                "actual_delta_norms": [],
                "pre_update_hidden_norms": [],
            }
            for names, ids in zip(target_names_by_row, target_token_ids_by_row)
        ],
    }
    applied = False

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        nonlocal applied
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        call = {
            "call_index": len(stats["hook_calls"]),
            "hidden_shape": [int(item) for item in hidden.shape],
            "phase": "full_prefix_before_cache_return" if not applied else "cached_continuation",
        }
        stats["hook_calls"].append(call)
        if applied:
            return output
        if tuple(hidden.shape[:2]) != tuple(padded_target_mask.shape):
            raise Stage4GenerationError(
                f"first_hook_call_not_full_prefix:{tuple(hidden.shape[:2])}!={tuple(padded_target_mask.shape)}"
            )
        mask = padded_target_mask.to(device=hidden.device, dtype=torch.bool)
        edited = hidden.clone()
        local_direction = direction.to(device=hidden.device, dtype=hidden.dtype)
        for row_index in range(int(hidden.shape[0])):
            positions = mask[row_index].nonzero(as_tuple=False).flatten()
            selected = hidden[row_index, positions]
            hidden_norms = selected.float().norm(dim=-1).clamp_min(1e-12)
            delta = -float(rho) * hidden_norms.to(selected.dtype).unsqueeze(-1) * local_direction
            updated = selected + delta
            edited[row_index, positions] = updated
            delta_norms = delta.float().norm(dim=-1)
            relative = delta_norms / hidden_norms
            row_stats = stats["per_row"][row_index]
            row_stats["actual_relative_norms"] = [
                float(item) for item in relative.detach().cpu().tolist()
            ]
            row_stats["actual_delta_norms"] = [
                float(item) for item in delta_norms.detach().cpu().tolist()
            ]
            row_stats["pre_update_hidden_norms"] = [
                float(item) for item in hidden_norms.detach().cpu().tolist()
            ]
        applied = True
        stats["num_applied_calls"] = 1
        return _replace_hidden(output, edited)

    handle = layers[block_index].register_forward_hook(hook)
    try:
        yield stats
    finally:
        handle.remove()


def _output_field(outputs: Any, name: str) -> Any:
    if hasattr(outputs, name):
        return getattr(outputs, name)
    if isinstance(outputs, Mapping):
        return outputs[name]
    raise Stage4GenerationError(f"model_output_missing:{name}")


def _position_ids(attention_mask: Any) -> Any:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 0)


def left_pad_sequences(sequences: Sequence[Sequence[int]], *, pad_token_id: int, device: Any) -> tuple[Any, Any, list[int]]:
    import torch

    if not sequences or any(len(row) == 0 for row in sequences):
        raise Stage4GenerationError("nonempty_sequence_batch_required")
    lengths = [len(row) for row in sequences]
    width = max(lengths)
    input_ids = torch.full(
        (len(sequences), width), int(pad_token_id), device=device, dtype=torch.long
    )
    attention_mask = torch.zeros((len(sequences), width), device=device, dtype=torch.long)
    for row_index, row in enumerate(sequences):
        values = torch.tensor([int(item) for item in row], device=device, dtype=torch.long)
        input_ids[row_index, width - len(row) :] = values
        attention_mask[row_index, width - len(row) :] = 1
    return input_ids, attention_mask, lengths


def _eos_set(eos_token_ids: int | Sequence[int] | None) -> set[int]:
    if eos_token_ids is None:
        return set()
    if isinstance(eos_token_ids, int):
        return {int(eos_token_ids)}
    return {int(item) for item in eos_token_ids}


def _continue_from_outputs(
    model: Any,
    *,
    initial_outputs: Any,
    attention_mask: Any,
    initial_output_ids: Sequence[Sequence[int]],
    counter_keys: Sequence[CounterKey] | None,
    sampling: SamplingSpec,
    eos_token_ids: int | Sequence[int] | None,
    greedy: bool = False,
) -> tuple[list[list[int]], list[str]]:
    """Sample batched continuations, keyed by each absolute output position."""

    import torch

    if greedy:
        if (
            float(sampling.temperature) != 0.0
            or float(sampling.top_p) != 1.0
            or int(sampling.max_new_tokens) != 2048
        ):
            raise Stage4GenerationError("formal_greedy_decoding_must_be_0_1_2048")
    else:
        sampling.validate()
    output_ids = [[int(item) for item in row] for row in initial_output_ids]
    if not greedy and (counter_keys is None or len(output_ids) != len(counter_keys)):
        raise Stage4GenerationError("counter_key_batch_length_mismatch")
    if any(len(row) > int(sampling.max_new_tokens) for row in output_ids):
        raise Stage4GenerationError("teacher_prefix_exceeds_max_new_tokens")
    logits = _output_field(initial_outputs, "logits")[:, -1, :]
    past = _output_field(initial_outputs, "past_key_values")
    if past is None:
        raise Stage4GenerationError("model_did_not_return_past_key_values")
    eos = _eos_set(eos_token_ids)
    active = [not row or row[-1] not in eos for row in output_ids]
    finish = ["prefix_eos" if not state else "" for state in active]
    max_steps = max(int(sampling.max_new_tokens) - len(row) for row in output_ids)

    for _step in range(max_steps):
        sampled: list[int] = []
        any_new = False
        for row_index in range(len(output_ids)):
            if not active[row_index] or len(output_ids[row_index]) >= int(sampling.max_new_tokens):
                fallback = next(iter(eos), 0)
                sampled.append(int(fallback))
                if active[row_index]:
                    active[row_index] = False
                    finish[row_index] = "length"
                continue
            if greedy:
                values = logits[row_index].detach().float()
                if not bool(torch.isfinite(values).all()):
                    raise Stage4GenerationError("nonfinite_logits")
                token_id = int(torch.argmax(values, dim=-1).item())
            else:
                absolute_position = len(output_ids[row_index])
                token_id = sample_top_p_from_uniform(
                    logits[row_index],
                    uniform=counter_keys[row_index].uniform(absolute_position),  # type: ignore[index]
                    temperature=float(sampling.temperature),
                    top_p=float(sampling.top_p),
                )
            output_ids[row_index].append(int(token_id))
            sampled.append(int(token_id))
            any_new = True
            if token_id in eos:
                active[row_index] = False
                finish[row_index] = "eos"
            elif len(output_ids[row_index]) >= int(sampling.max_new_tokens):
                active[row_index] = False
                finish[row_index] = "length"
        if not any_new or not any(active):
            break
        next_ids = torch.tensor(sampled, device=logits.device, dtype=torch.long).unsqueeze(1)
        # All cache rows advance together.  Finished rows are harmless carrier
        # rows; their logits and appended carrier ids are never recorded.
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=1,
        )
        position_ids = torch.tensor(
            [[int(attention_mask[row].sum().item()) - 1] for row in range(attention_mask.shape[0])],
            device=attention_mask.device,
            dtype=torch.long,
        )
        with torch.inference_mode():
            outputs = model(
                input_ids=next_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
        logits = _output_field(outputs, "logits")[:, -1, :]
        past = _output_field(outputs, "past_key_values")
    for index, state in enumerate(finish):
        if not state:
            finish[index] = "length" if len(output_ids[index]) >= int(sampling.max_new_tokens) else "stopped"
    return output_ids, finish


def natural_generate_batch(
    model: Any,
    *,
    prompt_token_ids: Sequence[Sequence[int]],
    counter_keys: Sequence[CounterKey],
    sampling: SamplingSpec,
    pad_token_id: int,
    eos_token_ids: int | Sequence[int] | None,
    device: Any,
) -> tuple[list[list[int]], list[str]]:
    """Naturally generate A0 or A1 in a token-by-token batched loop."""

    import torch

    if len(prompt_token_ids) != len(counter_keys):
        raise Stage4GenerationError("natural_generation_batch_length_mismatch")
    input_ids, attention_mask, _lengths = left_pad_sequences(
        prompt_token_ids, pad_token_id=int(pad_token_id), device=device
    )
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=_position_ids(attention_mask),
            use_cache=True,
            return_dict=True,
        )
    return _continue_from_outputs(
        model,
        initial_outputs=outputs,
        attention_mask=attention_mask,
        initial_output_ids=[[] for _ in prompt_token_ids],
        counter_keys=counter_keys,
        sampling=sampling,
        eos_token_ids=eos_token_ids,
    )


def natural_greedy_generate_batch(
    model: Any,
    *,
    prompt_token_ids: Sequence[Sequence[int]],
    pad_token_id: int,
    eos_token_ids: int | Sequence[int] | None,
    device: Any,
    max_new_tokens: int = 2048,
) -> tuple[list[list[int]], list[str]]:
    """Natural deterministic greedy A1 generation for benign formal sets."""

    import torch

    input_ids, attention_mask, _lengths = left_pad_sequences(
        prompt_token_ids, pad_token_id=int(pad_token_id), device=device
    )
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=_position_ids(attention_mask),
            use_cache=True,
            return_dict=True,
        )
    return _continue_from_outputs(
        model,
        initial_outputs=outputs,
        attention_mask=attention_mask,
        initial_output_ids=[[] for _ in prompt_token_ids],
        counter_keys=None,
        sampling=SamplingSpec(
            temperature=0.0, top_p=1.0, max_new_tokens=int(max_new_tokens)
        ),
        eos_token_ids=eos_token_ids,
        greedy=True,
    )


def counterfactual_generate_batch(
    model: Any,
    *,
    prompt_token_ids: Sequence[Sequence[int]],
    a1_output_token_ids: Sequence[Sequence[int]],
    target_plans: Sequence[TargetPlan],
    target_names: Sequence[str],
    unit_direction: Any,
    hidden_state_index: int,
    rho: float,
    counter_keys: Sequence[CounterKey] | None,
    sampling: SamplingSpec,
    pad_token_id: int,
    eos_token_ids: int | Sequence[int] | None,
    device: Any,
    greedy: bool = False,
) -> tuple[list[list[int]], list[str], list[dict[str, Any]]]:
    """Replay A1 through the last target, intervene, then continue freely."""

    import torch

    batch_size = len(prompt_token_ids)
    if not (
        batch_size
        == len(a1_output_token_ids)
        == len(target_plans)
        == (len(counter_keys) if counter_keys is not None else batch_size)
    ):
        raise Stage4GenerationError("counterfactual_generation_batch_length_mismatch")
    names = tuple(str(item) for item in target_names)
    if len(names) != 3:
        raise Stage4GenerationError(f"exactly_three_target_names_required:{names}")
    prefixes: list[list[int]] = []
    initial_outputs: list[list[int]] = []
    absolute_positions_by_row: list[list[int]] = []
    target_ids_by_row: list[list[int]] = []
    for prompt_ids, output_ids, plan in zip(
        prompt_token_ids, a1_output_token_ids, target_plans
    ):
        if not plan.structural_valid:
            raise Stage4GenerationError(f"a1_target_plan_not_structurally_valid:{plan.missing}")
        positions, target_ids = plan.for_names(names)
        last_position = max(positions)
        output_stop = int(last_position) - len(prompt_ids) + 1
        if output_stop <= 0 or output_stop > len(output_ids):
            raise Stage4GenerationError(
                f"invalid_teacher_replay_boundary:{output_stop}:output_len={len(output_ids)}"
            )
        prefixes.append(
            [int(item) for item in prompt_ids]
            + [int(item) for item in output_ids[:output_stop]]
        )
        initial_outputs.append([int(item) for item in output_ids[:output_stop]])
        absolute_positions_by_row.append(positions)
        target_ids_by_row.append(target_ids)

    input_ids, attention_mask, prefix_lengths = left_pad_sequences(
        prefixes, pad_token_id=int(pad_token_id), device=device
    )
    width = int(input_ids.shape[1])
    target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    padded_positions_by_row: list[list[int]] = []
    for row_index, (absolute_positions, prefix_len) in enumerate(
        zip(absolute_positions_by_row, prefix_lengths)
    ):
        left_pad = width - int(prefix_len)
        padded_positions = [left_pad + int(item) for item in absolute_positions]
        if any(pos < 0 or pos >= width for pos in padded_positions):
            raise Stage4GenerationError(f"padded_target_out_of_bounds:{padded_positions}:{width}")
        for position in padded_positions:
            target_mask[row_index, position] = True
        padded_positions_by_row.append(padded_positions)

    with exact_matched_relative_hook(
        model,
        hidden_state_index=int(hidden_state_index),
        padded_target_mask=target_mask,
        unit_direction=unit_direction,
        rho=float(rho),
        target_names_by_row=[names for _ in range(batch_size)],
        target_token_ids_by_row=target_ids_by_row,
    ) as hook_stats:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=_position_ids(attention_mask),
                use_cache=True,
                return_dict=True,
            )
        if int(hook_stats["num_applied_calls"]) != 1:
            raise Stage4GenerationError(
                f"intervention_hook_application_count:{hook_stats['num_applied_calls']}"
            )
        if _output_field(outputs, "past_key_values") is None:
            raise Stage4GenerationError("prefix_forward_returned_no_cache")
        hook_stats["prefix_forward_returned_cache_after_hook"] = True
        generated, finish = _continue_from_outputs(
            model,
            initial_outputs=outputs,
            attention_mask=attention_mask,
            initial_output_ids=initial_outputs,
            counter_keys=counter_keys,
            sampling=sampling,
            eos_token_ids=eos_token_ids,
            greedy=bool(greedy),
        )
    audits: list[dict[str, Any]] = []
    for row_index in range(batch_size):
        local = dict(hook_stats["per_row"][row_index])
        local.update(
            {
                "target_positions_absolute": absolute_positions_by_row[row_index],
                "target_positions_padded": padded_positions_by_row[row_index],
                "teacher_replay_output_tokens": len(initial_outputs[row_index]),
                "hook_timing": {
                    "registered_before_prefix_forward": True,
                    "applied_on_full_prefix": True,
                    "cache_returned_after_application": True,
                    "hidden_state_index": int(hidden_state_index),
                    "decoder_block_index": int(hook_stats["decoder_block_index"]),
                    "hook_calls": hook_stats["hook_calls"],
                },
            }
        )
        audits.append(local)
    return generated, finish, audits


def counterfactual_greedy_generate_batch(
    model: Any,
    *,
    prompt_token_ids: Sequence[Sequence[int]],
    a1_output_token_ids: Sequence[Sequence[int]],
    target_plans: Sequence[TargetPlan],
    target_names: Sequence[str],
    unit_direction: Any,
    hidden_state_index: int,
    rho: float,
    pad_token_id: int,
    eos_token_ids: int | Sequence[int] | None,
    device: Any,
    max_new_tokens: int = 2048,
) -> tuple[list[list[int]], list[str], list[dict[str, Any]]]:
    """Greedy minimal-prefix A2/A3/A4 continuation for formal benign sets."""

    return counterfactual_generate_batch(
        model,
        prompt_token_ids=prompt_token_ids,
        a1_output_token_ids=a1_output_token_ids,
        target_plans=target_plans,
        target_names=target_names,
        unit_direction=unit_direction,
        hidden_state_index=int(hidden_state_index),
        rho=float(rho),
        counter_keys=None,
        sampling=SamplingSpec(
            temperature=0.0, top_p=1.0, max_new_tokens=int(max_new_tokens)
        ),
        pad_token_id=int(pad_token_id),
        eos_token_ids=eos_token_ids,
        device=device,
        greedy=True,
    )


def prefix_kv_integrity_preflight(
    model: Any,
    *,
    prompt_token_ids: Sequence[int],
    a1_output_token_ids: Sequence[int],
    target_plan: TargetPlan,
    target_names: Sequence[str],
    unit_direction: Any,
    hidden_state_index: int,
    rho: float,
    pad_token_id: int,
    device: Any,
) -> dict[str, Any]:
    """Run one no-sampling CUDA/CPU check that later-layer K/V really changes.

    The check uses the same exact teacher prefix and hook as the formal engine,
    compares it with an unmodified prefix forward, and inspects only cache
    entries *after* the hooked decoder block.  It is an execution preflight,
    never an analysis outcome and never a source of regenerated samples.
    """

    import torch

    names = tuple(str(item) for item in target_names)
    positions, target_ids = target_plan.for_names(names)
    last_position = max(positions)
    output_stop = int(last_position) - len(prompt_token_ids) + 1
    if output_stop <= 0 or output_stop > len(a1_output_token_ids):
        raise Stage4GenerationError("kv_preflight_invalid_teacher_prefix_boundary")
    prefix = [int(item) for item in prompt_token_ids] + [
        int(item) for item in a1_output_token_ids[:output_stop]
    ]
    input_ids, attention_mask, _ = left_pad_sequences(
        [prefix], pad_token_id=int(pad_token_id), device=device
    )
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for position in positions:
        mask[0, int(position)] = True
    with torch.inference_mode():
        baseline = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=_position_ids(attention_mask),
            use_cache=True,
            return_dict=True,
        )
    with exact_matched_relative_hook(
        model,
        hidden_state_index=int(hidden_state_index),
        padded_target_mask=mask,
        unit_direction=unit_direction,
        rho=float(rho),
        target_names_by_row=[names],
        target_token_ids_by_row=[target_ids],
    ) as stats:
        with torch.inference_mode():
            steered = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=_position_ids(attention_mask),
                use_cache=True,
                return_dict=True,
            )
    report = later_kv_change_report(
        _output_field(baseline, "past_key_values"),
        _output_field(steered, "past_key_values"),
        decoder_block_index=int(stats["decoder_block_index"]),
    )
    report.update(
        {
            "status": "pass" if report["pass"] else "fail",
            "hidden_state_index": int(hidden_state_index),
            "decoder_block_index": int(stats["decoder_block_index"]),
            "target_names": list(names),
            "target_positions_absolute": positions,
            "touched_token_ids": target_ids,
            "rho": float(rho),
            "num_applied_calls": int(stats["num_applied_calls"]),
            "actual_relative_norms": stats["per_row"][0]["actual_relative_norms"],
            "hook_registered_before_prefix_forward": True,
            "cache_returned_after_hook": True,
        }
    )
    if not report["pass"]:
        raise Stage4GenerationError("intervention_did_not_change_any_later_layer_kv")
    return report


def rho_zero_reference_alias(
    *,
    prompt_token_ids: Sequence[int],
    a1_output_token_ids: Sequence[int],
    a1_content_hash: str,
) -> dict[str, Any]:
    """Return an explicitly aliased alpha=0 result, guaranteed bit exact.

    Replaying a long prefix and running A1 token-by-token can take numerically
    different attention kernels even when the mathematical delta is zero.
    Therefore alpha=0 is the exact A1 reference cell, not a second generation.
    """

    expected = content_sha256(prompt_token_ids, a1_output_token_ids)
    if str(a1_content_hash) != expected:
        raise Stage4GenerationError("rho_zero_alias_a1_content_hash_mismatch")
    return {
        "output_token_ids": [int(item) for item in a1_output_token_ids],
        "generated_content_sha256": expected,
        "rho_zero_bit_exact": True,
        "rho_zero_policy": "exact_a1_reference_alias_no_forward",
        "physical_touches": 0,
    }


def assert_rho_zero_bit_exact(a1_row: Mapping[str, Any], zero_row: Mapping[str, Any]) -> None:
    fields = ("prompt_token_ids", "output_token_ids", "generated_content_sha256")
    for field in fields:
        if a1_row.get(field) != zero_row.get(field):
            raise Stage4GenerationError(f"rho_zero_not_bit_exact:{field}")


def cache_layers(past_key_values: Any) -> list[tuple[Any, Any]]:
    """Normalize legacy or DynamicCache K/V storage for integrity checks."""

    cache = past_key_values
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    if isinstance(cache, (tuple, list)):
        result = []
        for layer in cache:
            if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                result.append((layer[0], layer[1]))
        return result
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    raise Stage4GenerationError("unsupported_past_key_values_structure")


def later_kv_change_report(
    baseline_past: Any,
    steered_past: Any,
    *,
    decoder_block_index: int,
) -> dict[str, Any]:
    """Verify that at least one *later* layer cache changes after injection."""

    import torch

    baseline = cache_layers(baseline_past)
    steered = cache_layers(steered_past)
    if len(baseline) != len(steered):
        raise Stage4GenerationError("cache_layer_count_mismatch")
    later = list(range(int(decoder_block_index) + 1, len(baseline)))
    if not later:
        raise Stage4GenerationError("no_later_decoder_layer_available_for_kv_integrity")
    per_layer = []
    changed = False
    for layer_index in later:
        key_delta = float(
            (baseline[layer_index][0].float() - steered[layer_index][0].float())
            .abs()
            .max()
            .item()
        )
        value_delta = float(
            (baseline[layer_index][1].float() - steered[layer_index][1].float())
            .abs()
            .max()
            .item()
        )
        layer_changed = bool(key_delta > 0.0 or value_delta > 0.0)
        changed = changed or layer_changed
        per_layer.append(
            {
                "decoder_block_index": int(layer_index),
                "key_max_abs_delta": key_delta,
                "value_max_abs_delta": value_delta,
                "changed": layer_changed,
            }
        )
    return {"pass": bool(changed), "later_layers": per_layer}


def request_fingerprint(
    *,
    binding: Mapping[str, Any],
    source: str,
    split: str,
    prompt_id: str,
    prompt_sha256: str,
    rollout_seed: int,
    draw_index: int,
    arm: str,
    alpha: float,
) -> str:
    required_binding = {
        "model_sha256",
        "tokenizer_sha256",
        "artifact_manifest_sha256",
        "config_file_sha256",
        "config_resolved_sha256",
        "ledger_sha256",
    }
    missing = sorted(key for key in required_binding if not str(binding.get(key) or ""))
    if missing:
        raise Stage4GenerationError(f"request_binding_missing:{missing}")
    for key in required_binding:
        require_sha256(binding[key], field=f"binding.{key}")
    require_sha256(prompt_sha256, field="prompt_sha256")
    if str(binding.get("model_condition") or "") == "full_sft":
        if str(binding.get("model_hash_kind") or "") != "terminal_checkpoint_manifest_sha256":
            raise Stage4GenerationError("full_sft_model_hash_must_bind_terminal_checkpoint_manifest")
        for key in (
            "stage2_provenance_sha256",
            "terminal_checkpoint_completion_marker_sha256",
        ):
            if not str(binding.get(key) or ""):
                raise Stage4GenerationError(f"full_sft_request_binding_missing:{key}")
            require_sha256(binding[key], field=f"binding.{key}")
    return sha256_text(
        canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "binding": dict(binding),
                "source": str(source),
                "split": str(split),
                "prompt_id": str(prompt_id),
                "prompt_sha256": str(prompt_sha256),
                "rollout_seed": int(rollout_seed),
                "draw_index": int(draw_index),
                "arm": str(arm),
                "alpha": float(alpha),
            }
        )
    )


def failure_content_sha256(request_sha256: str, failure: Mapping[str, Any]) -> str:
    return sha256_text(
        canonical_json(
            {"request_sha256": str(request_sha256), "failure": dict(failure)}
        )
    )


def row_integrity_sha256(row: Mapping[str, Any]) -> str:
    fields = {
        "schema_version": row.get("schema_version"),
        "cell_id": row.get("cell_id"),
        "request_sha256": row.get("request_sha256"),
        "binding": row.get("binding"),
        "phase": row.get("phase"),
        "source": row.get("source"),
        "split": row.get("split"),
        "task": row.get("task"),
        "dataset": row.get("dataset"),
        "prompt_id": row.get("prompt_id"),
        "family_id": row.get("family_id"),
        "draw_index": row.get("draw_index"),
        "rollout_seed": row.get("rollout_seed"),
        "arm": row.get("arm"),
        "model_condition": row.get("model_condition"),
        "alpha": row.get("alpha"),
        "rho": row.get("rho"),
        "scheduled": row.get("scheduled"),
        "generated": row.get("generated"),
        "generation_status": row.get("generation_status"),
        "prompt": row.get("prompt"),
        "prompt_sha256": row.get("prompt_sha256"),
        "prompt_token_ids": row.get("prompt_token_ids"),
        "counter_random_key": row.get("counter_random_key"),
        "reference_answer": row.get("reference_answer"),
        "benchmark_metadata": row.get("benchmark_metadata"),
        "benign_ledger_manifest_sha256": row.get("benign_ledger_manifest_sha256"),
        "decoding": row.get("decoding"),
        "output_token_ids": row.get("output_token_ids"),
        "generated_content_sha256": row.get("generated_content_sha256"),
        "generated_text": row.get("generated_text"),
        "generated_text_sha256": row.get("generated_text_sha256"),
        "generated_for_judge": row.get("generated_for_judge"),
        "generated_for_judge_sha256": row.get("generated_for_judge_sha256"),
        "finish_reason": row.get("finish_reason"),
        "length_truncated": row.get("length_truncated"),
        "broken": row.get("broken"),
        "broken_diagnostics": row.get("broken_diagnostics"),
        "a1_reference_content_sha256": row.get("a1_reference_content_sha256"),
        # The alpha-zero integrity control is a scientific invariant, not
        # decorative metadata.  Bind the top-level assertion so it cannot be
        # flipped after generation without invalidating the persisted row.
        "rho_zero_bit_exact": row.get("rho_zero_bit_exact"),
        "a1_target_plan": row.get("a1_target_plan"),
        "target_resolved": row.get("target_resolved"),
        "intervention_audit": row.get("intervention_audit"),
        "failure": row.get("failure"),
        "failure_content_sha256": row.get("failure_content_sha256"),
        "resampled": row.get("resampled"),
        "regeneration_attempts": row.get("regeneration_attempts"),
    }
    return sha256_text(canonical_json(fields))


def validate_resume_row(row: Mapping[str, Any], *, expected_request_sha256: str) -> None:
    """Resume only an exactly bound generated or scheduled-failure row."""

    if str(row.get("schema_version") or "") != SCHEMA_VERSION:
        raise Stage4GenerationError("resume_schema_mismatch")
    if str(row.get("request_sha256") or "") != str(expected_request_sha256):
        raise Stage4GenerationError("resume_request_binding_mismatch")
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise Stage4GenerationError("resume_prompt_text_missing")
    if str(row.get("prompt_sha256") or "") != sha256_text(prompt):
        raise Stage4GenerationError("resume_prompt_text_hash_mismatch")
    binding = row.get("binding")
    if not isinstance(binding, Mapping):
        raise Stage4GenerationError("resume_request_binding_payload_missing")
    try:
        recomputed_request = request_fingerprint(
            binding=binding,
            source=str(row.get("source") or ""),
            split=str(row.get("split") or ""),
            prompt_id=str(row.get("prompt_id") or ""),
            prompt_sha256=str(row.get("prompt_sha256") or ""),
            rollout_seed=int(row.get("rollout_seed")),
            draw_index=int(row.get("draw_index")),
            arm=str(row.get("arm") or ""),
            alpha=float(row.get("alpha")),
        )
    except (TypeError, ValueError) as exc:
        raise Stage4GenerationError("resume_request_metadata_missing") from exc
    if recomputed_request != str(row.get("request_sha256") or ""):
        raise Stage4GenerationError("resume_request_metadata_hash_mismatch")
    status = str(row.get("generation_status") or "")
    if status in {"complete", "rho_zero_reference_alias"}:
        prompt_ids = row.get("prompt_token_ids")
        output_ids = row.get("output_token_ids")
        if not isinstance(prompt_ids, list) or not isinstance(output_ids, list):
            raise Stage4GenerationError("resume_generated_row_missing_token_ids")
        expected_content = content_sha256(prompt_ids, output_ids)
        if str(row.get("generated_content_sha256") or "") != expected_content:
            raise Stage4GenerationError("resume_generated_content_hash_mismatch")
        generated_text = row.get("generated_text")
        judge_text = row.get("generated_for_judge")
        if not isinstance(generated_text, str) or not isinstance(judge_text, str):
            raise Stage4GenerationError("resume_generated_text_missing")
        if str(row.get("generated_text_sha256") or "") != sha256_text(generated_text):
            raise Stage4GenerationError("resume_generated_text_hash_mismatch")
        if str(row.get("generated_for_judge_sha256") or "") != sha256_text(judge_text):
            raise Stage4GenerationError("resume_judge_text_hash_mismatch")
        if generated_text != judge_text:
            raise Stage4GenerationError("resume_judge_text_must_equal_exact_decoded_generation")
        if status == "rho_zero_reference_alias":
            if row.get("rho_zero_bit_exact") is not True:
                raise Stage4GenerationError("resume_rho_zero_alias_flag_missing")
            try:
                alias_alpha = float(row.get("alpha"))
                alias_rho = float(row.get("rho"))
            except (TypeError, ValueError) as exc:
                raise Stage4GenerationError(
                    "resume_rho_zero_alias_strength_missing"
                ) from exc
            if not math.isclose(
                alias_alpha, 0.0, rel_tol=0.0, abs_tol=0.0
            ) or not math.isclose(alias_rho, 0.0, rel_tol=0.0, abs_tol=0.0):
                raise Stage4GenerationError("resume_rho_zero_alias_has_nonzero_strength")
            if str(row.get("a1_reference_content_sha256") or "") != expected_content:
                raise Stage4GenerationError("resume_rho_zero_alias_reference_hash_mismatch")
            audit = row.get("intervention_audit")
            if not isinstance(audit, Mapping):
                raise Stage4GenerationError("resume_rho_zero_alias_audit_missing")
            if audit.get("rho_zero_bit_exact") is not True:
                raise Stage4GenerationError("resume_rho_zero_alias_audit_flag_missing")
            if audit.get("rho_zero_policy") != "exact_a1_reference_alias_no_forward":
                raise Stage4GenerationError("resume_rho_zero_alias_policy_mismatch")
            if int(audit.get("physical_touches", -1)) != 0:
                raise Stage4GenerationError("resume_rho_zero_alias_physical_touch_detected")
    elif status == "scheduled_failure":
        failure = row.get("failure")
        if not isinstance(failure, Mapping):
            raise Stage4GenerationError("resume_failure_row_missing_failure_payload")
        expected_failure = failure_content_sha256(expected_request_sha256, failure)
        if str(row.get("failure_content_sha256") or "") != expected_failure:
            raise Stage4GenerationError("resume_failure_content_hash_mismatch")
    else:
        raise Stage4GenerationError(f"resume_unrecognized_generation_status:{status}")
    if str(row.get("row_integrity_sha256") or "") != row_integrity_sha256(row):
        raise Stage4GenerationError("resume_row_integrity_hash_mismatch")


def repetition_diagnostics(output_token_ids: Sequence[int]) -> dict[str, Any]:
    ids = [int(item) for item in output_token_ids]
    longest_run = 0
    run = 0
    previous = None
    for token_id in ids:
        run = run + 1 if token_id == previous else 1
        previous = token_id
        longest_run = max(longest_run, run)
    grams = [tuple(ids[index : index + 4]) for index in range(max(0, len(ids) - 3))]
    unique_fraction = float(len(set(grams)) / len(grams)) if grams else 1.0
    severe_repetition = bool(longest_run >= 32 or (len(ids) >= 64 and unique_fraction < 0.20))
    return {
        "empty_output": not ids,
        "output_token_count": len(ids),
        "longest_identical_token_run": int(longest_run),
        "unique_4gram_fraction": unique_fraction,
        "severe_repetition": severe_repetition,
        "broken": bool(not ids or severe_repetition),
    }


def binding_payload(
    *,
    run_id: str,
    phase: str,
    model_condition: str,
    model_sha256: str,
    tokenizer_sha256: str,
    artifact_manifest_sha256: str,
    config_file_sha256: str,
    config_resolved_sha256: str,
    ledger_sha256: str,
    ledger_manifest_sha256: str,
    layer: int | None,
    sampling: SamplingSpec,
    norm_cap: float,
    stage2_provenance_sha256: str | None = None,
    terminal_checkpoint_completion_marker_sha256: str | None = None,
    calibration_report_sha256: str | None = None,
) -> dict[str, Any]:
    normalized_phase = str(phase)
    normalized_condition = str(model_condition)
    allowed_phases = {
        "calibration",
        "final",
        "benign_capability",
        "benign_compliance",
        "benign_semantic",
    }
    if normalized_phase not in allowed_phases:
        raise Stage4GenerationError(f"unknown_formal_generation_phase:{normalized_phase}")
    if normalized_condition not in {"original_base", "full_sft"}:
        raise Stage4GenerationError(
            f"unknown_formal_model_condition:{normalized_condition}"
        )
    if normalized_phase == "calibration" and normalized_condition != "full_sft":
        raise Stage4GenerationError("calibration_generation_requires_full_sft")
    if normalized_phase.startswith("benign_") and normalized_condition != "full_sft":
        raise Stage4GenerationError("benign_generation_requires_full_sft")
    sha_fields = {
        "model_sha256": model_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "config_file_sha256": config_file_sha256,
        "config_resolved_sha256": config_resolved_sha256,
        "ledger_sha256": ledger_sha256,
        "ledger_manifest_sha256": ledger_manifest_sha256,
    }
    normalized_sha = {
        key: require_sha256(value, field=key) for key, value in sha_fields.items()
    }
    if normalized_condition == "full_sft":
        normalized_stage2 = require_sha256(
            stage2_provenance_sha256, field="stage2_provenance_sha256"
        )
        normalized_completion = require_sha256(
            terminal_checkpoint_completion_marker_sha256,
            field="terminal_checkpoint_completion_marker_sha256",
        )
    else:
        normalized_stage2 = None
        normalized_completion = None
    needs_calibration = normalized_phase == "final" or normalized_phase.startswith(
        "benign_"
    )
    normalized_calibration = (
        require_sha256(
            calibration_report_sha256, field="calibration_report_sha256"
        )
        if needs_calibration
        else None
    )
    if not needs_calibration and calibration_report_sha256 is not None:
        raise Stage4GenerationError(
            "calibration_phase_must_not_bind_a_selected_strength_report"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(run_id),
        "phase": normalized_phase,
        "model_condition": normalized_condition,
        "model_sha256": normalized_sha["model_sha256"],
        "model_hash_kind": (
            "terminal_checkpoint_manifest_sha256"
            if normalized_condition == "full_sft"
            else "base_model_content_sha256"
        ),
        "tokenizer_sha256": normalized_sha["tokenizer_sha256"],
        "artifact_manifest_sha256": normalized_sha["artifact_manifest_sha256"],
        "config_file_sha256": normalized_sha["config_file_sha256"],
        "config_resolved_sha256": normalized_sha["config_resolved_sha256"],
        "ledger_sha256": normalized_sha["ledger_sha256"],
        "ledger_manifest_sha256": normalized_sha["ledger_manifest_sha256"],
        "hidden_state_index": int(layer) if layer is not None else None,
        "sampling": asdict(sampling),
        "counter_sampler": COUNTER_SAMPLER_VERSION,
        "norm_cap": float(norm_cap),
        "stage2_provenance_sha256": normalized_stage2,
        "terminal_checkpoint_completion_marker_sha256": normalized_completion,
        "calibration_report_sha256": normalized_calibration,
        "forced_pause": False,
        "pause_suppression": False,
        "fsm": False,
        "projection_clamp": False,
        "safe_centroid": False,
        "lora": False,
    }


def tokenizer_content_fingerprint(tokenizer: Any) -> dict[str, Any]:
    """Stable generic tokenizer hash (no SFT-only pause-token assumption)."""

    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, Mapping) or not vocab:
        raise Stage4GenerationError("tokenizer_get_vocab_returned_no_vocabulary")
    vocab_rows = sorted((str(token), int(token_id)) for token, token_id in vocab.items())
    special_map = {
        str(key): str(value)
        for key, value in dict(getattr(tokenizer, "special_tokens_map", {}) or {}).items()
    }
    payload = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
        "vocab": vocab_rows,
        "special_tokens_map": special_map,
        "model_max_length": int(getattr(tokenizer, "model_max_length", 0)),
        "padding_side": str(getattr(tokenizer, "padding_side", "")),
        "truncation_side": str(getattr(tokenizer, "truncation_side", "")),
    }
    chat_template = getattr(tokenizer, "chat_template", None)
    chat_text = "" if chat_template is None else str(chat_template)
    core_sha256 = sha256_text(canonical_json(payload))
    chat_sha256 = sha256_text(chat_text)
    return {
        # A0 has no Stage2 provenance envelope, so its single declared
        # tokenizer hash covers both the tokenizer core and chat template.
        "sha256": sha256_text(
            canonical_json(
                {
                    "core_sha256": core_sha256,
                    "chat_template_sha256": chat_sha256,
                }
            )
        ),
        # This exactly matches Stage2 tokenizer_provenance.sha256 and allows
        # SFT to cross-check the historical two-field envelope.
        "stage2_core_sha256": core_sha256,
        "chat_template_sha256": chat_sha256,
        "chat_template_present": chat_template is not None,
        "vocabulary_size": len(vocab_rows),
        "class": payload["class"],
    }


def terminal_checkpoint_binding_from_provenance(
    provenance: Mapping[str, Any],
    sealed_checkpoint: Mapping[str, Any],
    *,
    expected_step: int = 1064,
) -> dict[str, Any]:
    """Cross-bind raw Stage2 provenance to its sealed terminal checkpoint.

    ``provenance.model.sha256`` is the immutable *base-model directory* hash.
    It must never be reused as the trained model identity.  The runtime SFT
    identity is the SHA-256 of checkpoint-1064's payload manifest.
    """

    checkpoints = provenance.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise Stage4GenerationError("stage2_provenance_checkpoints_missing")
    terminal = [
        row
        for row in checkpoints
        if isinstance(row, Mapping) and int(row.get("step", -1)) == int(expected_step)
    ]
    if len(terminal) != 1:
        raise Stage4GenerationError(
            f"stage2_provenance_requires_one_terminal_checkpoint:{len(terminal)}"
        )
    record = dict(terminal[0])
    manifest_sha = str(record.get("manifest_sha256") or "")
    sealed_manifest_sha = str(sealed_checkpoint.get("manifest_sha256") or "")
    if not manifest_sha or manifest_sha != sealed_manifest_sha:
        raise Stage4GenerationError(
            "stage2_provenance_terminal_manifest_does_not_match_checkpoint_seal"
        )
    if int(sealed_checkpoint.get("global_step", -1)) != int(expected_step):
        raise Stage4GenerationError("sealed_terminal_checkpoint_step_mismatch")
    expected_name = f"checkpoint-{int(expected_step)}"
    sealed_name = str(sealed_checkpoint.get("checkpoint_name") or "")
    if sealed_name != expected_name:
        raise Stage4GenerationError(
            f"sealed_terminal_checkpoint_name_mismatch:{sealed_name}!={expected_name}"
        )
    completion_sha = str(sealed_checkpoint.get("completion_marker_sha256") or "")
    if not completion_sha:
        raise Stage4GenerationError("sealed_terminal_completion_marker_hash_missing")
    recorded_completion_sha = str(record.get("completion_marker_sha256") or "")
    if not recorded_completion_sha or recorded_completion_sha != completion_sha:
        raise Stage4GenerationError(
            "stage2_provenance_terminal_completion_marker_does_not_match_checkpoint_seal"
        )
    model = provenance.get("model")
    if not isinstance(model, Mapping) or not str(model.get("sha256") or ""):
        raise Stage4GenerationError("stage2_base_model_hash_missing")
    return {
        "sha256": manifest_sha,
        "binding_kind": "terminal_checkpoint_manifest_sha256",
        "base_model_sha256": str(model["sha256"]),
        "terminal_checkpoint": {
            "step": int(expected_step),
            "name": sealed_name,
            "manifest_sha256": manifest_sha,
            "completion_marker_sha256": completion_sha,
            "payload_bytes": int(sealed_checkpoint.get("payload_bytes", 0)),
            "files": record.get("files"),
        },
    }


__all__ = [
    "COUNTER_SAMPLER_VERSION",
    "CounterKey",
    "FORMAL_TARGET_NAMES",
    "SCHEMA_VERSION",
    "SamplingSpec",
    "Stage4GenerationError",
    "TargetPlan",
    "assert_rho_zero_bit_exact",
    "binding_payload",
    "cache_layers",
    "canonical_json",
    "content_sha256",
    "counter_uniform",
    "counterfactual_generate_batch",
    "counterfactual_greedy_generate_batch",
    "exact_matched_relative_hook",
    "failure_content_sha256",
    "get_decoder_layers",
    "hidden_index_to_block_index",
    "later_kv_change_report",
    "natural_generate_batch",
    "natural_greedy_generate_batch",
    "prefix_kv_integrity_preflight",
    "repetition_diagnostics",
    "request_fingerprint",
    "require_sha256",
    "resolve_a1_target_plan",
    "rho_zero_reference_alias",
    "row_integrity_sha256",
    "sample_top_p_from_uniform",
    "sha256_bytes",
    "sha256_text",
    "stable_rollout_seed",
    "stable_shard",
    "terminal_checkpoint_binding_from_provenance",
    "tokenizer_content_fingerprint",
    "validate_resume_row",
]
