from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any


DEEPSEEK_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
DEEPSEEK_USER_TEMPLATE = "<｜User｜>"
DEEPSEEK_ASSISTANT_TEMPLATE = "<｜Assistant｜>"
DEFAULT_FORCED_PREFIX = "<think>\n"
PAUSE_TOKEN = "<|pause|>"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def first_label(row: dict[str, Any]) -> int | None:
    for field in ("binary_safety_label", "trajectory_safety_label", "safety_label", "label", "target"):
        value = row.get(field)
        if value is None:
            continue
        text = clean_text(value).lower()
        if text in {"0", "safe", "benign", "harmless", "refusal", "safe_refusal"}:
            return 0
        if text in {"1", "unsafe", "harmful", "unsafe_valid", "compliance"}:
            return 1
    return None


def prompt_from_row(row: dict[str, Any]) -> str:
    return clean_text(first_present(row, ("input", "prompt", "question", "query", "behavior", "goal")))


def build_prompt(prompt: str, forced_prefix: str = DEFAULT_FORCED_PREFIX) -> str:
    return f"{DEEPSEEK_BOS_TOKEN}{DEEPSEEK_USER_TEMPLATE}{prompt}{DEEPSEEK_ASSISTANT_TEMPLATE}{forced_prefix}"


def _resolve_path(value: Any, *, repo_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def sample_liveness_prompts(config: dict[str, Any], *, repo_root: Path, limit: int, seed: int) -> list[str]:
    eval_cfg = config.get("eval", {})
    specs = eval_cfg.get("dataset_specs") or []
    rng = random.Random(seed)
    prompts: list[str] = []
    for spec in specs:
        if not isinstance(spec, dict) or not spec.get("input_file"):
            continue
        path = _resolve_path(spec["input_file"], repo_root=repo_root)
        if not path.exists():
            continue
        label_filter = str(spec.get("label_filter", "all"))
        rows_per_label = int(spec.get("rows_per_label", max(1, limit)))
        rows = read_jsonl(path)
        buckets: dict[int, list[str]] = {0: [], 1: []}
        unlabeled: list[str] = []
        for row in rows:
            prompt = prompt_from_row(row)
            if not prompt:
                continue
            label = first_label(row)
            if label in buckets:
                buckets[label].append(prompt)
            else:
                unlabeled.append(prompt)
        wanted = {"all": (0, 1), "safe": (0,), "unsafe": (1,)}.get(label_filter, (0, 1))
        for label in wanted:
            bucket = list(buckets[label])
            rng.shuffle(bucket)
            prompts.extend(bucket[: min(rows_per_label, len(bucket))])
        if not prompts and unlabeled:
            rng.shuffle(unlabeled)
            prompts.extend(unlabeled[: min(rows_per_label, len(unlabeled))])
        if len(prompts) >= limit:
            break
    if len(prompts) > limit:
        prompts = prompts[:limit]
    if not prompts:
        raise FileNotFoundError("No liveness prompts found from eval.dataset_specs input files.")
    return prompts


def dtype_from_name(torch: Any, name: str) -> Any:
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    return "auto"


def get_transformer_layers(model: Any) -> Any:
    candidates = [
        ("model", "layers"),
        ("model", "model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in candidates:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise ValueError("Could not find transformer block list on model.")


def layer_to_block_index(layer_id: int) -> int:
    if layer_id <= 0:
        raise ValueError("HF hidden-state layer ids must be >= 1.")
    return layer_id - 1


def target_pause_ordinals(target_positions: list[str]) -> set[int]:
    return {int(item.removeprefix("pause_")) for item in target_positions if item.startswith("pause_")}


def make_position_masks(input_ids: Any, attention_mask: Any, *, pause_id: int, target_positions: list[str]) -> dict[str, Any]:
    import torch

    pause_mask = input_ids.eq(pause_id) & attention_mask.bool()
    if target_positions:
        ordinals = target_pause_ordinals(target_positions)
        pause_ordinals = pause_mask.cumsum(dim=1) - 1
        target_mask = torch.zeros_like(pause_mask)
        for ordinal in ordinals:
            target_mask |= pause_mask & pause_ordinals.eq(ordinal)
        pause_mask = target_mask
    content_mask = torch.zeros_like(pause_mask)
    bos_mask = torch.zeros_like(pause_mask)
    for row_idx in range(input_ids.shape[0]):
        valid = torch.flatnonzero(attention_mask[row_idx].bool())
        if valid.numel() == 0:
            continue
        bos_mask[row_idx, int(valid[0].item())] = True
        pauses = torch.flatnonzero(pause_mask[row_idx])
        if pauses.numel() == 0:
            continue
        first_pause = int(pauses[0].item())
        count = int(pauses.numel())
        start = max(int(valid[0].item()), first_pause - count)
        content_mask[row_idx, start:first_pause] = True
    return {"pause": pause_mask, "content": content_mask, "bos": bos_mask}


def normalize_direction(torch: Any, direction: Any, *, hidden_size: int, device: Any, dtype: Any) -> Any:
    if direction is None:
        direction = torch.randn(hidden_size, device=device, dtype=dtype)
    direction = direction.to(device=device, dtype=dtype).flatten()
    return direction / direction.norm().clamp_min(1e-12)


def final_token_kl(torch: Any, base_logits: Any, perturbed_logits: Any, attention_mask: Any) -> float:
    positions = attention_mask.shape[1] - 1 - torch.argmax(torch.flip(attention_mask.long(), dims=[1]), dim=1)
    rows = torch.arange(base_logits.shape[0], device=base_logits.device)
    base = base_logits[rows, positions, :].float()
    perturbed = perturbed_logits[rows, positions, :].float()
    base_logp = torch.log_softmax(base, dim=-1)
    perturbed_logp = torch.log_softmax(perturbed, dim=-1)
    base_p = torch.softmax(base, dim=-1)
    kl = (base_p * (base_logp - perturbed_logp)).sum(dim=-1)
    return float(kl.mean().detach().cpu().item())


def forward_with_injection(
    model: Any,
    layers: Any,
    encoded: dict[str, Any],
    *,
    layer: int,
    mask: Any,
    direction: Any,
    epsilon_multiplier: float,
    epsilon_base: float,
) -> Any:
    import torch

    block_idx = layer_to_block_index(layer)

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[:2] != mask.shape:
            return output
        selected = hidden[mask]
        if selected.numel() == 0:
            return output
        step_norm = float(epsilon_multiplier) * float(epsilon_base) * selected.float().norm(dim=-1).mean().clamp_min(1e-12)
        step = normalize_direction(
            torch,
            direction,
            hidden_size=hidden.shape[-1],
            device=hidden.device,
            dtype=hidden.dtype,
        ) * step_norm.to(device=hidden.device, dtype=hidden.dtype)
        edited = hidden.clone()
        edited[mask] = edited[mask] + step
        if isinstance(output, tuple):
            return (edited,) + output[1:]
        return edited

    handle = layers[block_idx].register_forward_hook(hook)
    try:
        return model(**encoded, use_cache=False)
    finally:
        handle.remove()


def build_liveness_prefixes(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    insert_pause_after_cot_tokens: int,
    n_insert_pauses: int,
    max_input_length: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> list[str]:
    import torch
    from transformers import set_seed

    base_prompts = [build_prompt(prompt) for prompt in prompts]
    if insert_pause_after_cot_tokens <= 0:
        return [prompt + (PAUSE_TOKEN * n_insert_pauses) for prompt in base_prompts]
    encoded = tokenizer(
        base_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
        add_special_tokens=False,
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_width = encoded["input_ids"].shape[1]
    set_seed(seed)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=insert_pause_after_cot_tokens,
            min_new_tokens=insert_pause_after_cot_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    prefixes = [
        tokenizer.decode(item[prompt_width:], skip_special_tokens=False)
        for item in generated
    ]
    return [base + prefix + (PAUSE_TOKEN * n_insert_pauses) for base, prefix in zip(base_prompts, prefixes)]


def injection_gain_metric(
    model: Any,
    tokenizer: Any,
    layers: Any,
    prefixes: list[str],
    *,
    pause_id: int,
    layer: int,
    target_positions: list[str],
    epsilon_multipliers: list[float],
    epsilon_base: float,
    batch_size: int,
    max_input_length: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    totals = {"pause": [], "content": [], "bos": []}
    hidden_size = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "n_embd", 0))
    if hidden_size <= 0:
        raise ValueError("Could not determine model hidden size for liveness injection directions.")
    direction = torch.randn(hidden_size, generator=generator, device=device)
    for start in range(0, len(prefixes), batch_size):
        batch = prefixes[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
            add_special_tokens=False,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        masks = make_position_masks(
            encoded["input_ids"],
            encoded["attention_mask"],
            pause_id=pause_id,
            target_positions=target_positions,
        )
        with torch.no_grad():
            base = model(**encoded, use_cache=False)
            for name, mask in masks.items():
                if not bool(mask.any()):
                    continue
                for epsilon in epsilon_multipliers:
                    perturbed = forward_with_injection(
                        model,
                        layers,
                        encoded,
                        layer=layer,
                        mask=mask,
                        direction=direction,
                        epsilon_multiplier=float(epsilon),
                        epsilon_base=epsilon_base,
                    )
                    kl = final_token_kl(torch, base.logits, perturbed.logits, encoded["attention_mask"])
                    totals[name].append(kl / max(float(epsilon), 1e-12))
    slopes = {key: (sum(values) / len(values) if values else 0.0) for key, values in totals.items()}
    pause = slopes["pause"]
    content = slopes["content"]
    bos = slopes["bos"]
    return {
        "status": "measured",
        "layer": layer,
        "target_positions": target_positions,
        "epsilon_multipliers": epsilon_multipliers,
        "epsilon_base": epsilon_base,
        "slopes": slopes,
        "pause_vs_content_gain": pause / max(content, 1e-12),
        "pause_vs_bos_gain": pause / max(bos, 1e-12),
        "n_prompts": len(prefixes),
    }


def attention_mass_metric(
    model: Any,
    tokenizer: Any,
    prefixes: list[str],
    *,
    pause_id: int,
    target_positions: list[str],
    batch_size: int,
    max_input_length: int,
) -> dict[str, Any]:
    import torch

    device = next(model.parameters()).device
    pause_masses = []
    content_masses = []
    for start in range(0, len(prefixes), batch_size):
        batch = prefixes[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
            add_special_tokens=False,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        masks = make_position_masks(
            encoded["input_ids"],
            encoded["attention_mask"],
            pause_id=pause_id,
            target_positions=target_positions,
        )
        with torch.no_grad():
            out = model(**encoded, output_attentions=True, use_cache=False)
        attentions = out.attentions
        if not attentions:
            return {"status": "incomplete", "reason": "model did not return attentions"}
        last_attn = attentions[-1].float().mean(dim=1)
        last_query = encoded["attention_mask"].shape[1] - 1 - torch.argmax(
            torch.flip(encoded["attention_mask"].long(), dims=[1]),
            dim=1,
        )
        for row_idx in range(encoded["input_ids"].shape[0]):
            query = int(last_query[row_idx].item())
            row_attn = last_attn[row_idx, query, :]
            pause_mass = float(row_attn[masks["pause"][row_idx]].sum().detach().cpu().item())
            content_mass = float(row_attn[masks["content"][row_idx]].sum().detach().cpu().item())
            pause_masses.append(pause_mass)
            content_masses.append(content_mass)
    pause_mean = sum(pause_masses) / max(1, len(pause_masses))
    content_mean = sum(content_masses) / max(1, len(content_masses))
    ratio = pause_mean / max(content_mean, 1e-12)
    status = "green" if pause_mean > 0.0 and ratio >= 0.25 else "yellow" if pause_mean > 0.0 else "red"
    return {
        "status": status,
        "pause_attention_mass": pause_mean,
        "content_attention_mass": content_mean,
        "pause_vs_content_attention_ratio": ratio,
        "n_prompts": len(prefixes),
    }


def run_model_liveness(
    config: dict[str, Any],
    *,
    model_path: str,
    repo_root: Path,
    prompt_limit: int | None = None,
    batch_size: int = 1,
    tests: list[str] | None = None,
    seed: int = 260704,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on GPU env.
        raise SystemExit("Stage4 liveness kernels require torch and transformers.") from exc

    runtime = config.get("runtime", {})
    model_cfg = config.get("model", {})
    steering = config.get("steering", {})
    liveness = config.get("liveness", {})
    tests = tests or [str(item) for item in liveness.get("tests", [])]
    prompt_limit = int(prompt_limit or liveness.get("num_prompts", 200))
    max_input_length = int(model_cfg.get("max_input_length", model_cfg.get("max_length", 4096)))
    torch_dtype = dtype_from_name(torch, str(runtime.get("torch_dtype", "bfloat16")))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=bool(model_cfg.get("trust_remote_code", False)))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    pause_ids = tokenizer(PAUSE_TOKEN, add_special_tokens=False).input_ids
    if len(pause_ids) != 1:
        raise SystemExit(f"Expected one-token pause id for liveness model {model_path}, got {pause_ids}")
    model_kwargs: dict[str, Any] = {"trust_remote_code": bool(model_cfg.get("trust_remote_code", False))}
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype
    device_map = liveness.get("device_map")
    if device_map:
        model_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if not device_map:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    model.eval()
    layers = get_transformer_layers(model)
    prompts = sample_liveness_prompts(config, repo_root=repo_root, limit=prompt_limit, seed=seed)
    prefixes = build_liveness_prefixes(
        model,
        tokenizer,
        prompts,
        insert_pause_after_cot_tokens=int(steering.get("insert_pause_after_cot_tokens", 3)),
        n_insert_pauses=int(steering.get("n_insert_pauses", 3)),
        max_input_length=max_input_length,
        temperature=float(liveness.get("temperature", 0.7)),
        top_p=float(liveness.get("top_p", 0.95)),
        seed=seed,
    )
    target_positions = [str(item) for item in steering.get("target_positions", ["pause_0", "pause_1", "pause_2"])]
    layer = int(steering.get("layer", (liveness.get("layers") or [14])[0]))
    metrics: dict[str, Any] = {}
    if "injection_gain" in tests:
        metrics["injection_gain"] = injection_gain_metric(
            model,
            tokenizer,
            layers,
            prefixes,
            pause_id=int(pause_ids[0]),
            layer=layer,
            target_positions=target_positions,
            epsilon_multipliers=[float(item) for item in liveness.get("epsilon_multipliers", [1.0, 2.0, 4.0])],
            epsilon_base=float(liveness.get("epsilon_base", 0.05)),
            batch_size=batch_size,
            max_input_length=max_input_length,
            seed=seed,
        )
    if "attention_mass" in tests:
        metrics["attention_mass"] = attention_mass_metric(
            model,
            tokenizer,
            prefixes,
            pause_id=int(pause_ids[0]),
            target_positions=target_positions,
            batch_size=batch_size,
            max_input_length=max_input_length,
        )
    for name in tests:
        if name not in metrics:
            metrics[name] = {"status": "incomplete", "reason": "kernel_not_implemented"}
    return {
        "model_under_test": model_path,
        "metrics": metrics,
        "prompt_count": len(prompts),
        "prefix_count": len(prefixes),
        "implemented_tests": [name for name in ("injection_gain", "attention_mass") if name in tests],
        "incomplete_tests": [name for name in tests if metrics.get(name, {}).get("status") == "incomplete"],
    }


def run_liveness_battery(
    config: dict[str, Any],
    *,
    repo_root: Path,
    prompt_limit: int | None = None,
    batch_size: int = 1,
    tests: list[str] | None = None,
    seed: int = 260704,
    include_positive_control: bool = True,
) -> dict[str, Any]:
    from cot_safety.steering.liveness import liveness_config, liveness_decision

    liveness = liveness_config(config)
    model_path = str(liveness.get("model_under_test") or "")
    if not model_path:
        raise ValueError("No model_under_test resolved for liveness battery.")
    report = run_model_liveness(
        config,
        model_path=model_path,
        repo_root=repo_root,
        prompt_limit=prompt_limit,
        batch_size=batch_size,
        tests=tests,
        seed=seed,
    )
    required_tests = [str(item) for item in liveness.get("tests", [])]
    report["decision"] = liveness_decision(report, required_tests=required_tests, gate=liveness.get("gate") or {})
    if include_positive_control and liveness.get("gate", {}).get("require_positive_control_green", True):
        control_status = str(liveness.get("positive_control_status") or "").lower()
        control_model = str(liveness.get("positive_control_model") or "")
        if control_model and not control_status.startswith(("missing", "invalid")):
            positive = run_model_liveness(
                config,
                model_path=control_model,
                repo_root=repo_root,
                prompt_limit=prompt_limit,
                batch_size=batch_size,
                tests=tests,
                seed=seed + 17,
            )
            positive["decision"] = liveness_decision(
                positive,
                required_tests=required_tests,
                gate=liveness.get("gate") or {},
            )
            report["positive_control"] = positive
        else:
            report["positive_control"] = {
                "decision": "missing",
                "model_under_test": control_model,
                "configured_positive_control_status": liveness.get("positive_control_status"),
            }
    return report
