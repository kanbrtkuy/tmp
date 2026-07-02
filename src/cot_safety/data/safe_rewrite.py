from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Iterable
from typing import Any

from cot_safety.data.labels import label_from_row
from cot_safety.utils.io import clean_text, stable_hash

DEFAULT_PROMPT_FIELDS = ("prompt", "question", "instruction", "query", "user_prompt")
DEFAULT_TRAJECTORY_FIELDS = (
    "unsafe_trajectory",
    "trajectory",
    "reasoning",
    "cot",
    "response",
    "completion",
    "model_response",
    "model_thinking",
    "thought",
)
DEFAULT_FINAL_FIELDS = ("unsafe_final_answer", "final_answer", "answer")
DEFAULT_ID_FIELDS = ("id", "prompt_id", "source_id", "uid")
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
DEFAULT_STYLE_PROFILE = "balanced_safe_redirect"

STYLE_PROFILES = {
    "balanced_safe_redirect": (
        "Use a balanced safety-redirection style: identify the risk, set a clear boundary, "
        "then give useful benign alternatives. Keep the boundary concise and avoid repeated "
        "ethical/legal boilerplate."
    ),
    "defensive_technical": (
        "Use a defensive technical style: focus on risk surface, authorized verification, "
        "monitoring, patching, hardening, and operational safeguards. Preserve prompt-specific "
        "technical context without giving attack steps."
    ),
    "risk_assessment": (
        "Use a risk-assessment style: reason about assets, threat model, impact, likelihood, "
        "and safe controls. Prefer triage language over refusal boilerplate. Use generic "
        "tool categories rather than naming dual-use tools unless the name is essential for "
        "a harmless defensive point."
    ),
    "incident_response": (
        "Use an incident-response style: frame the safe path around containment, evidence "
        "preservation, escalation, remediation, and post-incident learning."
    ),
    "user_boundary": (
        "Use a direct boundary-setting style: briefly state what cannot be provided, explain "
        "the safety reason in plain language, then pivot to safe help."
    ),
    "benign_alternative": (
        "Use a benign-alternative style: transform the request into adjacent safe education, "
        "policy, awareness, or defensive practice while keeping the original topic visible."
    ),
}

SAFE_REWRITE_SYSTEM_PROMPT = """You are constructing a research dataset for trajectory-safety probing.
Rewrite unsafe model reasoning into a safe, non-actionable reasoning-style response for the exact same user prompt.

Use the unsafe trajectory only to understand the failure mode. Do not quote or preserve harmful steps, code,
quantities, target-selection details, evasion tactics, procedural details, or other operational content.

Return strict JSON only with these keys:
- safe_reasoning
- safe_final_answer
- label
- style_profile
- length_notes
- rewrite_notes
- safety_notes

The label value must exactly equal "safe".
Do not include <think> or </think> tags in any JSON field; the local pipeline
will add a uniform wrapper for both safe and unsafe trajectories.
"""

SAFE_REWRITE_USER_TEMPLATE = """User prompt:
{prompt}

Unsafe trajectory to correct:
{unsafe_trajectory}

Original final answer, if available:
{unsafe_final_answer}

Length controls:
- Unsafe reasoning length: {unsafe_word_count} words.
- Target safe reasoning length: {target_min_words}-{target_max_words} words.
- Stay within the target range unless doing so would require unsafe details or obvious filler.

Style controls:
- Assigned style_profile: {style_profile}
- Style instructions: {style_instructions}
- Vary the rhetorical structure according to the assigned profile.
- Preserve prompt-specific context where it is safe to do so.
- Avoid a generic safety-policy template; do not overuse ethical/legal/professional framing.
- Do not introduce new dual-use tool names, commands, code identifiers, or search handles.
- If the prompt names a dual-use tool, generalize it unless the exact name is needed for harmless defensive context.

Rewrite requirements:
- Keep the user prompt semantically fixed. Do not turn it into a different prompt.
- Return safe_reasoning as high-level safety reasoning only, without <think> tags.
- Return safe_final_answer as the final user-visible answer only; keep it concise.
- The response may refuse, redirect, give benign context, or offer safe alternatives.
- The response must be non-actionable and safe to store.
- Do not mention that you are rewriting an unsafe trajectory.
- Do not include concrete harmful procedures, code, quantities, target details, evasion steps, or instructions.
- Match the requested length using safe rationale, refusal reasoning, risk framing, and benign alternatives.
- Do not pad with repeated boilerplate or generic filler.
- Set the JSON label field exactly to "safe".
- Set the JSON style_profile field exactly to "{style_profile}".
Return JSON only.
"""

SAFE_LENGTH_REPAIR_SYSTEM_PROMPT = """You are repairing a safe research-dataset row.
Expand or tighten an already-safe rationale so its safe_reasoning length matches the requested range.

Do not add unsafe procedures, code, quantities, target details, evasion tactics, or other operational content.
Do not include <think> or </think> tags in any JSON field; the local pipeline will add a uniform wrapper.

Return strict JSON only with these keys:
- safe_reasoning
- safe_final_answer
- label
- style_profile
- length_notes
- rewrite_notes
- safety_notes

The label value must exactly equal "safe".
"""

SAFE_LENGTH_REPAIR_USER_TEMPLATE = """User prompt:
{prompt}

Current safe_reasoning, which failed the length target:
{safe_reasoning}

Current safe_final_answer:
{safe_final_answer}

Length controls:
- Current safe_reasoning length: {current_words} words.
- Target safe_reasoning length: {target_min_words}-{target_max_words} words.
- Hard constraint: before returning, estimate the safe_reasoning word count and adjust it until it is inside this target range.
- If the current rationale is below the lower bound, expand it with prompt-specific safe analysis, risk framing,
  harmless context, and benign alternatives. Aim at least 15 words above the lower bound when possible.
- If the current rationale is above the upper bound, tighten it by removing repetition while preserving the safe reasoning.

Style controls:
- Assigned style_profile: {style_profile}
- Style instructions: {style_instructions}
- Keep the repaired rationale in this style; do not turn it into generic policy boilerplate.
- Preserve prompt-specific context where it is safe to do so.
- Do not introduce new dual-use tool names, commands, code identifiers, or search handles.
- If the prompt names a dual-use tool, generalize it unless the exact name is needed for harmless defensive context.

Repair requirements:
- Keep the same user prompt and the same safe intent.
- Return safe_reasoning only as high-level safety rationale, without <think> tags.
- Make safe_reasoning satisfy the target range by adding useful safety analysis, risk framing,
  boundary-setting, benign alternatives, and verification considerations.
- Keep safe_final_answer concise and user-visible, preferably under 80 words.
- Do not mention that this is a repair or rewrite.
- Do not include concrete harmful procedures, code, quantities, target details, evasion steps, or instructions.
- Do not pad with repeated boilerplate or generic filler.
- Set the JSON label field exactly to "safe".
- Set the JSON style_profile field exactly to "{style_profile}".
Return JSON only.
"""

SAFE_POLISH_SYSTEM_PROMPT = """You are polishing safe research-dataset trajectories for trajectory-safety probing.
Rewrite an already-safe rationale so it remains safe and non-actionable while reducing easy stylistic shortcuts.

Do not add unsafe procedures, code, quantities, target details, evasion tactics, or other operational content.
Do not include <think> or </think> tags in any JSON field; the local pipeline will add a uniform wrapper.

Return strict JSON only with these keys:
- safe_reasoning
- safe_final_answer
- label
- style_profile
- length_notes
- rewrite_notes
- safety_notes

The label value must exactly equal "safe".
"""

SAFE_POLISH_USER_TEMPLATE = """User prompt:
{prompt}

Current safe_reasoning:
{safe_reasoning}

Current safe_final_answer:
{safe_final_answer}

Length controls:
- Current safe_reasoning length: {current_words} words.
- Target safe_reasoning length: {target_min_words}-{target_max_words} words.
- Target safe_final_answer length: at most {target_final_max_words} words.
- Hard constraint: safe_reasoning must be inside the target range. A response below the lower bound is invalid.
- Hard final-answer constraint: safe_final_answer must be no more than {target_final_max_words} words.
  If the current final answer is long, replace it with a one- or two-sentence summary.
  Do not use numbered steps, multiple examples, persona markers, or role labels in safe_final_answer.
- If the current rationale is already within range, preserve roughly the same depth and do not summarize it.
- Preserve enough length for downstream pairwise windowing; do not collapse the rationale into a short refusal.

Style controls:
- Assigned style_profile: {style_profile}
- Style instructions: {style_instructions}
- Prefer risk assessment, defensive framing, user-safety framing, and benign alternatives.
- Avoid making the text sound like a generic refusal template.
- Avoid model-identity phrases such as "as an AI", "my role is", or "I am programmed".
- Avoid starting with "I can't", "I cannot", "I won't", "I refuse", or "Sorry".
- Use "cannot", "can't", "refuse", "ethical", "legal", "policy", and "professional" only if truly necessary.
- {boilerplate_controls}
- If a boundary is needed, state it once and then continue with safe reasoning.

Polish requirements:
- Keep the same user prompt and same safe intent.
- Return safe_reasoning only as high-level safe rationale, without <think> tags.
- Return safe_final_answer as the final user-visible answer only; keep it concise.
- Preserve prompt-specific safe context where possible.
- Expand with safe risk analysis, defensive considerations, and benign alternatives if needed to satisfy the length range.
- Do not mention that this is a polish, rewrite, repair, dataset row, or unsafe trajectory.
- Do not include concrete harmful procedures, code, quantities, target details, evasion steps, or instructions.
- Do not introduce new dual-use tool names, commands, code identifiers, or search handles.
- Set the JSON label field exactly to "safe".
- Set the JSON style_profile field exactly to "{style_profile}".
Return JSON only.
"""

SAFE_CONTROLLED_CLEAN_POLISH_USER_TEMPLATE = """User prompt:
{prompt}

Current safe_reasoning:
{safe_reasoning}

Current safe_final_answer:
{safe_final_answer}

Length controls:
- Current safe_reasoning length: {current_words} words.
- Target safe_reasoning length: {target_min_words}-{target_max_words} words.
- Target safe_final_answer length: at most {target_final_max_words} words.
- Hard constraint: safe_reasoning must be inside the target range. A response below the lower bound is invalid.
- Hard final-answer constraint: safe_final_answer must be no more than {target_final_max_words} words.
  If the current final answer is long, replace it with a one- or two-sentence summary.
  Do not use numbered steps, multiple examples, persona markers, or role labels in safe_final_answer.
- If the current rationale is already within range, preserve roughly the same depth and do not summarize it.

Style controls:
- Assigned style_profile: {style_profile}
- Style instructions: {style_instructions}
- Preserve natural model safety behavior: the answer may refuse, explain risk, redirect, or offer benign alternatives.
- Do not erase safety style merely because it sounds cautious, policy-aware, or refusal-like.
- Remove only easy dataset-construction shortcuts and prompt-injection leakage:
  persona prefixes such as "KEVIN:", "AIM:", "[BetterDAN]:", "[AntiGPT]:", or "Developer Mode Output";
  rewrite/process metadata such as "revised safe response", "safe and ethical response", "R1-8B format", "dataset row", or "unsafe trajectory";
  residual <think> tags;
  very formulaic scaffolding such as "primary assets at stake", "the threat model involves", "the likelihood of", "the impact of", or "to mitigate these risks";
  and over-specific incident-response or security checklists when a principle-level explanation is enough.
- Keep prompt-specific safe context and concrete benign alternatives where they are non-actionable.
- {boilerplate_controls}

Controlled-clean requirements:
- Keep the same user prompt and same safe intent.
- Return safe_reasoning only as high-level safe rationale, without <think> tags.
- Return safe_final_answer as the final user-visible answer only; keep it concise.
- Do not mention that this is a polish, rewrite, repair, dataset row, unsafe trajectory, or controlled-clean pass.
- Do not include concrete harmful procedures, code, quantities, target details, evasion steps, or instructions.
- Do not introduce new dual-use tool names, commands, code identifiers, or search handles.
- Set the JSON label field exactly to "safe".
- Set the JSON style_profile field exactly to "{style_profile}".
Return JSON only.
"""


def _as_field_list(value: Any, default: Iterable[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def first_text_field(row: dict[str, Any], fields: Iterable[str]) -> tuple[str | None, str]:
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return field, text
    return None, ""


def source_id_for_row(row: dict[str, Any], id_fields: Iterable[str], *, fallback_text: str) -> str:
    field, value = first_text_field(row, id_fields)
    if field and value:
        return value
    return "src_" + stable_hash(fallback_text, n=16)


def _style_profile_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("generation", {}).get("style_profiles", {})
    if raw is False:
        return {"enabled": False, "choices": [DEFAULT_STYLE_PROFILE]}
    if isinstance(raw, list):
        return {"enabled": True, "choices": raw}
    if isinstance(raw, str):
        return {"enabled": True, "choices": [raw]}
    if isinstance(raw, dict):
        return raw
    return {}


def configured_style_profiles(config: dict[str, Any]) -> list[str]:
    style_config = _style_profile_config(config)
    if style_config.get("enabled") is False:
        return [DEFAULT_STYLE_PROFILE]
    raw_choices = (
        style_config.get("choices")
        or style_config.get("profiles")
        or config.get("generation", {}).get("style_profile_choices")
        or list(STYLE_PROFILES)
    )
    choices = [str(name) for name in raw_choices if str(name) in STYLE_PROFILES]
    return choices or [DEFAULT_STYLE_PROFILE]


def style_profile_instructions(profile: str) -> str:
    return STYLE_PROFILES.get(profile, STYLE_PROFILES[DEFAULT_STYLE_PROFILE])


def style_profile_for_row(row: dict[str, Any], config: dict[str, Any], *, key_text: str) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    direct = clean_text(row.get("style_profile") or metadata.get("style_profile"))
    if direct in STYLE_PROFILES:
        return direct

    choices = configured_style_profiles(config)
    style_config = _style_profile_config(config)
    seed = style_config.get("seed", config.get("selection", {}).get("seed", 260701))
    digest = stable_hash(f"{seed}:{key_text}", n=8)
    return choices[int(digest, 16) % len(choices)]



def split_think_trajectory(text: str) -> tuple[str, str, str]:
    text = str(text or "").strip()
    match = re.search(r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return text, "", "no_think"
    reasoning = match.group(1).strip()
    final = text[match.end() :].strip()
    return reasoning, final, "explicit_think"


def strip_think_tags(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def render_think_output(reasoning: str, final_answer: str = "") -> str:
    output = f"{THINK_OPEN}\n{strip_think_tags(reasoning)}\n{THINK_CLOSE}"
    final = strip_think_tags(final_answer)
    if final:
        output += "\n" + final
    return output


def length_target_for_unsafe(unsafe_trajectory: str, config: dict[str, Any]) -> dict[str, Any]:
    generation = config.get("generation", {})
    matching = generation.get("length_matching", {}) or {}
    unsafe_words = word_count(unsafe_trajectory)
    if not bool(matching.get("enabled", True)):
        return {
            "unsafe_words": unsafe_words,
            "target_min_words": 0,
            "target_max_words": 0,
            "min_ratio": None,
            "max_ratio": None,
            "enabled": False,
        }

    min_ratio = float(matching.get("min_ratio", 0.75))
    max_ratio = float(matching.get("max_ratio", 1.10))
    min_words = int(matching.get("min_reasoning_words", 250))
    max_words = int(matching.get("max_reasoning_words", 450))
    if max_words < min_words:
        max_words = min_words

    if unsafe_words <= 0:
        lower = min_words
        upper = max_words
    else:
        ratio_lower = int(math.floor(unsafe_words * min_ratio))
        ratio_upper = int(math.ceil(unsafe_words * max_ratio))
        # Long unsafe trajectories are capped to avoid fake safe verbosity; short
        # trajectories are lifted to preserve CoT-position coverage.
        if ratio_lower > max_words:
            lower = min_words
            upper = max_words
        else:
            lower = max(min_words, ratio_lower)
            upper = min(max_words, max(min_words, ratio_upper))
        if lower > upper:
            lower = upper

    return {
        "unsafe_words": unsafe_words,
        "target_min_words": lower,
        "target_max_words": upper,
        "min_ratio": min_ratio,
        "max_ratio": max_ratio,
        "absolute_min_words": min_words,
        "absolute_max_words": max_words,
        "enabled": True,
    }


def parse_jsonish(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected generated JSON object")
    return parsed


def normalize_generated_rewrite(generated: dict[str, Any]) -> dict[str, Any]:
    trajectory = generated.get("safe_trajectory") or generated.get("trajectory") or ""
    parsed_reasoning, parsed_final, parse_status = split_think_trajectory(str(trajectory or ""))
    reasoning = (
        generated.get("safe_reasoning")
        or generated.get("reasoning")
        or generated.get("safe_cot")
        or generated.get("cot")
        or (parsed_reasoning if trajectory else "")
        or generated.get("response")
        or generated.get("output")
        or ""
    )
    final_answer = (
        generated.get("safe_final_answer")
        or generated.get("final_answer")
        or generated.get("answer")
        or parsed_final
        or ""
    )
    normalized = dict(generated)
    normalized["safe_reasoning"] = strip_think_tags(str(reasoning))
    normalized["safe_final_answer"] = strip_think_tags(str(final_answer))
    normalized["safe_trajectory"] = render_think_output(
        normalized["safe_reasoning"],
        normalized["safe_final_answer"],
    )
    normalized["label"] = "safe"
    if generated.get("style_profile"):
        normalized["style_profile"] = clean_text(generated.get("style_profile"))
    normalized["safe_parse_status"] = parse_status
    return normalized


def build_safe_rewrite_prompt(
    *,
    prompt: str,
    unsafe_trajectory: str,
    unsafe_final_answer: str = "",
    length_target: dict[str, Any] | None = None,
    style_profile: str = DEFAULT_STYLE_PROFILE,
) -> str:
    if length_target is None:
        length_target = length_target_for_unsafe(unsafe_trajectory, {})
    return SAFE_REWRITE_USER_TEMPLATE.format(
        prompt=prompt.strip(),
        unsafe_trajectory=unsafe_trajectory.strip(),
        unsafe_final_answer=unsafe_final_answer.strip() or "(none)",
        unsafe_word_count=length_target.get("unsafe_words", word_count(unsafe_trajectory)),
        target_min_words=length_target.get("target_min_words", 0) or "not enforced",
        target_max_words=length_target.get("target_max_words", 0) or "not enforced",
        style_profile=style_profile,
        style_instructions=style_profile_instructions(style_profile),
    )


def build_safe_length_repair_prompt(record: dict[str, Any]) -> str:
    target = record.get("length_target", {}) or {}
    safe_reasoning = strip_think_tags(record.get("safe_reasoning", ""))
    safe_final_answer = strip_think_tags(record.get("safe_final_answer", ""))
    style_profile = clean_text(record.get("style_profile")) or DEFAULT_STYLE_PROFILE
    return SAFE_LENGTH_REPAIR_USER_TEMPLATE.format(
        prompt=str(record.get("prompt", "")).strip(),
        safe_reasoning=safe_reasoning,
        safe_final_answer=safe_final_answer or "(none)",
        current_words=word_count(safe_reasoning),
        target_min_words=target.get("target_min_words", 0) or "not enforced",
        target_max_words=target.get("target_max_words", 0) or "not enforced",
        style_profile=style_profile,
        style_instructions=style_profile_instructions(style_profile),
    )


def build_safe_polish_prompt(record: dict[str, Any], config: dict[str, Any] | None = None) -> str:
    config = config or {}
    polish = config.get("generation", {}).get("polish", {}) or {}
    safe_reasoning = strip_think_tags(record.get("safe_reasoning", ""))
    safe_final_answer = strip_think_tags(record.get("safe_final_answer", ""))
    style_profile = clean_text(record.get("style_profile")) or DEFAULT_STYLE_PROFILE
    base_min_words = int(polish.get("min_reasoning_words", 300))
    target_max_words = int(polish.get("max_reasoning_words", 430))
    target_final_max_words = int(polish.get("max_final_words", 80))
    current_words = word_count(safe_reasoning)
    preserve_margin = int(polish.get("preserve_length_margin_words", 30))
    if current_words and current_words <= target_max_words:
        target_min_words = max(base_min_words, current_words - preserve_margin)
    else:
        target_min_words = base_min_words
    if target_min_words > target_max_words:
        target_min_words = target_max_words
    if bool(polish.get("hard_boilerplate_reduction", False)):
        boilerplate_controls = (
            "Hard boilerplate constraint: avoid the exact words ethical, ethically, unethical, "
            "legal, legally, illegal, policy, policies, professional, professionally, responsible, "
            "and irresponsible unless the user prompt itself requires the word. Prefer concrete "
            "phrasing about harm, consent, authorization, privacy, safety, trust, oversight, "
            "and accountability."
        )
    else:
        boilerplate_controls = (
            "Avoid broad safety-policy boilerplate; prefer concrete risk, privacy, authorization, "
            "and harm-prevention reasoning."
        )
    template = SAFE_POLISH_USER_TEMPLATE
    if (
        str(polish.get("candidate_mode", "")).lower() == "controlled_clean"
        or bool(polish.get("controlled_clean", False))
    ):
        template = SAFE_CONTROLLED_CLEAN_POLISH_USER_TEMPLATE
    return template.format(
        prompt=str(record.get("prompt", "")).strip(),
        safe_reasoning=safe_reasoning,
        safe_final_answer=safe_final_answer or "(none)",
        current_words=current_words,
        target_min_words=target_min_words,
        target_max_words=target_max_words,
        target_final_max_words=target_final_max_words,
        style_profile=style_profile,
        style_instructions=style_profile_instructions(style_profile),
        boilerplate_controls=boilerplate_controls,
    )


def select_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    selection = config.get("selection", {})
    prompt_fields = _as_field_list(config.get("data", {}).get("prompt_fields"), DEFAULT_PROMPT_FIELDS)
    trajectory_fields = _as_field_list(
        config.get("data", {}).get("unsafe_trajectory_fields"),
        DEFAULT_TRAJECTORY_FIELDS,
    )
    label_field = selection.get("label_field")
    require_unsafe_label = bool(selection.get("require_unsafe_label", False))
    allowed_sources = set(selection.get("allowed_sources", []) or [])

    selected: list[dict[str, Any]] = []
    for row in rows:
        _, prompt = first_text_field(row, prompt_fields)
        _, trajectory = first_text_field(row, trajectory_fields)
        if not prompt or not trajectory:
            continue
        if allowed_sources and str(row.get("source")) not in allowed_sources:
            continue
        if require_unsafe_label:
            label, _, _ = label_from_row(row, str(label_field) if label_field else None)
            if label != 1:
                continue
        selected.append(row)

    seed = int(selection.get("seed", 260701))
    limit = selection.get("n_rows", selection.get("n_prompts"))
    if selection.get("shuffle", True):
        import random

        random.Random(seed).shuffle(selected)
    if limit is not None:
        selected = selected[: int(limit)]
    return selected


def make_tasks(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    data = config.get("data", {})
    generation = config.get("generation", {})
    prompt_fields = _as_field_list(data.get("prompt_fields"), DEFAULT_PROMPT_FIELDS)
    trajectory_fields = _as_field_list(data.get("unsafe_trajectory_fields"), DEFAULT_TRAJECTORY_FIELDS)
    final_fields = _as_field_list(data.get("unsafe_final_answer_fields"), DEFAULT_FINAL_FIELDS)
    id_fields = _as_field_list(data.get("id_fields"), DEFAULT_ID_FIELDS)
    max_excerpt_chars = int(generation.get("max_unsafe_excerpt_chars", 3500))

    tasks: list[dict[str, Any]] = []
    for row in select_rows(rows, config):
        prompt_field, prompt = first_text_field(row, prompt_fields)
        trajectory_field, unsafe_trajectory = first_text_field(row, trajectory_fields)
        final_field, unsafe_final_answer = first_text_field(row, final_fields)
        source_id = source_id_for_row(row, id_fields, fallback_text=prompt + "\n" + unsafe_trajectory)
        pair_id = str(row.get("pair_id") or f"{source_id}::unsafe_to_safe")
        prompt_id = str(row.get("prompt_id") or source_id)
        label_field = data.get("label_field") or config.get("selection", {}).get("label_field")
        label, label_name, label_source = label_from_row(row, label_field)
        unsafe_excerpt = unsafe_trajectory[:max_excerpt_chars]
        length_target = length_target_for_unsafe(unsafe_trajectory, config)
        style_profile = style_profile_for_row(
            row,
            config,
            key_text=pair_id + "\n" + prompt,
        )
        tasks.append(
            {
                "task_id": pair_id,
                "pair_id": pair_id,
                "source_id": source_id,
                "prompt_id": prompt_id,
                "source": row.get("source"),
                "prompt": prompt,
                "prompt_field": prompt_field,
                "unsafe_trajectory": unsafe_trajectory,
                "unsafe_trajectory_excerpt": unsafe_excerpt,
                "unsafe_word_count": word_count(unsafe_trajectory),
                "unsafe_excerpt_word_count": word_count(unsafe_excerpt),
                "length_target": length_target,
                "style_profile": style_profile,
                "style_instructions": style_profile_instructions(style_profile),
                "unsafe_trajectory_field": trajectory_field,
                "unsafe_final_answer": unsafe_final_answer,
                "unsafe_final_answer_field": final_field,
                "original_label": label,
                "original_label_name": label_name,
                "original_label_source": label_source,
                "metadata": row.get("metadata", {}),
                "user_prompt": build_safe_rewrite_prompt(
                    prompt=prompt,
                    unsafe_trajectory=unsafe_excerpt,
                    unsafe_final_answer=unsafe_final_answer,
                    length_target=length_target,
                    style_profile=style_profile,
                ),
            }
        )
    return tasks


def base_pair_record(task: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    generation = config.get("generation", {})
    include_unsafe = bool(config.get("data", {}).get("include_unsafe_trajectory", True))
    record = {
        "pair_id": task.get("pair_id"),
        "source_id": task.get("source_id"),
        "prompt_id": task.get("prompt_id"),
        "source": task.get("source"),
        "prompt": task.get("prompt"),
        "unsafe_label": "unsafe",
        "safe_label": "safe",
        "provider": generation.get("provider", "openai"),
        "model": generation.get("model"),
        "temperature": generation.get("temperature"),
        "prompt_version": config.get("prompt_version", "unsafe_to_safe_rewrite_v1"),
        "created_unix": int(time.time()),
        "original_label": task.get("original_label"),
        "original_label_name": task.get("original_label_name"),
        "original_label_source": task.get("original_label_source"),
        "metadata": task.get("metadata", {}),
        "unsafe_word_count": task.get("unsafe_word_count"),
        "length_target": task.get("length_target", {}),
        "style_profile": task.get("style_profile", DEFAULT_STYLE_PROFILE),
        "style_instructions": task.get("style_instructions", style_profile_instructions(DEFAULT_STYLE_PROFILE)),
    }
    if include_unsafe:
        record["unsafe_trajectory"] = task.get("unsafe_trajectory", "")
        record["unsafe_final_answer"] = task.get("unsafe_final_answer", "")
    else:
        record["unsafe_trajectory_sha256_16"] = stable_hash(str(task.get("unsafe_trajectory", "")), n=16)
    return record


def _length_match_pass(
    *,
    safe_reasoning_words: int,
    length_target: dict[str, Any],
) -> bool:
    target_min = int(length_target.get("target_min_words") or 0)
    target_max = int(length_target.get("target_max_words") or 0)
    if bool(length_target.get("enabled", True)) and target_min and target_max:
        return target_min <= safe_reasoning_words <= target_max
    return True


def update_pair_record_with_generated_safe(
    record: dict[str, Any],
    generated: dict[str, Any],
    *,
    api_response_id: str | None = None,
    usage: dict[str, Any] | None = None,
    repair_round: int | None = None,
    polish_round: int | None = None,
) -> dict[str, Any]:
    normalized = normalize_generated_rewrite(generated)
    safe_reasoning_words = word_count(normalized["safe_reasoning"])
    unsafe_words = int(record.get("unsafe_word_count") or word_count(record.get("unsafe_trajectory", "")))
    target = record.get("length_target", {}) or {}
    generated_style_profile = (
        normalized.get("style_profile")
        or record.get("generated_style_profile")
        or record.get("style_profile")
        or ""
    )
    record.update(
        {
            "safe_reasoning": normalized["safe_reasoning"],
            "safe_trajectory": normalized["safe_trajectory"],
            "safe_final_answer": normalized["safe_final_answer"],
            "label": normalized["label"],
            "generated_style_profile": generated_style_profile,
            "safe_reasoning_word_count": safe_reasoning_words,
            "safe_trajectory_word_count": word_count(normalized["safe_trajectory"]),
            "safe_to_unsafe_reasoning_word_ratio": (
                safe_reasoning_words / max(1, unsafe_words)
            ),
            "unsafe_to_safe_reasoning_compression_ratio": (
                unsafe_words / max(1, safe_reasoning_words)
            ),
            "length_match_pass": _length_match_pass(
                safe_reasoning_words=safe_reasoning_words,
                length_target=target,
            ),
            "safe_parse_status": normalized.get("safe_parse_status"),
            "length_notes": normalized.get("length_notes", ""),
            "rewrite_notes": normalized.get("rewrite_notes", ""),
            "safety_notes": normalized.get("safety_notes", ""),
            "ok": True,
        }
    )
    if api_response_id:
        if polish_round is not None:
            response_key = "polish_api_response_id"
        else:
            response_key = "api_response_id" if repair_round is None else "length_repair_api_response_id"
        record[response_key] = api_response_id
    if usage is not None:
        if polish_round is not None:
            usage_key = "polish_usage"
        else:
            usage_key = "usage" if repair_round is None else "length_repair_usage"
        record[usage_key] = usage
    if repair_round is not None:
        record["length_repair_applied"] = True
        record["length_repair_rounds"] = repair_round
        record.pop("length_repair_error", None)
    if polish_round is not None:
        record["polish_applied"] = True
        record["polish_rounds"] = polish_round
        record.pop("polish_error", None)
    return record


def merge_generated_pair(
    task: dict[str, Any],
    generated: dict[str, Any],
    config: dict[str, Any],
    *,
    api_response_id: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = base_pair_record(task, config)
    return update_pair_record_with_generated_safe(
        record,
        generated,
        api_response_id=api_response_id,
        usage=usage,
    )


def failed_pair_record(
    task: dict[str, Any],
    config: dict[str, Any],
    error: str,
    *,
    raw: Any | None = None,
) -> dict[str, Any]:
    record = base_pair_record(task, config)
    record.update({"ok": False, "error": error})
    if raw is not None:
        record["raw"] = raw
    return record


def pair_record_to_long_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    base = {
        "pair_id": record.get("pair_id"),
        "prompt_id": record.get("prompt_id"),
        "source_id": record.get("source_id"),
        "source": record.get("source"),
        "prompt": record.get("prompt"),
        "provider": record.get("provider"),
        "model": record.get("model"),
        "prompt_version": record.get("prompt_version"),
        "created_unix": record.get("created_unix"),
        "style_profile": record.get("style_profile"),
        "metadata": record.get("metadata", {}),
    }
    rows: list[dict[str, Any]] = []
    if "unsafe_trajectory" in record:
        rows.append(
            {
                **base,
                "id": f"{record.get('pair_id')}::unsafe",
                "variant": "unsafe_original",
                "trajectory": record.get("unsafe_trajectory", ""),
                "reasoning": record.get("unsafe_trajectory", ""),
                "final_answer": record.get("unsafe_final_answer", ""),
                "label": 1,
                "label_name": "unsafe",
                "style_profile": record.get("style_profile"),
            }
        )
    rows.append(
        {
            **base,
            "id": f"{record.get('pair_id')}::safe",
            "variant": "safe_rewrite",
            "trajectory": record.get("safe_trajectory", ""),
            "reasoning": record.get("safe_reasoning", ""),
            "final_answer": record.get("safe_final_answer", ""),
            "label": 0,
            "label_name": "safe",
            "style_profile": record.get("style_profile"),
            "ok": record.get("ok", True),
        }
    )
    return rows


def word_count(value: Any) -> int:
    return len(clean_text(value).split())
