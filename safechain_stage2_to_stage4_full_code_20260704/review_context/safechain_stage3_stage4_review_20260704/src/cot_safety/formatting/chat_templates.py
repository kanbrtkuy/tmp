from __future__ import annotations

from cot_safety.schemas import ChatTemplate


DEEPSEEK_R1_DISTILL = ChatTemplate(
    name="deepseek_r1_distill",
    bos_token="<｜begin▁of▁sentence｜>",
    user_template="<｜User｜>",
    assistant_template="<｜Assistant｜>",
    think_open="<think>",
    think_close="</think>",
)


def render_prompt_completion(
    prompt: str,
    completion: str,
    template: ChatTemplate,
    *,
    append_eos: str | None = None,
) -> str:
    text = f"{template.bos_token}{template.user_template}{prompt}{template.assistant_template}{completion}"
    if append_eos:
        text += append_eos
    return text


def template_from_config(config: dict) -> ChatTemplate:
    return ChatTemplate(
        name=str(config.get("name", "custom")),
        bos_token=str(config.get("bos_token", "")),
        user_template=str(config.get("user_template", "")),
        assistant_template=str(config.get("assistant_template", "")),
        eos_token=str(config.get("eos_token", "")),
        think_open=str(config.get("think_open", "<think>")),
        think_close=str(config.get("think_close", "</think>")),
    )
