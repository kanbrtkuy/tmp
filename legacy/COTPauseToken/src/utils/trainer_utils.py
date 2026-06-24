from src.utils.constants import (
    DEEPSEEK_ASSISTANT_TEMPLATE,
    DEEPSEEK_BOS_TOKEN,
    DEEPSEEK_USER_TEMPLATE,
)


def encode_response_template(tokenizer, response_template: str) -> list[int]:
    token_ids = tokenizer.encode(response_template, add_special_tokens=False)
    if token_ids and token_ids[0] == getattr(tokenizer, "bos_token_id", None):
        token_ids = token_ids[1:]
    return token_ids


def deepseek_pause_sft_formatting_function(example, eos_token):
    rows = []
    for prompt, completion in zip(example["input"], example["output"]):
        rows.append(
            f"{DEEPSEEK_BOS_TOKEN}{DEEPSEEK_USER_TEMPLATE}{prompt}"
            f"{DEEPSEEK_ASSISTANT_TEMPLATE}{completion}{eos_token}"
        )
    return rows
