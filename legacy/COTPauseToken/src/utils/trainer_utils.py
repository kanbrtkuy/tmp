import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict

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


def load_pause_sft_dataset(
    data_dir: str | None = None,
    train_file: str = "train.json",
    val_file: str = "val.json",
    test_file: str = "test.json",
) -> DatasetDict:
    root = Path(data_dir or os.environ["PAUSE_SFT_DATA_DIR"]).expanduser()
    split_files = {
        "train": train_file,
        "val": val_file,
        "test": test_file,
    }
    splits = {}
    for split, filename in split_files.items():
        path = root / filename
        if split == "test" and not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        splits[split] = Dataset.from_list(rows)
    return DatasetDict(splits)
