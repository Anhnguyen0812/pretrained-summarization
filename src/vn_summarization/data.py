from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset

from .utils import LOGGER, resolve_path


ARTICLE_COL = "article"
SUMMARY_COL = "summary"


def clean_text(value: str) -> str:
    text = unicodedata.normalize("NFC", str(value))
    text = text.replace("\u00a0", " ").replace("\ufeff", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_splits(config: dict[str, Any], config_path: Path) -> DatasetDict:
    data_cfg = config["data"]
    train_file = resolve_path(data_cfg["train_file"], config_path)
    valid_file = resolve_path(data_cfg["valid_file"], config_path)
    LOGGER.info("loading train=%s", train_file)
    LOGGER.info("loading valid=%s", valid_file)
    return load_dataset(
        "parquet",
        data_files={"train": str(train_file), "validation": str(valid_file)},
    )


def maybe_select(split, max_samples: int | None, seed: int):
    if not max_samples:
        return split
    max_samples = min(int(max_samples), len(split))
    return split.shuffle(seed=seed).select(range(max_samples))


def preprocess_dataset(dataset: DatasetDict, tokenizer, config: dict[str, Any]) -> DatasetDict:
    data_cfg = config["data"]
    seed = int(config["training"].get("seed", 42))
    prefix = data_cfg.get("source_prefix", "") or ""
    max_source_length = int(data_cfg["max_source_length"])
    max_target_length = int(data_cfg["max_target_length"])
    num_proc = int(data_cfg.get("preprocessing_num_proc", 1))

    dataset = DatasetDict(
        {
            "train": maybe_select(dataset["train"], data_cfg.get("max_train_samples"), seed),
            "validation": maybe_select(
                dataset["validation"], data_cfg.get("max_eval_samples"), seed
            ),
        }
    )

    def preprocess_batch(examples):
        inputs = [prefix + clean_text(text) for text in examples[ARTICLE_COL]]
        targets = [clean_text(text) for text in examples[SUMMARY_COL]]
        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    column_names = dataset["train"].column_names
    return dataset.map(
        preprocess_batch,
        batched=True,
        num_proc=num_proc,
        remove_columns=column_names,
        desc="tokenizing",
    )

