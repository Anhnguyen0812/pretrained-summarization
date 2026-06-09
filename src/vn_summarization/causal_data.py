from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import DatasetDict

from .data import ARTICLE_COL, SUMMARY_COL, clean_text, load_splits, maybe_select


DEFAULT_PROMPT_TEMPLATE = (
    "Bạn là hệ thống tóm tắt văn bản tiếng Việt. Tóm tắt văn bản sau thành một đoạn "
    "ngắn, giữ các ý chính, không bịa thông tin, không giải thích.\n\n"
    "Văn bản:\n{article}\n\nTóm tắt:\n"
)


def build_prompt(article: str, template: str = DEFAULT_PROMPT_TEMPLATE) -> str:
    return template.format(article=clean_text(article))


def preprocess_causal_dataset(dataset: DatasetDict, tokenizer, config: dict[str, Any]) -> DatasetDict:
    data_cfg = config["data"]
    seed = int(config["training"].get("seed", 42))
    max_source_length = int(data_cfg.get("max_source_length", 1024))
    max_target_length = int(data_cfg.get("max_target_length", 180))
    max_length = int(data_cfg.get("max_length", max_source_length + max_target_length))
    prompt_template = data_cfg.get("prompt_template") or DEFAULT_PROMPT_TEMPLATE
    num_proc = int(data_cfg.get("preprocessing_num_proc", 1))

    dataset = DatasetDict(
        {
            "train": maybe_select(dataset["train"], data_cfg.get("max_train_samples"), seed),
            "validation": maybe_select(dataset["validation"], data_cfg.get("max_eval_samples"), seed),
        }
    )

    def preprocess_batch(examples):
        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []
        for article, summary in zip(examples[ARTICLE_COL], examples[SUMMARY_COL]):
            prompt = build_prompt(article, prompt_template)
            target = clean_text(summary) + (tokenizer.eos_token or "")
            prompt_ids = tokenizer(
                prompt,
                truncation=True,
                max_length=max_source_length,
                add_special_tokens=True,
            )["input_ids"]
            target_ids = tokenizer(
                target,
                truncation=True,
                max_length=max_target_length,
                add_special_tokens=False,
            )["input_ids"]

            input_ids = (prompt_ids + target_ids)[:max_length]
            labels = [-100] * len(prompt_ids) + target_ids
            labels = labels[: len(input_ids)]
            attention_mask = [1] * len(input_ids)

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            labels_batch.append(labels)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels_batch,
        }

    return dataset.map(
        preprocess_batch,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset["train"].column_names,
        desc="tokenizing causal lm",
    )


def load_causal_splits(config: dict[str, Any], config_path: Path) -> DatasetDict:
    return load_splits(config, config_path)

