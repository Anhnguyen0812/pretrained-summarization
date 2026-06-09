from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .causal_data import load_causal_splits, preprocess_causal_dataset
from .utils import (
    LOGGER,
    apply_overrides,
    configure_logging,
    count_parameters,
    infer_precision,
    load_yaml,
    log_runtime,
    save_json,
    set_all_seeds,
)


class CausalDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int | None = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            pad_len = max_len - len(label)
            if self.tokenizer.padding_side == "left":
                padded_labels.append([-100] * pad_len + label)
            else:
                padded_labels.append(label + [-100] * pad_len)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def _supported_training_args(raw_args: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(TrainingArguments.__init__)
    supported = set(signature.parameters)
    return {key: value for key, value in raw_args.items() if key in supported}


def _trainer_processing_kwargs(tokenizer) -> dict[str, Any]:
    signature = inspect.signature(Trainer.__init__)
    if "tokenizer" in signature.parameters:
        return {"tokenizer": tokenizer}
    if "processing_class" in signature.parameters:
        return {"processing_class": tokenizer}
    return {}


def load_tokenizer_and_model(config: dict[str, Any], for_training: bool = True):
    model_cfg = config["model"]
    training_cfg = config.get("training", {})
    name_or_path = model_cfg["name_or_path"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    cache_dir = model_cfg.get("cache_dir")

    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        use_fast=bool(model_cfg.get("use_fast_tokenizer", True)),
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = model_cfg.get("padding_side", "right")

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "cache_dir": cache_dir,
    }
    dtype = model_cfg.get("torch_dtype")
    if dtype == "float16":
        model_kwargs["torch_dtype"] = torch.float16
    elif dtype == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif dtype == "float32":
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(name_or_path, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "use_cache") and for_training:
        model.config.use_cache = False

    if for_training and bool(training_cfg.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    max_parameters = int(model_cfg.get("max_parameters", 3_000_000_000))
    params = count_parameters(model)
    if params["total"] >= max_parameters:
        raise ValueError(f"Model has {params['total']} parameters, violates limit {max_parameters}.")
    LOGGER.info("causal model=%s params=%s", name_or_path, params)
    return tokenizer, model


def apply_lora_if_enabled(model, config: dict[str, Any]):
    lora_cfg = config.get("lora", {})
    if not lora_cfg.get("enabled", True):
        return model

    from peft import LoraConfig, TaskType, get_peft_model

    target_modules = lora_cfg.get("target_modules") or [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    LOGGER.info("causal lora target_modules=%s params=%s", target_modules, count_parameters(model))
    return model


def build_training_args(config: dict[str, Any]) -> TrainingArguments:
    training_cfg = config["training"]
    fp16, bf16 = infer_precision(str(training_cfg.get("precision", "auto")))
    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_args = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": bool(training_cfg.get("overwrite_output_dir", False)),
        "do_train": True,
        "do_eval": True,
        "per_device_train_batch_size": int(training_cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(training_cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(training_cfg.get("learning_rate", 1e-4)),
        "num_train_epochs": float(training_cfg.get("num_train_epochs", 2)),
        "max_steps": int(training_cfg.get("max_steps", -1)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.01)),
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.03)),
        "lr_scheduler_type": training_cfg.get("lr_scheduler_type", "cosine"),
        "optim": training_cfg.get("optim", "adamw_torch"),
        "logging_steps": int(training_cfg.get("logging_steps", 50)),
        "eval_steps": int(training_cfg.get("eval_steps", 250)),
        "save_steps": int(training_cfg.get("save_steps", 250)),
        "save_total_limit": int(training_cfg.get("save_total_limit", 2)),
        "load_best_model_at_end": bool(training_cfg.get("load_best_model_at_end", True)),
        "metric_for_best_model": training_cfg.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": bool(training_cfg.get("greater_is_better", False)),
        "fp16": fp16,
        "bf16": bf16,
        "gradient_checkpointing": bool(training_cfg.get("gradient_checkpointing", False)),
        "report_to": training_cfg.get("report_to", ["tensorboard"]),
        "seed": int(training_cfg.get("seed", 42)),
        "data_seed": int(training_cfg.get("seed", 42)),
        "remove_unused_columns": False,
        "save_safetensors": bool(training_cfg.get("save_safetensors", True)),
        "ddp_find_unused_parameters": training_cfg.get("ddp_find_unused_parameters", None),
    }

    strategy = training_cfg.get("strategy", "steps")
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        raw_args["eval_strategy"] = strategy
    else:
        raw_args["evaluation_strategy"] = strategy
    raw_args["save_strategy"] = training_cfg.get("save_strategy", strategy)
    return TrainingArguments(**_supported_training_args(raw_args))


def train(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    set_all_seeds(int(config["training"].get("seed", 42)))
    log_runtime()
    tokenizer, model = load_tokenizer_and_model(config, for_training=True)
    model = apply_lora_if_enabled(model, config)

    raw_dataset = load_causal_splits(config, config_path)
    tokenized = preprocess_causal_dataset(raw_dataset, tokenizer, config)
    training_args = build_training_args(config)
    data_collator = CausalDataCollator(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        **_trainer_processing_kwargs(tokenizer),
    )
    LOGGER.info("start causal LM training output_dir=%s", training_args.output_dir)
    train_result = trainer.train(resume_from_checkpoint=config["training"].get("resume_from_checkpoint"))
    trainer.save_model(Path(training_args.output_dir) / "best")
    tokenizer.save_pretrained(Path(training_args.output_dir) / "best")
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    eval_metrics = trainer.evaluate()
    trainer.save_metrics("eval", eval_metrics)
    save_json(config, Path(training_args.output_dir) / "resolved_config.json")
    LOGGER.info("done causal eval=%s", eval_metrics)
    return eval_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune causal LM summarizer with LoRA.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config, config_path = load_yaml(args.config)
    config = apply_overrides(config, args.overrides)
    train(config, config_path)


if __name__ == "__main__":
    main()

