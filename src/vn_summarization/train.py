from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from .data import load_splits, preprocess_dataset
from .metrics import build_compute_metrics
from .modeling import apply_lora_if_enabled, load_tokenizer_and_model
from .utils import (
    LOGGER,
    apply_overrides,
    configure_logging,
    infer_precision,
    load_yaml,
    log_runtime,
    save_json,
    set_all_seeds,
)


def _supported_training_args(raw_args: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    supported = set(signature.parameters)
    return {key: value for key, value in raw_args.items() if key in supported}


def _trainer_processing_kwargs(tokenizer) -> dict[str, Any]:
    signature = inspect.signature(Seq2SeqTrainer.__init__)
    if "tokenizer" in signature.parameters:
        return {"tokenizer": tokenizer}
    if "processing_class" in signature.parameters:
        return {"processing_class": tokenizer}
    return {}


def build_training_args(config: dict[str, Any]) -> Seq2SeqTrainingArguments:
    training_cfg = config["training"]
    generation_cfg = config.get("generation", {})
    fp16, bf16 = infer_precision(str(training_cfg.get("precision", "auto")))
    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_args = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": bool(training_cfg.get("overwrite_output_dir", False)),
        "do_train": True,
        "do_eval": True,
        "predict_with_generate": True,
        "per_device_train_batch_size": int(training_cfg.get("per_device_train_batch_size", 2)),
        "per_device_eval_batch_size": int(training_cfg.get("per_device_eval_batch_size", 4)),
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(training_cfg.get("learning_rate", 3e-5)),
        "num_train_epochs": float(training_cfg.get("num_train_epochs", 5)),
        "max_steps": int(training_cfg.get("max_steps", -1)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.01)),
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.06)),
        "lr_scheduler_type": training_cfg.get("lr_scheduler_type", "cosine"),
        "label_smoothing_factor": float(training_cfg.get("label_smoothing_factor", 0.0)),
        "optim": training_cfg.get("optim", "adamw_torch"),
        "logging_steps": int(training_cfg.get("logging_steps", 50)),
        "eval_steps": int(training_cfg.get("eval_steps", 250)),
        "save_steps": int(training_cfg.get("save_steps", 250)),
        "save_total_limit": int(training_cfg.get("save_total_limit", 3)),
        "load_best_model_at_end": bool(training_cfg.get("load_best_model_at_end", True)),
        "metric_for_best_model": training_cfg.get("metric_for_best_model", "rougeL"),
        "greater_is_better": bool(training_cfg.get("greater_is_better", True)),
        "fp16": fp16,
        "bf16": bf16,
        "gradient_checkpointing": bool(training_cfg.get("gradient_checkpointing", False)),
        "report_to": training_cfg.get("report_to", ["tensorboard"]),
        "seed": int(training_cfg.get("seed", 42)),
        "data_seed": int(training_cfg.get("seed", 42)),
        "generation_max_length": int(generation_cfg.get("max_length", config["data"]["max_target_length"])),
        "generation_num_beams": int(generation_cfg.get("num_beams", 4)),
        "remove_unused_columns": True,
        "save_safetensors": bool(training_cfg.get("save_safetensors", True)),
        "ddp_find_unused_parameters": training_cfg.get("ddp_find_unused_parameters", None),
    }

    strategy = training_cfg.get("strategy", "steps")
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        raw_args["eval_strategy"] = strategy
    else:
        raw_args["evaluation_strategy"] = strategy
    raw_args["save_strategy"] = training_cfg.get("save_strategy", strategy)

    return Seq2SeqTrainingArguments(**_supported_training_args(raw_args))


def train(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    seed = int(config["training"].get("seed", 42))
    set_all_seeds(seed)
    log_runtime()

    tokenizer, model = load_tokenizer_and_model(config, for_training=True)
    model = apply_lora_if_enabled(model, config)

    raw_dataset = load_splits(config, config_path)
    tokenized = preprocess_dataset(raw_dataset, tokenizer, config)

    training_args = build_training_args(config)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None,
    )

    callbacks = []
    patience = config["training"].get("early_stopping_patience")
    if patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(patience)))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=callbacks,
        **_trainer_processing_kwargs(tokenizer),
    )

    LOGGER.info("start training output_dir=%s", training_args.output_dir)
    train_result = trainer.train(resume_from_checkpoint=config["training"].get("resume_from_checkpoint"))
    trainer.save_model(Path(training_args.output_dir) / "best")
    tokenizer.save_pretrained(Path(training_args.output_dir) / "best")

    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate(
        max_length=int(config.get("generation", {}).get("max_length", config["data"]["max_target_length"])),
        num_beams=int(config.get("generation", {}).get("num_beams", 4)),
    )
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    save_json(config, Path(training_args.output_dir) / "resolved_config.json")
    LOGGER.info("done eval=%s", eval_metrics)
    return eval_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune pretrained seq2seq summarizer.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config with dotted key, example: --set training.max_steps=10",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config, config_path = load_yaml(args.config)
    config = apply_overrides(config, args.overrides)
    train(config, config_path)


if __name__ == "__main__":
    main()
