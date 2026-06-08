from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from .data import ARTICLE_COL, SUMMARY_COL, clean_text, load_splits, preprocess_dataset
from .metrics import build_compute_metrics
from .modeling import load_tokenizer_and_model
from .utils import apply_overrides, configure_logging, load_yaml, save_json


def _trainer_processing_kwargs(tokenizer) -> dict:
    signature = inspect.signature(Seq2SeqTrainer.__init__)
    if "tokenizer" in signature.parameters:
        return {"tokenizer": tokenizer}
    if "processing_class" in signature.parameters:
        return {"processing_class": tokenizer}
    return {}


def evaluate_model(config: dict, config_path: Path, model_path: str, predictions_path: str | None):
    config = dict(config)
    config["model"] = dict(config["model"])
    config["model"]["name_or_path"] = model_path
    tokenizer, model = load_tokenizer_and_model(config, for_training=False)
    raw_dataset = load_splits(config, config_path)
    tokenized = preprocess_dataset(raw_dataset, tokenizer, config)

    args = Seq2SeqTrainingArguments(
        output_dir=str(Path(config["training"].get("output_dir", "outputs/eval_tmp")) / "eval_tmp"),
        do_train=False,
        do_eval=True,
        predict_with_generate=True,
        per_device_eval_batch_size=int(config["training"].get("per_device_eval_batch_size", 4)),
        generation_max_length=int(config.get("generation", {}).get("max_length", 160)),
        generation_num_beams=int(config.get("generation", {}).get("num_beams", 4)),
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100),
        compute_metrics=build_compute_metrics(tokenizer),
        **_trainer_processing_kwargs(tokenizer),
    )
    output = trainer.predict(
        tokenized["validation"],
        max_length=int(config.get("generation", {}).get("max_length", 160)),
        num_beams=int(config.get("generation", {}).get("num_beams", 4)),
    )
    metrics = {key: float(value) for key, value in output.metrics.items()}
    save_json(metrics, Path(model_path) / "validation_metrics.json")

    if predictions_path:
        preds = tokenizer.batch_decode(output.predictions, skip_special_tokens=True)
        out_path = Path(predictions_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for source, target, pred in zip(
                raw_dataset["validation"][ARTICLE_COL], raw_dataset["validation"][SUMMARY_COL], preds
            ):
                record = {
                    "article": clean_text(source),
                    "summary": clean_text(target),
                    "prediction": clean_text(pred),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned summarizer.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--predictions_path")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config, config_path = load_yaml(args.config)
    config = apply_overrides(config, args.overrides)
    metrics = evaluate_model(config, config_path, args.model_path, args.predictions_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
