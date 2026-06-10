from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluate
import torch
from tqdm.auto import tqdm

from .causal_data import build_prompt, load_causal_splits
from .data import ARTICLE_COL, SUMMARY_COL, clean_text, maybe_select
from .train_causal_lm import load_tokenizer_and_model
from .utils import apply_overrides, configure_logging, load_yaml, save_json


def _load_model_for_eval(config: dict[str, Any], model_path: str):
    model_dir = Path(model_path)
    if (model_dir / "adapter_config.json").exists():
        tokenizer, model = load_tokenizer_and_model(config, for_training=False)
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, model_path)
        return tokenizer, model

    eval_config = dict(config)
    eval_config["model"] = dict(eval_config["model"])
    eval_config["model"]["name_or_path"] = model_path
    return load_tokenizer_and_model(eval_config, for_training=False)


def _clean_generation(text: str) -> str:
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    summary_markers = ["Tóm tắt:", "Tom tat:", "TÃ³m táº¯t:"]
    for marker in summary_markers:
        if marker in text:
            text = text.split(marker)[-1].strip()
    article_markers = ["Văn bản:", "Van ban:", "VÄƒn báº£n:"]
    for marker in article_markers:
        if marker in text:
            text = text.split(marker)[0].strip()
    return clean_text(text)

def evaluate_model(config: dict[str, Any], config_path: Path, model_path: str, predictions_path: str):
    tokenizer, model = _load_model_for_eval(config, model_path)
    tokenizer.padding_side = "left"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    raw = load_causal_splits(config, config_path)
    seed = int(config["training"].get("seed", 42))
    valid = maybe_select(raw["validation"], config["data"].get("max_eval_samples"), seed)
    prompt_template = config["data"].get("prompt_template")
    gen_cfg = config.get("generation", {})
    max_source_length = int(config["data"].get("max_source_length", 1024))
    batch_size = int(config["training"].get("per_device_eval_batch_size", 1))

    predictions: list[str] = []
    references: list[str] = []
    articles: list[str] = []

    for start in tqdm(range(0, len(valid), batch_size), desc="generating"):
        rows = valid[start : start + batch_size]
        prompts = [build_prompt(article, prompt_template) for article in rows[ARTICLE_COL]]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_length,
        ).to(device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(gen_cfg.get("max_new_tokens", 180)),
                do_sample=bool(gen_cfg.get("do_sample", False)),
                num_beams=int(gen_cfg.get("num_beams", 4)),
                repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.08)),
                no_repeat_ngram_size=int(gen_cfg.get("no_repeat_ngram_size", 3)),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for output_ids in outputs:
            gen_ids = output_ids[input_len:]
            predictions.append(_clean_generation(tokenizer.decode(gen_ids, skip_special_tokens=True)))
        references.extend(clean_text(summary) for summary in rows[SUMMARY_COL])
        articles.extend(clean_text(article) for article in rows[ARTICLE_COL])

    rouge = evaluate.load("rouge")
    metrics = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
    )
    metrics = {key: round(value * 100, 4) for key, value in metrics.items()}
    metrics["gen_len"] = round(sum(len(pred.split()) for pred in predictions) / max(len(predictions), 1), 4)

    out_path = Path(predictions_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for article, reference, prediction in zip(articles, references, predictions):
            f.write(
                json.dumps(
                    {"article": article, "summary": reference, "prediction": prediction},
                    ensure_ascii=False,
                )
                + "\n"
            )
    save_json(metrics, out_path.parent / "validation_metrics.json")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate causal LM summarizer.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config, config_path = load_yaml(args.config)
    config = apply_overrides(config, args.overrides)
    evaluate_model(config, config_path, args.model_path, args.predictions_path)


if __name__ == "__main__":
    main()
