from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .evaluate import evaluate_model as evaluate_seq2seq
from .evaluate_causal_lm import evaluate_model as evaluate_causal
from .utils import configure_logging, save_json


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_metric(metrics: dict[str, Any], name: str) -> Any:
    candidates = [name, f"eval_{name}", f"test_{name}", f"predict_{name}"]
    for key in candidates:
        if key in metrics:
            return metrics[key]
    for key, value in metrics.items():
        if key.endswith("_" + name):
            return value
    return ""


def find_run_dirs(runs_root: Path, run_glob: str) -> list[Path]:
    candidates = sorted(path for path in runs_root.glob(run_glob) if path.is_dir())
    return [path for path in candidates if (path / "resolved_config.json").exists()]


def infer_kind(config: dict[str, Any]) -> str:
    if config.get("data", {}).get("prompt_template"):
        return "causal"
    return "seq2seq"


def model_path_for_run(run_dir: Path) -> Path:
    best = run_dir / "best"
    if best.exists() and has_model_artifacts(best):
        return best
    checkpoints = sorted(
        (path for path in run_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[-1]) if path.name.split("-")[-1].isdigit() else -1,
        reverse=True,
    )
    for checkpoint in checkpoints:
        if has_model_artifacts(checkpoint):
            return checkpoint
    return run_dir


def has_model_artifacts(model_dir: Path) -> bool:
    return any(
        (model_dir / name).exists()
        for name in ["adapter_model.safetensors", "model.safetensors", "pytorch_model.bin"]
    )


def prepare_test_config(
    config: dict[str, Any],
    test_file: Path,
    eval_dir: Path,
    max_test_samples: int | None,
    generation_max_new_tokens: int | None,
    generation_num_beams: int | None,
) -> dict[str, Any]:
    config = json.loads(json.dumps(config))
    config.setdefault("data", {})["valid_file"] = str(test_file)
    config.setdefault("training", {})["output_dir"] = str(eval_dir)
    config["data"]["max_eval_samples"] = int(max_test_samples) if max_test_samples else None
    if generation_max_new_tokens is not None:
        generation = config.setdefault("generation", {})
        generation["max_new_tokens"] = int(generation_max_new_tokens)
        generation["max_length"] = int(generation_max_new_tokens)
    if generation_num_beams is not None:
        config.setdefault("generation", {})["num_beams"] = int(generation_num_beams)
    return config


def evaluate_run(
    run_dir: Path,
    test_file: Path,
    out_dir: Path,
    max_test_samples: int | None,
    eval_batch_size: int | None,
    generation_max_new_tokens: int | None,
    generation_num_beams: int | None,
) -> dict[str, Any]:
    config_path = run_dir / "resolved_config.json"
    config = load_json(config_path)
    kind = infer_kind(config)
    eval_dir = out_dir / run_dir.name
    eval_dir.mkdir(parents=True, exist_ok=True)
    test_config = prepare_test_config(
        config, test_file, eval_dir, max_test_samples, generation_max_new_tokens, generation_num_beams
    )
    if eval_batch_size is not None:
        test_config.setdefault("training", {})["per_device_eval_batch_size"] = int(eval_batch_size)
    save_json(test_config, eval_dir / "resolved_test_config.json")

    predictions_path = eval_dir / "predictions_test.jsonl"
    model_path = model_path_for_run(run_dir)
    if kind == "causal":
        metrics = evaluate_causal(test_config, config_path, str(model_path), str(predictions_path))
    else:
        metrics = evaluate_seq2seq(test_config, config_path, str(model_path), str(predictions_path))

    metrics_path = eval_dir / "test_metrics.json"
    save_json(metrics, metrics_path)
    return {
        "run": run_dir.name,
        "kind": kind,
        "model": config.get("model", {}).get("name_or_path", ""),
        "rouge1": pick_metric(metrics, "rouge1"),
        "rouge2": pick_metric(metrics, "rouge2"),
        "rougeL": pick_metric(metrics, "rougeL"),
        "loss": pick_metric(metrics, "loss"),
        "gen_len": pick_metric(metrics, "gen_len"),
        "max_steps": config.get("training", {}).get("max_steps", ""),
        "train_samples": config.get("data", {}).get("max_train_samples", ""),
        "test_samples": max_test_samples or "all",
        "source_len": config.get("data", {}).get("max_source_length", ""),
        "target_len": config.get("data", {}).get("max_target_length", ""),
        "lora_r": config.get("lora", {}).get("r", ""),
        "beams": config.get("generation", {}).get("num_beams", ""),
        "metrics_file": str(metrics_path),
        "predictions_file": str(predictions_path),
    }


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "run",
        "kind",
        "model",
        "rouge1",
        "rouge2",
        "rougeL",
        "loss",
        "gen_len",
        "max_steps",
        "train_samples",
        "test_samples",
        "source_len",
        "target_len",
        "lora_r",
        "beams",
        "metrics_file",
        "predictions_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    columns = ["run", "kind", "model", "rouge1", "rouge2", "rougeL", "loss", "gen_len"]
    lines = ["# Test Results", "", "| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate downloaded Kaggle checkpoints on local test data.")
    parser.add_argument("--runs_root", required=True, help="Folder containing downloaded run dirs.")
    parser.add_argument("--test_file", required=True, help="Test parquet with article and summary columns.")
    parser.add_argument("--out_dir", default="local_test_eval", help="Output folder for metrics/predictions.")
    parser.add_argument("--run_glob", default="*", help="Glob for run dirs under runs_root.")
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--generation_max_new_tokens", type=int, default=None)
    parser.add_argument("--generation_num_beams", type=int, default=None)
    args = parser.parse_args()

    configure_logging()
    runs_root = Path(args.runs_root)
    test_file = Path(args.test_file).resolve()
    out_dir = Path(args.out_dir)
    run_dirs = find_run_dirs(runs_root, args.run_glob)
    if not run_dirs:
        raise FileNotFoundError(f"No run dirs with resolved_config.json found under {runs_root}")

    rows = [
        evaluate_run(
            run_dir,
            test_file,
            out_dir,
            args.max_test_samples,
            args.eval_batch_size,
            args.generation_max_new_tokens,
            args.generation_num_beams,
        )
        for run_dir in run_dirs
    ]
    rows.sort(key=lambda row: to_float(row.get("rougeL")), reverse=True)
    write_csv(rows, out_dir / "test_results.csv")
    write_markdown(rows, out_dir / "test_results.md")
    save_json(rows[0], out_dir / "best_test_run.json")
    print((out_dir / "test_results.md").read_text(encoding="utf-8"))
    print("CSV:", out_dir / "test_results.csv")
    print("Best:", out_dir / "best_test_run.json")


if __name__ == "__main__":
    main()
