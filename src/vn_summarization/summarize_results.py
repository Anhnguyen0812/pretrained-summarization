from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRIC_NAMES = ["loss", "rouge1", "rouge2", "rougeL", "gen_len", "runtime"]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metric(metrics: dict[str, Any], name: str) -> Any:
    candidates = [name, f"eval_{name}", f"test_{name}", f"predict_{name}"]
    for key in candidates:
        if key in metrics:
            return metrics[key]
    for key, value in metrics.items():
        if key.endswith("_" + name):
            return value
    return ""


def _run_name(path: Path, root: Path) -> str:
    try:
        return path.parent.relative_to(root).as_posix()
    except ValueError:
        return path.parent.as_posix()


def _config_summary(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "resolved_config.json"
    if not config_path.exists():
        return {}
    try:
        config = _load_json(config_path)
    except json.JSONDecodeError:
        return {}

    model = config.get("model", {})
    training = config.get("training", {})
    data = config.get("data", {})
    generation = config.get("generation", {})
    lora = config.get("lora", {})
    return {
        "model": model.get("name_or_path", ""),
        "lora": bool(lora.get("enabled", False)),
        "lr": training.get("learning_rate", ""),
        "epochs": training.get("num_train_epochs", ""),
        "max_steps": training.get("max_steps", ""),
        "source_len": data.get("max_source_length", ""),
        "target_len": data.get("max_target_length", ""),
        "beams": generation.get("num_beams", ""),
        "length_penalty": generation.get("length_penalty", ""),
        "repeat_penalty": generation.get("repetition_penalty", ""),
    }


def collect_rows(root: Path) -> list[dict[str, Any]]:
    metric_files = sorted(root.glob("**/eval_results.json")) + sorted(
        root.glob("**/validation_metrics.json")
    )
    seen: set[Path] = set()
    rows: list[dict[str, Any]] = []
    for path in metric_files:
        if path in seen:
            continue
        seen.add(path)
        metrics = _load_json(path)
        run_dir = path.parent
        row: dict[str, Any] = {
            "run": _run_name(path, root),
            "metric_file": path.relative_to(root).as_posix(),
        }
        row.update(_config_summary(run_dir))
        for name in METRIC_NAMES:
            row[name] = _metric(metrics, name)
        rows.append(row)

    rows.sort(key=lambda row: _to_float(row.get("rougeL")), reverse=True)
    return rows


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "run",
        "model",
        "lora",
        "rouge1",
        "rouge2",
        "rougeL",
        "gen_len",
        "loss",
        "runtime",
        "lr",
        "epochs",
        "max_steps",
        "source_len",
        "target_len",
        "beams",
        "length_penalty",
        "repeat_penalty",
        "metric_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["run", "model", "lora", "rouge1", "rouge2", "rougeL", "gen_len", "loss"]
    lines = ["# Result Summary", "", "| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(column, "")) for column in columns) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment metrics.")
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--md", default=None)
    parser.add_argument("--best_json", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    rows = collect_rows(root)
    if not rows:
        print(f"No metrics found under {root}")
        return

    csv_path = Path(args.csv) if args.csv else root / "summary_results.csv"
    md_path = Path(args.md) if args.md else root / "summary_results.md"
    best_path = Path(args.best_json) if args.best_json else root / "best_run.json"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    best_path.write_text(json.dumps(rows[0], ensure_ascii=False, indent=2), encoding="utf-8")

    print(md_path.read_text(encoding="utf-8"))
    print(f"CSV: {csv_path}")
    print(f"Best: {best_path}")


if __name__ == "__main__":
    main()

