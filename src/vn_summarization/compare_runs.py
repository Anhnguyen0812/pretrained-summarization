from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(root.glob("**/eval_results.json")):
        with path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append({"run": str(path.parent.relative_to(root)), **metrics})
    for path in sorted(root.glob("**/validation_metrics.json")):
        with path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append({"run": str(path.parent.relative_to(root)), **metrics})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare run metrics under an outputs directory.")
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--metric", default="eval_rougeL")
    args = parser.parse_args()

    rows = load_metrics(Path(args.root))
    rows.sort(key=lambda row: float(row.get(args.metric, -1)), reverse=True)
    if not rows:
        print("No metrics found.")
        return

    keys = ["run", "eval_rouge1", "eval_rouge2", "eval_rougeL", "eval_gen_len"]
    print("\t".join(keys))
    for row in rows:
        print("\t".join(str(row.get(key, "")) for key in keys))


if __name__ == "__main__":
    main()

