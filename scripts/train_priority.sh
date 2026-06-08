#!/usr/bin/env bash
set -euo pipefail

configs=(
  "configs/vit5_base.yaml"
  "configs/vit5_base_lora.yaml"
  "configs/vit5_news_warmstart.yaml"
  "configs/bartpho_syllable.yaml"
  "configs/mt5_base.yaml"
)

for config in "${configs[@]}"; do
  echo "Running ${config}"
  python -m vn_summarization.train --config "${config}"
done

python -m vn_summarization.compare_runs --root outputs --metric eval_rougeL

