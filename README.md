# Vietnamese Abstractive Summarization Fine-tuning

This repo covers requirement 2 of the assignment: fine-tune pretrained language models under 3B parameters for abstractive summarization.

Dataset expected in the parent folder:

- `../train-00000-of-00001.parquet`
- `../valid-00000-of-00001.parquet`

Both files must contain:

- `article`: source document
- `summary`: target summary

## Environment

Recommended: Python 3.10 or 3.11 with CUDA GPU. Local CPU can run only smoke tests.

```powershell
cd pretrained-summarization
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

On Colab/Kaggle, copy this folder and the two parquet files, then run the same install command from the repo root.

## Model Priority

Train in this order:

1. `VietAI/vit5-base`: main Vietnamese T5 baseline.
2. `VietAI/vit5-base` with LoRA: lower overfit risk, faster, good for ablation.
3. `VietAI/vit5-base-vietnews-summarization`: strong warm-start, but disclose possible domain overlap with news summarization.
4. `vinai/bartpho-syllable`: Vietnamese BART-style comparison, no external word segmentation needed.
5. `google/mt5-base`: multilingual fallback, usually heavier and slower than ViT5.

Official model cards:

- https://huggingface.co/VietAI/vit5-base
- https://huggingface.co/VietAI/vit5-base-vietnews-summarization
- https://huggingface.co/vinai/bartpho-syllable
- https://huggingface.co/google/mt5-base

## Quick Smoke Test

This downloads `google/mt5-small`, trains 2 steps on 16 samples, and verifies the pipeline.

```powershell
python -m vn_summarization.train --config configs/smoke_mt5_small.yaml
```

## Train Main Runs

Single run:

```powershell
python -m vn_summarization.train --config configs/vit5_base.yaml
```

## Kaggle Run

Kaggle setup:

1. Enable GPU accelerator.
2. Enable internet, at least for first model download.
3. Add the two parquet files as a Kaggle Dataset.
4. Upload this repo, or clone it into `/kaggle/working/pretrained-summarization`.

Run from a Kaggle notebook cell:

```bash
cd /kaggle/working/pretrained-summarization
python -m pip install -q -e .
python -m vn_summarization.train --config configs/vit5_base.yaml
```

This repo pins `transformers==4.46.3` and `tokenizers==0.20.3`. Do not use Transformers v5 for ViT5; its tokenizer conversion path can fail on the legacy ViT5 tokenizer metadata.

For Kaggle T4x2, prefer DDP through Accelerate:

```bash
cd /kaggle/working/pretrained-summarization
python -m pip install -q -e .
python -m accelerate.commands.launch --multi_gpu --num_processes 2 --mixed_precision fp16 \
  -m vn_summarization.train --config configs/vit5_base_t4x2.yaml
```

LoRA on T4x2:

```bash
python -m accelerate.commands.launch --multi_gpu --num_processes 2 --mixed_precision fp16 \
  -m vn_summarization.train --config configs/vit5_base_lora_t4x2.yaml
```

The configs use `../train-00000-of-00001.parquet` and `../valid-00000-of-00001.parquet`, but the loader also searches `/kaggle/input/**/train-00000-of-00001.parquet` and `/kaggle/input/**/valid-00000-of-00001.parquet` automatically.

For a quick low-cost check:

```bash
cd /kaggle/working/pretrained-summarization
python -m pip install -q -e .
python -m vn_summarization.train --config configs/smoke_mt5_small.yaml
```

Priority batch:

```powershell
.\scripts\train_priority.ps1
```

Linux/Colab:

```bash
bash scripts/train_priority.sh
```

## Evaluate

```powershell
python -m vn_summarization.evaluate `
  --config configs/vit5_base.yaml `
  --model_path outputs/vit5_base/best `
  --predictions_path outputs/vit5_base/predictions_valid.jsonl
```

Compare all finished runs:

```powershell
python -m vn_summarization.compare_runs --root outputs --metric eval_rougeL
```

## Predict

```powershell
python -m vn_summarization.predict `
  --model_path outputs/vit5_base/best `
  --text "<paste Vietnamese article here>"
```

For BARTpho, use empty prefix:

```powershell
python -m vn_summarization.predict --model_path outputs/bartpho_syllable/best --prefix "" --text "..."
```

## Built-in Anti-overfitting Controls

- Validation checkpoint selection by `rougeL`.
- Early stopping.
- Label smoothing.
- Weight decay.
- Dropout adjustment before model loading.
- Cosine learning-rate schedule with warmup.
- Small effective batch through gradient accumulation.
- LoRA option to train fewer parameters.
- No-repeat n-gram and repetition penalty during generation.

## Best-run Decision Rule

Prefer the run with the best validation `eval_rougeL`, then check:

- `eval_rouge2`: higher means better phrase-level content retention.
- `eval_gen_len`: must stay near target summaries, about 100 Vietnamese syllable tokens in this dataset.
- Prediction samples: reject models with repeated phrases, hallucinated names/numbers, or summaries that only copy the first sentence.
- Train/eval loss gap: if train loss keeps falling but ROUGE stops improving, lower epochs or switch to LoRA/lower LR.

## Practical Next Improvements

After first full training, use the best model and run these ablations one at a time:

1. Decode search: beams `4` vs `6`, length penalty `0.9`, `1.0`, `1.1`, repetition penalty `1.05`, `1.08`.
2. Input length: `512` vs `768` vs `1024`, if GPU memory allows.
3. Target length: `128` vs `160`, based on `eval_gen_len` and qualitative quality.
4. Regularization: label smoothing `0.05` vs `0.1`, dropout `0.1` vs `0.15`, weight decay `0.01` vs `0.02`.
5. LR: `1e-5`, `2e-5`, `3e-5` for full fine-tune; `5e-5`, `1e-4`, `2e-4` for LoRA.
6. LoRA rank: `8`, `16`, `32`; keep dropout `0.05` or `0.1`.
7. Freeze encoder for 1 quick run if validation overfit appears early.
8. Report qualitative error analysis from `predictions_valid.jsonl`.

Do not tune many knobs at once. Keep one baseline run fixed so the report has a clean comparison.
