# Vietnamese Abstractive Summarization Fine-tuning

This repo covers requirement 2 of the assignment: fine-tune pretrained language models under 3B parameters for abstractive summarization.

Dataset expected in the parent folder:

- `../train-00000-of-00001.parquet`
- `../valid-00000-of-00001.parquet`

Both files must contain:

- `article`: source document
- `summary`: target summary

## Kết quả fine-tune đã đạt được

Dưới đây là kết quả fine-tune đã ghi nhận trong repo (tham khảo thêm tại [readme/12_ket_qua_va_dien_giai.md](readme/12_ket_qua_va_dien_giai.md)):

| Mô hình | Validation ROUGE-1 | Validation ROUGE-2 | Validation ROUGE-L | Test ROUGE-1 | Test ROUGE-2 | Test ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|
| Transformer scratch | 0.5768 | 0.2087 | 0.3053 | 0.5747 | 0.2038 | 0.3045 |
| ViT5-base full fine-tuning | 0.7417 | 0.4709 | 0.4924 | 0.7422 | 0.4675 | 0.4889 |
| ViT5-base LoRA rank 16 | 0.7297 | 0.4484 | 0.4733 | 0.7262 | 0.4408 | 0.4663 |
| BARTpho-syllable full fine-tuning | - | - | - | 0.7347 | 0.4617 | 0.4807 |
| ViT5 VietNews warm start | - | - | - | 0.7161 | 0.4426 | 0.4728 |

Kết luận nhanh: ViT5-base full fine-tuning là cấu hình mạnh nhất trong các thí nghiệm hiện có, với ROUGE-L test khoảng 0.4889.

## Kết quả các mô hình causal LM

Các số dưới đây được trích từ báo cáo LaTeX và phản ánh các thí nghiệm ablation trên tập con validation cố định (500 mẫu), nên không nên so sánh trực tiếp với kết quả full validation/test ở bảng trên.

| Mô hình / cấu hình | Ngữ cảnh | LoRA rank | Mẫu train | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3, prompt cơ bản | 768 | 8 | 2500 | 0.5938 | 0.2729 | 0.3379 |
| Qwen3, prompt nghiêm ngặt | 768 | 8 | 2500 | 0.6044 | 0.2682 | 0.3333 |
| Qwen3, prompt nghiêm ngặt | 768 | 16 | 2500 | 0.6565 | 0.2987 | 0.3611 |
| Qwen3, prompt nghiêm ngặt | 1024 | 8 | 2500 | 0.6652 | 0.3122 | 0.3792 |
| Qwen3, prompt nghiêm ngặt, thêm dữ liệu, beam 2 | 1024 | 16 | 7000 | 0.6834 | 0.3213 | 0.3787 |
| DeepSeek-R1-Distill-Qwen-1.5B | 768 | 8 | 2500 | 0.4481 | 0.1695 | 0.2619 |

Nhận xét quan trọng từ báo cáo:

- Tăng ngữ cảnh từ 768 lên 1024 token là cải tiến hiệu quả nhất trong các thí nghiệm causal LM quan sát được.
- Tăng LoRA rank giúp cải thiện ROUGE-1/2, nhưng không tiếp tục nâng ROUGE-L một cách rõ rệt.
- DeepSeek-R1-Distill-Qwen-1.5B thấp hơn Qwen3 trong các thí nghiệm này.

## Minh chứng Kaggle

Các notebook Kaggle dùng để chạy/đánh giá các thí nghiệm này:

- [NLP_sumarization_Causal_LM](https://www.kaggle.com/code/anhnguyen0812/nlp-sumarization-causal-lm)
- [nlp-finetune](https://www.kaggle.com/code/anhnguyen0812/nlp-finetune)
- [continue](https://www.kaggle.com/code/anhnguyenphi/continue)
- [test-nlp](https://www.kaggle.com/code/anhnguyenphi/test-nlp)

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

This repo uses `transformers>=4.51.0,<5` and `tokenizers>=0.22.0,<=0.23.0`. Qwen3 needs Transformers 4.51+; recent Transformers builds require Tokenizers 0.22.x-0.23.x. Keep Transformers below v5 because ViT5 tokenizer conversion can fail on legacy metadata.

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

## Evaluate Downloaded Kaggle Checkpoints On Local Test

Download each Kaggle run folder with at least:

- `resolved_config.json`
- `best/adapter_config.json`
- `best/adapter_model.safetensors`
- tokenizer files inside `best/`

Keep folders under one root, for example:

```text
downloaded_kaggle_runs/
  qwen3_prompt_strict_r8_300s_report/
    resolved_config.json
    best/
      adapter_config.json
      adapter_model.safetensors
      tokenizer.json
      tokenizer_config.json
  qwen3_all_on_r16_700s_report/
    resolved_config.json
    best/
      ...
```

Then evaluate every downloaded run on `../test-00000-of-00001.parquet`:

```powershell
cd pretrained-summarization
python -m pip install -e .
python -m vn_summarization.evaluate_runs_on_test `
  --runs_root ..\downloaded_kaggle_runs `
  --test_file ..\test-00000-of-00001.parquet `
  --out_dir ..\local_test_eval
```

For a quick local check before full test, add `--max_test_samples 50`. Full Qwen/DeepSeek generation on CPU is very slow; use CUDA locally if available.

Outputs:

- `local_test_eval/test_results.csv`
- `local_test_eval/test_results.md`
- `local_test_eval/best_test_run.json`
- `local_test_eval/<run>/predictions_test.jsonl`

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
