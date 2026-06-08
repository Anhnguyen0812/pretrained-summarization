# Experiment Plan

## Goal

Maximize validation ROUGE while avoiding overfit on the 10,775 train and 1,349 validation samples.

Primary metric: ROUGE-L.

Required metrics: ROUGE-1, ROUGE-2, ROUGE-L.

Secondary checks:

- Generated length near the target distribution.
- No repeated phrases.
- No hallucinated named entities, dates, or numbers.
- Good coverage of main article points.

## Stage 0: Smoke Test

Run:

```powershell
python -m vn_summarization.train --config configs/smoke_mt5_small.yaml
```

Pass condition:

- Dataset loads.
- Tokenizer works.
- ROUGE computes.
- Checkpoint saves to `outputs/smoke_mt5_small/best`.

## Stage 1: Main Baselines

Train:

```powershell
python -m vn_summarization.train --config configs/vit5_base.yaml
python -m vn_summarization.train --config configs/bartpho_syllable.yaml
python -m vn_summarization.train --config configs/mt5_base.yaml
```

Expected first choice: `vit5_base`, because it is Vietnamese-specific and text-to-text.

Expected risk:

- `mt5_base`: slower, more multilingual capacity than needed.
- `bartpho_syllable`: strong Vietnamese denoising pretraining, but may need more decode tuning.

## Stage 2: High-score Branches

Train:

```powershell
python -m vn_summarization.train --config configs/vit5_base_lora.yaml
python -m vn_summarization.train --config configs/vit5_news_warmstart.yaml
```

Interpretation:

- If full ViT5 beats LoRA, continue full fine-tune with LR and length ablations.
- If LoRA is close or better, continue LoRA rank and LR ablations; this usually overfits less.
- If VietNews warm-start wins by a lot, use it as best score but disclose that it was already summarization-tuned.

## Stage 3: Ablations

Run one change at a time against the current best config.

Decode ablation:

- `generation.num_beams=6`
- `generation.length_penalty=1.1`
- `generation.repetition_penalty=1.08`
- `generation.no_repeat_ngram_size=4`

Input-length ablation:

- `data.max_source_length=512`
- `data.max_source_length=1024`

Regularization ablation:

- `training.learning_rate=0.00002`
- `training.label_smoothing_factor=0.05`
- `training.dropout=0.15`
- `training.weight_decay=0.02`

LoRA ablation:

- `lora.r=8`
- `lora.r=32`
- `lora.lora_dropout=0.1`

Example override command:

```powershell
python -m vn_summarization.train --config configs/vit5_base.yaml `
  --set training.output_dir=outputs/vit5_base_lr2e5 `
  --set training.learning_rate=0.00002
```

## Stage 4: Choose Next Direction

Use:

```powershell
python -m vn_summarization.compare_runs --root outputs --metric eval_rougeL
```

Decision tree:

- Best ROUGE-L and clean samples: keep model, tune generation only.
- ROUGE high but summaries too long: reduce `max_length` or raise `length_penalty` only if outputs are verbose.
- ROUGE high but repeated phrases: increase `no_repeat_ngram_size` to 4 or `repetition_penalty` to 1.08.
- Train loss drops, validation ROUGE flat: reduce epochs, lower LR, increase dropout, or switch to LoRA.
- Summary misses later article facts: increase `max_source_length` to 1024 if GPU allows.
- Hallucination with names/numbers: lower LR, use stronger early stopping, and prefer LoRA/warm-start only if samples improve.

## Report Table Template

| Model | Fine-tune type | Params trained | ROUGE-1 | ROUGE-2 | ROUGE-L | Notes |
|---|---:|---:|---:|---:|---:|---|
| ViT5-base | Full | all | | | | baseline |
| ViT5-base | LoRA r16 | adapter only | | | | anti-overfit |
| ViT5 VietNews | Full | all | | | | warm-start |
| BARTpho syllable | Full | all | | | | architecture comparison |
| mT5-base | Full | all | | | | multilingual comparison |

