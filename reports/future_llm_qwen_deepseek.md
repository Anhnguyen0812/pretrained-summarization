# Future Work: Qwen/DeepSeek Causal LLMs

This note is intentionally separated from the current ViT5/BARTpho seq2seq pipeline. Do not mix these experiments into the main runs until the pretrained fine-tuning baseline is stable.

## Why Later

The current repo is built for encoder-decoder summarization models through `AutoModelForSeq2SeqLM`:

- ViT5
- BARTpho
- mT5

Qwen and DeepSeek-style models are usually causal LMs. They need a different training path:

- `AutoModelForCausalLM`
- prompt-response formatting
- label masking so loss is computed only on the answer span
- LoRA/QLoRA to fit Kaggle GPU memory
- different generation defaults
- extra hallucination checks

Adding this now would increase scope and make the required assignment baseline harder to finish cleanly.

## Candidate Models To Verify Later

Before using any model, verify its HuggingFace model card, parameter count, license, and context length. The assignment requires pretrained models under 3B parameters.

Possible candidates to check later:

- Qwen small instruct models under 3B.
- Qwen summarization-capable instruct checkpoints under 3B.
- DeepSeek distilled Qwen checkpoints under 3B.
- Avoid DeepSeek 7B/8B/14B/32B models because they violate the 3B limit.

Use only candidates that satisfy:

- `< 3B` parameters.
- License allows academic fine-tuning/use.
- Supports Vietnamese reasonably well.
- Fits Kaggle T4x2 with LoRA or QLoRA.

## Proposed Training Method

Use supervised fine-tuning with LoRA/QLoRA.

Prompt format:

```text
Bạn là hệ thống tóm tắt văn bản tiếng Việt. Hãy tóm tắt văn bản sau, giữ các ý chính, không bịa thông tin.

Văn bản:
{article}

Tóm tắt:
{summary}
```

At training time:

- Input contains instruction + article + target summary.
- Labels are masked with `-100` for instruction/article tokens.
- Loss is computed only over summary tokens.

At inference time:

```text
Bạn là hệ thống tóm tắt văn bản tiếng Việt. Hãy tóm tắt văn bản sau, giữ các ý chính, không bịa thông tin.

Văn bản:
{article}

Tóm tắt:
```

The model generates only the summary continuation.

## Required Repo Changes

Add separate files instead of modifying the seq2seq path:

- `src/vn_summarization/train_causal_lm.py`
- `src/vn_summarization/evaluate_causal_lm.py`
- `src/vn_summarization/predict_causal_lm.py`
- `configs/qwen_lora.yaml`
- `configs/deepseek_distill_lora.yaml`

Needed implementation pieces:

- Load with `AutoModelForCausalLM`.
- Load tokenizer with left padding for generation if needed.
- Add LoRA through PEFT.
- Optional 4-bit quantization if Kaggle memory is tight.
- Build custom data collator with label masking.
- Generate summaries from prompts and strip prompt text before ROUGE.
- Save predictions to JSONL for error analysis.

## Suggested Hyperparameters

Start conservative to avoid overfitting:

| Setting | Value |
|---|---:|
| LoRA rank | 8 or 16 |
| LoRA alpha | 16 or 32 |
| LoRA dropout | 0.05 to 0.1 |
| LR | 1e-4 to 2e-4 |
| epochs | 2 to 4 |
| max source length | 1024 to 1536 tokens |
| max target length | 160 to 220 tokens |
| weight decay | 0.01 |
| warmup ratio | 0.03 to 0.06 |
| early stopping | validation ROUGE-L |

For T4x2:

- Use `accelerate launch --multi_gpu --num_processes 2`.
- Prefer LoRA first.
- Use QLoRA only if full precision LoRA does not fit.

## Evaluation

Use the same required metrics:

- ROUGE-1
- ROUGE-2
- ROUGE-L

Also inspect samples for:

- hallucinated names, numbers, dates
- repeated phrases
- overly verbose summaries
- summaries in the wrong language
- instruction leakage such as repeating `Tóm tắt:`

## When To Continue This Direction

Only continue Qwen/DeepSeek if one of these is true:

- ViT5/BARTpho plateau and LLM LoRA gives higher ROUGE-L.
- LLM outputs are qualitatively better with fewer hallucinations.
- Kaggle T4x2 can train within session limits.

Stop this direction if:

- validation ROUGE is lower than ViT5 by a large margin
- outputs copy prompt text
- hallucination rate is high
- training is too slow for Kaggle session limits

## Reporting Angle

If implemented later, present it as an additional improvement, not the required pretrained baseline.

Report comparison table:

| Model | Type | Train method | ROUGE-1 | ROUGE-2 | ROUGE-L | Notes |
|---|---|---|---:|---:|---:|---|
| ViT5-base | encoder-decoder | full fine-tune | | | | required baseline |
| ViT5-base | encoder-decoder | LoRA | | | | anti-overfit |
| Qwen small | causal LM | LoRA/QLoRA | | | | future improvement |
| DeepSeek distill small | causal LM | LoRA/QLoRA | | | | future improvement |

