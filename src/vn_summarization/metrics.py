from __future__ import annotations

from typing import Any, Callable

import evaluate
import numpy as np


def _postprocess_text(texts: list[str]) -> list[str]:
    return [text.strip() for text in texts]

def _sanitize_token_ids(values, tokenizer):
    ids = np.asarray(values)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    vocab_size = len(tokenizer)
    ids = np.where(ids < 0, pad_token_id, ids)
    ids = np.where(ids >= vocab_size, pad_token_id, ids)
    return ids.astype(np.int64, copy=False)


def build_compute_metrics(tokenizer) -> Callable[[Any], dict[str, float]]:
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds) -> dict[str, float]:
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = _sanitize_token_ids(preds, tokenizer)
        labels = _sanitize_token_ids(np.where(labels != -100, labels, tokenizer.pad_token_id), tokenizer)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = _postprocess_text(decoded_preds)
        decoded_labels = _postprocess_text(decoded_labels)

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            rouge_types=["rouge1", "rouge2", "rougeL"],
            use_stemmer=False,
        )
        result = {key: round(value * 100, 4) for key, value in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = round(float(np.mean(prediction_lens)), 4)
        return result

    return compute_metrics

