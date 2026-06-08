from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from .utils import LOGGER, count_parameters


def _set_dropout(model_config, dropout: float | None) -> None:
    if dropout is None:
        return
    for attr in (
        "dropout",
        "dropout_rate",
        "attention_dropout",
        "activation_dropout",
        "classifier_dropout",
    ):
        if hasattr(model_config, attr):
            setattr(model_config, attr, float(dropout))


def load_tokenizer_and_model(config: dict[str, Any], for_training: bool = True):
    model_cfg = config["model"]
    training_cfg = config.get("training", {})
    name_or_path = model_cfg["name_or_path"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    cache_dir = model_cfg.get("cache_dir")

    model_config = AutoConfig.from_pretrained(
        name_or_path,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    tokenizer = _load_tokenizer(name_or_path, model_config, model_cfg, trust_remote_code, cache_dir)
    _set_dropout(model_config, training_cfg.get("dropout"))
    model = AutoModelForSeq2SeqLM.from_pretrained(
        name_or_path,
        config=model_config,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    if for_training and bool(training_cfg.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if for_training and bool(training_cfg.get("freeze_encoder", False)):
        LOGGER.info("freezing encoder")
        for parameter in model.get_encoder().parameters():
            parameter.requires_grad = False

    generation_cfg = config.get("generation", {})
    for key, value in generation_cfg.items():
        if hasattr(model.generation_config, key):
            setattr(model.generation_config, key, value)

    params = count_parameters(model)
    max_parameters = int(model_cfg.get("max_parameters", 3_000_000_000))
    if params["total"] >= max_parameters:
        raise ValueError(
            f"Model has {params['total']} parameters, which violates limit {max_parameters}."
        )
    LOGGER.info("model=%s params=%s", name_or_path, params)
    return tokenizer, model


def _load_tokenizer(name_or_path: str, model_config, model_cfg: dict[str, Any], trust_remote_code: bool, cache_dir):
    use_fast = bool(model_cfg.get("use_fast_tokenizer", True))
    tokenizer_errors: list[str] = []

    # ViT5 uses a T5 SentencePiece tokenizer. Some recent Transformers/tokenizers
    # builds on Kaggle fail while converting this tokenizer to native fast format.
    if getattr(model_config, "model_type", "") == "t5":
        try:
            return _load_t5_sentencepiece_tokenizer(name_or_path, cache_dir)
        except Exception as exc:
            tokenizer_errors.append(f"direct T5 SentencePiece tokenizer failed: {exc!r}")

    for fast in ([use_fast, False] if use_fast else [False, True]):
        try:
            return AutoTokenizer.from_pretrained(
                name_or_path,
                use_fast=fast,
                trust_remote_code=trust_remote_code,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            tokenizer_errors.append(f"AutoTokenizer use_fast={fast} failed: {exc!r}")

    raise RuntimeError(
        "Could not load tokenizer for "
        f"{name_or_path}. Attempts:\n- " + "\n- ".join(tokenizer_errors)
    )


def _load_t5_sentencepiece_tokenizer(name_or_path: str, cache_dir):
    from huggingface_hub import hf_hub_download
    from transformers import T5Tokenizer

    model_path = Path(name_or_path)
    if model_path.exists():
        spiece_path = model_path / "spiece.model"
        tokenizer_config_path = model_path / "tokenizer_config.json"
    else:
        spiece_path = Path(
            hf_hub_download(name_or_path, filename="spiece.model", cache_dir=cache_dir)
        )
        try:
            tokenizer_config_path = Path(
                hf_hub_download(name_or_path, filename="tokenizer_config.json", cache_dir=cache_dir)
            )
        except Exception:
            tokenizer_config_path = None

    if not spiece_path.exists():
        raise FileNotFoundError(f"Missing SentencePiece model: {spiece_path}")

    tokenizer_config = _read_tokenizer_config(tokenizer_config_path)
    extra_ids = int(tokenizer_config.get("extra_ids", 100))

    kwargs = {
        "eos_token": tokenizer_config.get("eos_token", "</s>"),
        "unk_token": tokenizer_config.get("unk_token", "<unk>"),
        "pad_token": tokenizer_config.get("pad_token", "<pad>"),
        "extra_ids": extra_ids,
        "sp_model_kwargs": tokenizer_config.get("sp_model_kwargs", {}),
    }

    signature = inspect.signature(T5Tokenizer.__init__)
    if "legacy" in signature.parameters:
        kwargs["legacy"] = False
    if "vocab_file" in signature.parameters:
        kwargs["vocab_file"] = str(spiece_path)
        tokenizer = T5Tokenizer(**kwargs)
    elif "vocab" in signature.parameters:
        kwargs["vocab"] = str(spiece_path)
        tokenizer = T5Tokenizer(**kwargs)
    else:
        tokenizer = T5Tokenizer(str(spiece_path), **kwargs)

    return tokenizer


def _read_tokenizer_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def apply_lora_if_enabled(model, config: dict[str, Any]):
    lora_cfg = config.get("lora", {})
    if not lora_cfg.get("enabled", False):
        return model

    from peft import LoraConfig, TaskType, get_peft_model

    target_modules = lora_cfg.get("target_modules", "auto")
    if target_modules == "auto":
        model_type = getattr(model.config, "model_type", "")
        if model_type in {"t5", "mt5"}:
            target_modules = ["q", "v"]
        elif model_type in {"mbart", "bart"}:
            target_modules = ["q_proj", "v_proj"]
        else:
            target_modules = ["q", "v", "q_proj", "v_proj"]

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    LOGGER.info("lora target_modules=%s params=%s", target_modules, count_parameters(model))
    return model
