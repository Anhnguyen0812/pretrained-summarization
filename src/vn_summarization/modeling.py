from __future__ import annotations

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

    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        use_fast=bool(model_cfg.get("use_fast_tokenizer", True)),
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    model_config = AutoConfig.from_pretrained(
        name_or_path,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
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
