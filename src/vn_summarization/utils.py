from __future__ import annotations

import json
import logging
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


LOGGER = logging.getLogger("vn_summarization")


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )


def load_yaml(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f), config_path


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path_value: str | Path, config_path: Path | None = None) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path

    candidates = [Path.cwd() / path]
    if config_path is not None:
        candidates.extend([config_path.parent / path, config_path.parent.parent / path])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    basename = path.name
    search_roots = [Path.cwd(), Path.cwd().parent]
    if Path("/kaggle/input").exists():
        search_roots.append(Path("/kaggle/input"))
    if Path("/kaggle/working").exists():
        search_roots.append(Path("/kaggle/working"))

    for root in search_roots:
        try:
            matches = sorted(root.rglob(basename))
        except OSError:
            continue
        if matches:
            return matches[0].resolve()
    return candidates[0].resolve()


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def parse_override(raw: str) -> dict[str, Any]:
    key, sep, value = raw.partition("=")
    if not sep or not key:
        raise ValueError(f"Invalid override: {raw}. Expected key=value.")

    try:
        parsed_value = yaml.safe_load(value)
    except yaml.YAMLError:
        parsed_value = value

    parts = key.split(".")
    current: dict[str, Any] = {}
    root = current
    for part in parts[:-1]:
        current[part] = {}
        current = current[part]
    current[parts[-1]] = parsed_value
    return root


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    if not overrides:
        return config
    updated = deepcopy(config)
    for raw in overrides:
        updated = deep_update(updated, parse_override(raw))
    return updated


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"total": int(total), "trainable": int(trainable)}


def infer_precision(precision: str) -> tuple[bool, bool]:
    precision = precision.lower()
    if precision == "fp16":
        return True, False
    if precision == "bf16":
        return False, True
    if precision == "fp32":
        return False, False
    if precision != "auto":
        raise ValueError(f"Unsupported precision: {precision}")

    if not torch.cuda.is_available():
        return False, False
    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    return (not bf16_supported), bf16_supported


def log_runtime() -> None:
    LOGGER.info("python pid=%s cwd=%s", os.getpid(), Path.cwd())
    if torch.cuda.is_available():
        LOGGER.info("cuda devices=%s", torch.cuda.device_count())
        for idx in range(torch.cuda.device_count()):
            LOGGER.info("cuda:%s %s", idx, torch.cuda.get_device_name(idx))
    else:
        LOGGER.info("cuda unavailable")
