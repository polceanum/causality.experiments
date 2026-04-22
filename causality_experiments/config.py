from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {config_path} must contain a mapping.")
    data.setdefault("name", config_path.stem)
    return data


def require_keys(config: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{context} is missing required key(s): {joined}")
