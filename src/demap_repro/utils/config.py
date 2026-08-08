"""YAML configuration helpers.

This module is intentionally small: it loads YAML files into dictionaries and
supports a simple recursive merge for overrides.

We keep schemas lightweight so the repo can evolve quickly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML root to be a mapping (dict), got {type(data)} in {p}")
    return data


# Backwards-compatible alias used throughout the repo.
def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file.

    Historically some modules used the name `load_config`. Keeping this alias
    avoids churn and keeps CLI modules simple.
    """

    return load_yaml(path)


def deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge *override* into *base* and return *base* (mutated)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, Mapping):
            deep_merge(base[k], v)  # type: ignore[arg-type]
        else:
            base[k] = v
    return base


def load_and_merge_yaml(*paths: str | Path) -> Dict[str, Any]:
    """Load multiple YAML files in order and deep-merge them (later wins)."""
    merged: Dict[str, Any] = {}
    for p in paths:
        deep_merge(merged, load_yaml(p))
    return merged
