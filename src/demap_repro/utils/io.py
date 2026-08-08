"""Small I/O helpers (primarily to standardize directory creation).

These wrappers are intentionally lightweight and keep pandas/numpy as the
underlying implementations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from .paths import ensure_dir


def read_parquet(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_parquet(path, **kwargs)


def write_parquet(df: pd.DataFrame, path: str | Path, *, index: bool = False, **kwargs) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_parquet(p, index=index, **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path, *, index: bool = False, **kwargs) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=index, **kwargs)


def write_json(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=indent, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))
