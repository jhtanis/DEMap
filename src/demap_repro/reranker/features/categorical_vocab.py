#!/usr/bin/env python3
"""Fixed, training-derived categorical vocabulary for HGBC provenance features.

Replaces the table-relative ``pd.Categorical(col).codes`` encoding (which assigned
codes from the sorted unique values of *whatever table was being scored*, so the
same string could receive different codes depending on batch composition) with a
vocabulary that is:

  * fitted on the TRAINING split only;
  * persisted next to the model (``categorical_vocab.json``);
  * applied identically at train and inference time — the same input string always
    maps to the same code, regardless of row order, chunking, or which other
    datasets happen to be in the table.

Schema v2 — single reserved category, always code 0:

  * ``__FALLBACK__``: any value without usable provenance — absent (None/NaN),
    empty string, or a nonempty value not in the fitted training vocabulary.
    Fallback examples are created during training by deterministic query-level
    provenance dropout (see ``demap.features.provenance_dropout``), so the model
    has a genuinely TRAINED branch for this state.

The *cause* of fallback (missing vs unseen) is deliberately not a model
distinction — with zero natural examples of either state in training, two
synthetic states would be trained on identically-distributed maskings, a fake
distinction (audit 2026-08-03). The cause IS distinguishable in telemetry:
:meth:`CategoricalVocab.telemetry` reports known / missing / unknown counts for
manifests and monitoring.

Codes are dense integers in ``[0, len(vocab))`` and are intended for sklearn
``HistGradientBoosting*`` **native categorical** handling
(``categorical_features=...``), which treats them as unordered categories via
set-based splits — never as ordinal magnitudes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np
import pandas as pd

FALLBACK_TOKEN = "__FALLBACK__"
RESERVED_TOKENS = (FALLBACK_TOKEN,)
FALLBACK_CODE = 0

VOCAB_FORMAT = "demap_categorical_vocab"
VOCAB_SCHEMA_VERSION = 2
VOCAB_FILENAME = "categorical_vocab.json"


def _is_missing(values: pd.Series) -> pd.Series:
    """True where the value is absent (None/NaN) or the empty string."""
    s = values.astype(object)
    return pd.isna(s) | (s.astype(str) == "")


def _normalize(values: pd.Series) -> pd.Series:
    """Map absent/empty values to ``__FALLBACK__``; everything else to ``str``.
    Row-local and vectorized; no table-relative state."""
    out = values.astype(object).astype(str)
    out[_is_missing(values).values] = FALLBACK_TOKEN
    return out


class CategoricalVocab:
    """Per-column fixed vocabularies: token list index == integer code."""

    def __init__(self, columns: Dict[str, List[str]], *, fitted_on: str = ""):
        for col, vocab in columns.items():
            if list(vocab[: len(RESERVED_TOKENS)]) != list(RESERVED_TOKENS):
                raise ValueError(
                    f"vocab for {col!r} must start with {RESERVED_TOKENS}, got {vocab[:1]!r}"
                )
            if len(set(vocab)) != len(vocab):
                raise ValueError(f"vocab for {col!r} contains duplicate tokens")
        self.columns = {c: list(v) for c, v in columns.items()}
        self.fitted_on = fitted_on

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        cols: Iterable[str],
        *,
        fitted_on: str = "",
    ) -> "CategoricalVocab":
        """Fit one vocabulary per column from ``df`` (the TRAINING rows only,
        BEFORE any provenance dropout is applied, so every training category is
        retained in the vocabulary).

        Vocabulary = ``__FALLBACK__`` + sorted distinct nonempty training values.
        Raw training values colliding with the reserved token are a hard error.
        """
        columns: Dict[str, List[str]] = {}
        for col in cols:
            if col not in df.columns:
                columns[col] = list(RESERVED_TOKENS)
                continue
            raw = df[col].dropna().astype(str)
            clash = set(raw) & set(RESERVED_TOKENS)
            if clash:
                raise ValueError(
                    f"training values for {col!r} collide with reserved tokens: {sorted(clash)}"
                )
            observed = _normalize(df[col])
            uniques = sorted(set(observed) - {FALLBACK_TOKEN})
            columns[col] = list(RESERVED_TOKENS) + uniques
        return cls(columns, fitted_on=fitted_on)

    # --------------------------------------------------------------- encode
    def encode_series(self, values: pd.Series, col: str) -> np.ndarray:
        """Encode one column to float codes. Deterministic and row-local:
        absent/empty AND unseen-nonempty -> code 0 (``__FALLBACK__``),
        known -> fixed index."""
        if col not in self.columns:
            raise KeyError(f"column {col!r} not in fitted vocabulary {sorted(self.columns)}")
        vocab = self.columns[col]
        codes = pd.Categorical(_normalize(values), categories=vocab).codes.astype(float)
        codes[codes < 0] = float(FALLBACK_CODE)
        return codes

    def telemetry(self, values: pd.Series, col: str) -> Dict[str, int]:
        """Count known / missing / unknown inputs for this column.

        ``missing`` (absent or empty) and ``unknown`` (nonempty, not in the
        fitted vocabulary) both ENCODE to the fallback code; this method keeps
        the distinction observable for manifests and monitoring.
        """
        if col not in self.columns:
            raise KeyError(f"column {col!r} not in fitted vocabulary {sorted(self.columns)}")
        missing = _is_missing(values)
        known_set = set(self.columns[col]) - set(RESERVED_TOKENS)
        known = ~missing & values.astype(object).astype(str).isin(known_set)
        unknown = ~missing & ~known
        return {
            "n": int(len(values)),
            "known": int(known.sum()),
            "missing": int(missing.sum()),
            "unknown": int(unknown.sum()),
        }

    def cardinality(self, col: str) -> int:
        return len(self.columns[col])

    # -------------------------------------------------------------- persist
    def to_dict(self) -> dict:
        return {
            "format": VOCAB_FORMAT,
            "schema_version": VOCAB_SCHEMA_VERSION,
            "reserved": {"fallback": FALLBACK_TOKEN, "fallback_code": FALLBACK_CODE},
            "fitted_on": self.fitted_on,
            "columns": self.columns,
        }

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CategoricalVocab":
        """Load a persisted vocabulary; fail closed on anything incompatible
        (including schema v1 files, which used two untrained reserved tokens)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"categorical vocabulary not found: {path}")
        obj = json.loads(path.read_text())
        if obj.get("format") != VOCAB_FORMAT:
            raise ValueError(f"{path}: not a {VOCAB_FORMAT} file (format={obj.get('format')!r})")
        if obj.get("schema_version") != VOCAB_SCHEMA_VERSION:
            raise ValueError(
                f"{path}: unsupported schema_version {obj.get('schema_version')!r} "
                f"(expected {VOCAB_SCHEMA_VERSION})"
            )
        return cls(obj["columns"], fitted_on=obj.get("fitted_on", ""))
