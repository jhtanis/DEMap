"""Adapter for the CDE Match-derived components whose redistribution is unresolved.

Three modules in the research repository are derivative works of Oracle PL/SQL
supplied to us by NCI, or are built directly on top of that derivative:

===============================  ==================================================
research module                  what this pipeline needs from it
===============================  ==================================================
``demap.features.keyword_retriever``  ``generate_candidates`` — the long keyword
                                 provenance rows ``(query_id, cde_id, rule, field,
                                 rule_rank, rule_score)`` used to build the ``kw_*``
                                 evidence features and the CDE Match-Fuzzy arm.
``demap.features.cdematch``      ``compute_cdematch_features`` and
                                 ``CDEMATCH_FEATURE_COLUMNS`` — per-query rank and
                                 score statistics over the candidate union.
``demap.features.cde_match_clone``   the Python approximation to NCI CDE Match,
                                 evaluated as its own method in Table 4, and
                                 ``ExactMatchControl``, its allowance gate.
===============================  ==================================================

Until a written redistribution determination exists, **those implementations are
not part of this repository** and are not in its git history. This module is the
seam. It declares the API the rest of the pipeline depends on and resolves it at
runtime from an implementation the user supplies.

Resolution order
----------------
1. the module named by the ``DEMAP_CDE_MATCH_PACKAGE`` environment variable
   (default ``demap.features``), if it is importable;
2. otherwise :class:`CDEMatchUnavailable` is raised, naming the symbol and the
   open gate.

Every stage that touches these symbols is marked in ``manifests/paper_scope.yaml``
with ``release_gate: cde_match_derivative_unresolved``. Stages downstream of the
candidate pool can instead consume a **precomputed** candidate or feature
artifact and never reach this module at all — that is the intended public path
while the gate is open.

Nothing here reimplements or approximates the gated logic.
"""
from __future__ import annotations

import importlib
import os
from typing import Any

__all__ = [
    "CDEMatchUnavailable",
    "DEFAULT_PACKAGE",
    "package_name",
    "is_available",
    "load_symbol",
    "generate_candidates",
    "compute_cdematch_features",
    "cdematch_feature_columns",
    "exact_match_control",
]

DEFAULT_PACKAGE = "demap.features"

#: Symbols this pipeline requires, mapped to the submodule that provides them.
REQUIRED_SYMBOLS = {
    "generate_candidates": "keyword_retriever",
    "compute_cdematch_features": "cdematch",
    "CDEMATCH_FEATURE_COLUMNS": "cdematch",
    "ExactMatchControl": "cde_match_clone",
}


class CDEMatchUnavailable(ImportError):
    """Raised when a gated CDE Match symbol is needed but not installed."""

    def __init__(self, symbol: str, submodule: str, package: str, cause: str = ""):
        super().__init__(
            f"{symbol!r} (from {package}.{submodule}) is required for this stage but is "
            f"not importable{f': {cause}' if cause else ''}.\n"
            "\n"
            "This symbol belongs to the NCI CDE Match derivative, whose redistribution "
            "status is unresolved, so it is not shipped with demap_repro.\n"
            "\n"
            "Options:\n"
            f"  1. Install an implementation exposing {package}.{submodule}.{symbol} and "
            "point DEMAP_CDE_MATCH_PACKAGE at its parent package.\n"
            "  2. Skip this stage and supply the precomputed candidate/feature artifact "
            "it would have produced; every downstream stage accepts one.\n"
            "  3. Use the BM25 lexical arm, which has no CDE Match derivation and ships "
            "in full (demap_repro.lexical.bm25)."
        )
        self.symbol = symbol
        self.submodule = submodule
        self.package = package


def package_name() -> str:
    """Package that provides the gated CDE Match submodules."""
    return os.environ.get("DEMAP_CDE_MATCH_PACKAGE", DEFAULT_PACKAGE)


def load_symbol(symbol: str) -> Any:
    """Resolve one gated symbol, or raise :class:`CDEMatchUnavailable`."""
    try:
        submodule = REQUIRED_SYMBOLS[symbol]
    except KeyError:
        raise KeyError(
            f"{symbol!r} is not part of the declared CDE Match interface; "
            f"expected one of {sorted(REQUIRED_SYMBOLS)}"
        ) from None
    package = package_name()
    try:
        mod = importlib.import_module(f"{package}.{submodule}")
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim in the message
        raise CDEMatchUnavailable(symbol, submodule, package, str(exc)) from exc
    try:
        return getattr(mod, symbol)
    except AttributeError as exc:
        raise CDEMatchUnavailable(
            symbol, submodule, package, f"module imported but has no attribute {symbol!r}"
        ) from exc


def is_available() -> bool:
    """True iff every declared symbol resolves. Used to skip gated tests."""
    try:
        for symbol in REQUIRED_SYMBOLS:
            load_symbol(symbol)
    except CDEMatchUnavailable:
        return False
    return True


# -- thin pass-throughs, resolved lazily so importing this module never fails ---

def generate_candidates(*args, **kwargs):
    """Long keyword provenance rows. See ``keyword_retriever.generate_candidates``."""
    return load_symbol("generate_candidates")(*args, **kwargs)


def compute_cdematch_features(*args, **kwargs):
    """Per-query CDE Match rank/score features over a candidate union."""
    return load_symbol("compute_cdematch_features")(*args, **kwargs)


def cdematch_feature_columns():
    """Column names produced by :func:`compute_cdematch_features`."""
    return list(load_symbol("CDEMATCH_FEATURE_COLUMNS"))


def exact_match_control(*args, **kwargs):
    """The clone's exact-match allowance gate.

    Note that the deterministic mask itself is ours and ships in
    :mod:`demap_repro.lexical.mask`; only this wrapper around the clone's
    application of it is gated.
    """
    return load_symbol("ExactMatchControl")(*args, **kwargs)
