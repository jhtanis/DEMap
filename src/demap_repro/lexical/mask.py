"""Shared deterministic query-level exact-match eligibility mask.

ONE mask governs whether a query's gold CDE's exact-matching metadata rows are
eligible, for BOTH lexical methods:

  - the Python approximation to NCI CDE Match (``cde_match_clone.ExactMatchControl``);
  - CDE Match-Fuzzy (the keyword arm's exact-rule gate).

The mask controls row *eligibility* only — it never alters query text and never
mutates the catalog.

Properties (locked):
  - **Deterministic and process-stable**: the decision is a pure function of
    ``(seed, query_id)`` via SHA-1 — no RNG state, no ``PYTHONHASHSEED``
    dependence, identical across processes and hosts.
  - **Stable key**: the query's ``query_id`` (a content hash assigned at dataset
    build time), not its position or text.
  - **Identical for both lexical methods**: both call :func:`query_allowed`.
  - **Nested across allowance levels**: a query is admitted iff
    ``hash01(query_id) < allow_rate``, so the admitted set at a lower rate is a
    subset of the admitted set at any higher rate (50% ⊂ 60% ⊂ 70% ⊂ 80% ⊂ 100%).

The hash formula is byte-identical to the historical
``ExactMatchControl.query_allowed`` (seed 42), so the primary 0.70 operating
point reproduces the same admitted query set as every published artifact.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Iterable, Sequence

MASK_VERSION = "exact_match_mask_v1"
DEFAULT_SEED = 42
STABLE_KEY = "query_id"


def hash01(query_id: object, seed: int = DEFAULT_SEED) -> float:
    """Uniform-[0,1) deterministic hash of (seed, query_id)."""
    h = hashlib.sha1(f"{seed}|{query_id}".encode("utf-8")).hexdigest()[:12]
    return int(h, 16) / float(16 ** 12)


def query_allowed(query_id: object, allow_rate: float, seed: int = DEFAULT_SEED) -> bool:
    """True iff the query's exact gold-metadata rows stay eligible.

    Clamped at the ends: ``allow_rate >= 1.0`` admits every query,
    ``allow_rate <= 0.0`` admits none.
    """
    if allow_rate >= 1.0:
        return True
    if allow_rate <= 0.0:
        return False
    return hash01(query_id, seed) < allow_rate


def allowed_set(query_ids: Iterable[object], allow_rate: float,
                seed: int = DEFAULT_SEED) -> set:
    """The admitted subset of ``query_ids`` at ``allow_rate``."""
    return {q for q in query_ids if query_allowed(q, allow_rate, seed)}


def mask_record(query_ids: Sequence[object], allow_rate: float,
                seed: int = DEFAULT_SEED) -> Dict[str, object]:
    """Machine-readable provenance block for summaries/manifests."""
    qs = [str(q) for q in query_ids]
    uniq = sorted(set(qs))
    n_allowed = sum(1 for q in uniq if query_allowed(q, allow_rate, seed))
    return {
        "mask_version": MASK_VERSION,
        "stable_key": STABLE_KEY,
        "namespace_seed": seed,
        "allow_rate_expected": float(allow_rate),
        "n_queries": len(uniq),
        "n_allowed": n_allowed,
        "allow_rate_realized": (n_allowed / len(uniq)) if uniq else None,
        "nested": True,
    }
