"""Per-split routing for the final fixed-K feature build.

This module makes explicit a decision that, in the research repository, was
spread across three files and one Slurm driver: **which evaluation splits are
treated as caDSR-derived**, and therefore

1. route to the ``allow0.70`` CDE Match-Fuzzy candidate tables rather than the
   ``allow1.0`` external tables, and
2. receive the deterministic gold-scoped exact-match control on their keyword
   provenance (see :mod:`demap_repro.lexical.mask`).

Why the distinction exists
--------------------------
The four caDSR-derived evaluation sets (Test, CCTG, OID ALT, CDASH) are built
from the same metadata the CDE catalog exposes, so an exact string match between
a query and its gold CDE's metadata is partly an artifact of construction rather
than evidence a real user would have. The paper therefore withholds that
evidence for a deterministic 30% of those queries (the 70% allowance).

GDC and CIMAC are externally curated: their queries were written by other
organizations against their own conventions, so an exact match there is
legitimate signal and is never withheld. They run at allowance 1.0.

``cctg`` and ``oid_alt`` derive from ``external_holdout_org``, and ``cdash``
from ``external_holdout_refslice``. Despite the "external" in those legacy split
names they ARE caDSR-derived, and their fuzzy candidate tables were generated at
allow0.70. Routing them to the a1.0 branch would silently reintroduce exact
gold evidence the paper withheld and inflate every lexical and reranker number
on three of the six evaluation datasets.

Provenance
----------
The membership set below is copied verbatim from
``scripts/build_hgbc_feature_table.py:_CADSR_DERIVED_SPLITS`` in the research
repository, where it has been permanent since 2026-07-10. The hydration policy
is copied from ``src/demap/features/candidate_union.py:DEFAULT_HYDRATION_POLICY``.
``.scratch/demap/paper_v13_scientific_audit/run_fixed_k_stepG_v2.py`` — the
executed driver of the final build — asserts exactly these two facts before
running; those assertions are preserved as
``tests/tier1_invariants/test_split_routing.py``.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

__all__ = [
    "CADSR_DERIVED_SPLITS",
    "CANONICAL_EVAL_SPLITS",
    "PAPER_FEATURE_TABLE_SPLITS",
    "DEFAULT_HYDRATION_POLICY",
    "CADSR_ALLOWANCE",
    "EXTERNAL_ALLOWANCE",
    "is_cadsr_derived",
    "allowance_for_split",
    "fuzzy_table_filename",
    "hydration_for_split",
]


#: Splits whose exact gold-metadata evidence is controlled by the allowance mask.
CADSR_DERIVED_SPLITS = frozenset({
    # Internal benchmark splits.
    "test", "val_dev", "val_train", "train", "train_noleak", "train_natural",
    "train_natural_noleak", "natural_internal_eval",
    # Legacy source splits the canonical evaluation sets were carved from.
    "external_holdout_org", "external_holdout_standard", "external_holdout_refslice",
    # Canonical final-reranker externals. Their CDE Match-Fuzzy tables are generated
    # at allow0.70, so they must be exact-controlled like caDSR splits AND read the
    # a0.70 tables. cctg/oid_alt come from external_holdout_org, cdash from
    # external_holdout_refslice. gdc_combined/cimac_v2 deliberately stay out.
    "cctg", "oid_alt", "cdash",
})

#: The six canonical evaluation datasets reported in the paper.
CANONICAL_EVAL_SPLITS = ("test", "cctg", "oid_alt", "cdash", "gdc_combined", "cimac_v2")

#: Every split present in the final paper feature table, in report order.
PAPER_FEATURE_TABLE_SPLITS = ("val_train", "val_dev") + CANONICAL_EVAL_SPLITS

#: Exact-match allowance applied to caDSR-derived splits (the paper operating point).
CADSR_ALLOWANCE = 0.70

#: Exact-match allowance applied to externally curated splits: no withholding.
EXTERNAL_ALLOWANCE = 1.0

#: CDE-side permissible-value hydration per split. caDSR-derived splits stay
#: "thinned" (PV_BLOCK_CDE only, preserving leakage protection); externally
#: curated splits are "hydrated" (fall back to the catalog PV summary when the
#: block is empty).
DEFAULT_HYDRATION_POLICY: Mapping[str, str] = MappingProxyType({
    "train": "thinned",
    "val": "thinned",
    "val_train": "thinned",
    "val_dev": "thinned",
    "test": "thinned",
    "natural_internal_eval": "thinned",
    "external_holdout_org": "thinned",
    "external_holdout_refslice": "thinned",
    "external_holdout_gdc_altnames": "hydrated",
    "external_holdout_gdc_questiontext": "hydrated",
    "cimac_appendix_a_eval": "hydrated",
    "cimac_v2": "hydrated",
    "theradex6_test": "hydrated",
    # Canonical final-reranker evaluation datasets.
    "cctg": "thinned",
    "oid_alt": "thinned",
    "cdash": "thinned",
    "gdc_combined": "hydrated",
})


def is_cadsr_derived(split: str) -> bool:
    """True iff ``split`` is exact-controlled and reads the a0.70 fuzzy tables."""
    return split in CADSR_DERIVED_SPLITS


def allowance_for_split(split: str) -> float:
    """Exact-match allowance rate that governs ``split`` in the paper build."""
    return CADSR_ALLOWANCE if is_cadsr_derived(split) else EXTERNAL_ALLOWANCE


def fuzzy_table_filename(split: str) -> str:
    """File name of the CDE Match-Fuzzy candidate table for ``split``.

    caDSR-derived splits read the ``a070`` tables; external splits read ``a10``.
    The suffix is part of the file name in the paper artifact layout, so routing
    errors surface as a missing-input failure rather than as silently wrong
    numbers.
    """
    if is_cadsr_derived(split):
        return f"cde_match_clone_candidates_{split}_fuzzy_a070.parquet"
    return f"cde_match_clone_candidates_{split}_fuzzy_a10.parquet"


def hydration_for_split(split: str) -> str:
    """CDE-side PV hydration policy (``"thinned"`` or ``"hydrated"``)."""
    try:
        return DEFAULT_HYDRATION_POLICY[split]
    except KeyError:  # pragma: no cover - guarded by tests
        raise KeyError(
            f"no hydration policy registered for split {split!r}; add it to "
            "DEFAULT_HYDRATION_POLICY rather than defaulting silently"
        ) from None
