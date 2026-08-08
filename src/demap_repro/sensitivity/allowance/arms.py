"""Exact-match allowance sensitivity: the two analysis arms and what separates them.

Both lexical methods gate their exact-match evidence through one deterministic
query-level mask (:mod:`demap_repro.lexical.mask`). S6.2 sweeps that gate from 0%
to 100% and asks how much each method depends on it.

At every rate the CDE Match-Fuzzy candidates, the merged candidate pools and every
allowance-dependent feature are **regenerated**. FT-MPNet and FT-MedCPT are held
fixed throughout — they are not lexical and do not see the mask.

The two arms answer different questions
---------------------------------------
``fixed_070``
    The shipped HGBC, trained once at the 70% allowance, scored at each inference
    allowance. This is inference-time mismatch: what happens to a *deployed*
    system when the world stops looking like its training distribution. It is the
    curve in Figure S6 and Table S6, and the robustness claim in S6.2 rests on it.

``retrained_per_rate``
    A fresh HGBC trained under the identical frozen protocol at each allowance.
    This is adaptation: what the method could achieve if retrained for the new
    regime. It appears only in Figure S7, paired against ``fixed_070``.

Conflating them would overstate robustness — the retrained arm cannot suffer
train/inference mismatch by construction, so reading its curve as evidence about
the deployed system is a category error. Hence the arm is an explicit, required
parameter everywhere in this package.

Dataset tiers
-------------
Not every row in the aggregate is robustness evidence, and the artifact says so:

``primary``
    The four caDSR-derived evaluation sets. The mask applies to them, so sweeping
    it is meaningful. This is what S6.2, Figure S6, Figure S7 and Table S6 report.
``diagnostic_in_sample_not_robustness_evidence``
    Validation Training and Validation Dev. Validation Training is what the HGBC
    was fitted on and Validation Dev is what selected its hyperparameters, so
    their curves are in-sample and prove nothing about robustness. Retained for
    diagnosis, never reported as evidence.
``secondary_external_fixed_allowance_1.0``
    GDC and CIMAC. They run at allowance 1.0 by construction and are never masked,
    so their "curve" is a flat line. Retained as a control: movement there would
    mean the mask had leaked onto sets it must never touch.

Migration
---------
The research repository ran this as two script families —
``.scratch/v17_claude/J_allowance/`` (12 files, the retrained arm) and
``manuscript/v17_claude_reports/K_fixed070/`` (8 files, the fixed arm) — chained
through Slurm. Both are unified here; ``K2_aggregate.py`` built the single CSV
that backs all three manuscript items, and that role is now
:func:`demap_repro.sensitivity.allowance.aggregate.build_four_method_table`.
"""
from __future__ import annotations

from typing import Dict, Tuple

__all__ = [
    "ARMS", "ARM_FIXED", "ARM_RETRAINED",
    "ALLOWANCE_RATES", "OPERATING_RATE",
    "PRIMARY_DATASETS", "DIAGNOSTIC_DATASETS", "EXTERNAL_DATASETS",
    "DATASET_TIER", "TIER_PRIMARY", "TIER_DIAGNOSTIC", "TIER_EXTERNAL",
    "METHODS", "TABLE_S6_METHODS", "FIGURE_S6_METHODS", "FIGURE_S7_METHODS",
    "HELD_FIXED", "tier_for_dataset", "describe_arm",
]

ARM_FIXED = "hgbc_fixed_070"
ARM_RETRAINED = "hgbc_retrained_per_rate"

ARMS: Dict[str, str] = {
    ARM_FIXED: (
        "The shipped HGBC, trained once at the 70% allowance and scored at each "
        "inference allowance. Measures inference-time mismatch in a deployed "
        "pipeline."),
    ARM_RETRAINED: (
        "A fresh HGBC trained under the identical frozen protocol at each "
        "allowance. Measures adaptation of the training procedure, not deployment "
        "robustness."),
}

#: The swept rates. 0.70 is the operating point used everywhere else in the paper.
ALLOWANCE_RATES: Tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.8, 1.0)
OPERATING_RATE = 0.70

TIER_PRIMARY = "primary"
TIER_DIAGNOSTIC = "diagnostic_in_sample_not_robustness_evidence"
TIER_EXTERNAL = "secondary_external_fixed_allowance_1.0"

PRIMARY_DATASETS: Tuple[str, ...] = ("test", "cctg", "oid_alt", "cdash")
DIAGNOSTIC_DATASETS: Tuple[str, ...] = ("val_train", "val_dev")
EXTERNAL_DATASETS: Tuple[str, ...] = ("gdc_combined", "cimac_v2")

DATASET_TIER: Dict[str, str] = {
    **{d: TIER_PRIMARY for d in PRIMARY_DATASETS},
    **{d: TIER_DIAGNOSTIC for d in DIAGNOSTIC_DATASETS},
    **{d: TIER_EXTERNAL for d in EXTERNAL_DATASETS},
}

#: Every method in the aggregate CSV.
METHODS: Tuple[str, ...] = (
    "python_cde_match_approx", "cde_match_fuzzy", ARM_FIXED, ARM_RETRAINED,
)

#: Table S6 and Figure S6 report three methods; the retrained arm is Figure S7 only.
TABLE_S6_METHODS: Tuple[str, ...] = (
    "python_cde_match_approx", "cde_match_fuzzy", ARM_FIXED,
)
FIGURE_S6_METHODS = TABLE_S6_METHODS
FIGURE_S7_METHODS: Tuple[str, ...] = (ARM_FIXED, ARM_RETRAINED)

#: Components held constant across the whole sweep. Neither is lexical, so neither
#: sees the mask; letting either vary would confound the comparison.
HELD_FIXED: Tuple[str, ...] = ("ft_mpnet_biencoder", "ft_medcpt_crossencoder")


def tier_for_dataset(dataset: str) -> str:
    """Evidence tier of ``dataset`` in the allowance analysis."""
    try:
        return DATASET_TIER[dataset]
    except KeyError:  # pragma: no cover - guarded by tests
        raise KeyError(
            f"{dataset!r} has no allowance-analysis tier; classify it explicitly "
            "rather than treating it as primary evidence"
        ) from None


def describe_arm(arm: str) -> str:
    """Human-readable description of an analysis arm."""
    try:
        return ARMS[arm]
    except KeyError:  # pragma: no cover - guarded by tests
        raise KeyError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}") from None
