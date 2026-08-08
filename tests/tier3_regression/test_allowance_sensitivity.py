"""Exact-match allowance sensitivity: Table S6, Figure S6, Figure S7, S6.2.

One 120-row aggregate backs all four. It is small and public, so it is committed
and every claim replays offline.

The distinction these tests exist to protect is between the two HGBC arms. The
fixed arm is trained once at 70% and scored everywhere — that is the robustness
evidence. The retrained arm is refitted at each rate and cannot suffer
train/inference mismatch by construction, so reading it as robustness evidence
would overstate the result. Figure S7 shows them together precisely to keep them
apart.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from demap_repro.sensitivity.allowance.arms import (
    ALLOWANCE_RATES,
    ARM_FIXED,
    ARM_RETRAINED,
    ARMS,
    DIAGNOSTIC_DATASETS,
    EXTERNAL_DATASETS,
    FIGURE_S7_METHODS,
    HELD_FIXED,
    OPERATING_RATE,
    PRIMARY_DATASETS,
    TABLE_S6_METHODS,
    TIER_DIAGNOSTIC,
    TIER_EXTERNAL,
    TIER_PRIMARY,
    describe_arm,
    tier_for_dataset,
)
from demap_repro.sensitivity.allowance.report import (
    figure_s6_series,
    figure_s7_series,
    load_sensitivity,
    robustness_summary,
    table_s6,
)

pytestmark = [pytest.mark.tier3, pytest.mark.parity]

CSV = (Path(__file__).resolve().parents[1] / "fixtures"
       / "allowance_sensitivity_four_methods.csv")


@pytest.fixture(scope="module")
def sensitivity():
    return load_sensitivity(CSV)


@pytest.fixture(scope="module")
def summary(sensitivity):
    return robustness_summary(sensitivity)


# --------------------------------------------------------------------------
# protocol
# --------------------------------------------------------------------------

def test_six_allowance_rates_including_the_operating_point():
    assert ALLOWANCE_RATES == (0.0, 0.5, 0.6, 0.7, 0.8, 1.0)
    assert OPERATING_RATE in ALLOWANCE_RATES


def test_both_arms_are_named_and_described():
    assert set(ARMS) == {ARM_FIXED, ARM_RETRAINED}
    assert "deployed" in describe_arm(ARM_FIXED)
    assert "Adaptation" in describe_arm(ARM_RETRAINED) or \
           "adaptation" in describe_arm(ARM_RETRAINED)


def test_unknown_arm_raises():
    with pytest.raises(KeyError, match="unknown arm"):
        describe_arm("hgbc_something_else")


def test_biencoder_and_crossencoder_are_held_fixed():
    """S6.2/Figure S6: 'FT-MPNet, FT-MedCPT and HGBC weights remained fixed.'"""
    assert HELD_FIXED == ("ft_mpnet_biencoder", "ft_medcpt_crossencoder")


def test_dataset_tiers_are_explicit():
    for d in PRIMARY_DATASETS:
        assert tier_for_dataset(d) == TIER_PRIMARY
    for d in DIAGNOSTIC_DATASETS:
        assert tier_for_dataset(d) == TIER_DIAGNOSTIC
    for d in EXTERNAL_DATASETS:
        assert tier_for_dataset(d) == TIER_EXTERNAL
    with pytest.raises(KeyError):
        tier_for_dataset("some_new_set")


# --------------------------------------------------------------------------
# the aggregate
# --------------------------------------------------------------------------

def test_aggregate_has_120_rows(sensitivity):
    assert len(sensitivity) == 120


def test_primary_tier_is_four_methods_by_four_datasets_by_six_rates(sensitivity):
    primary = sensitivity[sensitivity["tier"] == TIER_PRIMARY]
    assert len(primary) == 96
    assert set(primary["dataset"]) == set(PRIMARY_DATASETS)
    assert primary["method"].nunique() == 4
    assert sorted(primary["allow_rate"].unique()) == list(ALLOWANCE_RATES)


def test_validation_rows_are_marked_as_not_robustness_evidence(sensitivity):
    """Validation Training fitted the model and Validation Dev selected its
    hyperparameters, so their curves are in-sample."""
    diag = sensitivity[sensitivity["tier"] == TIER_DIAGNOSTIC]
    assert set(diag["dataset"]) == set(DIAGNOSTIC_DATASETS)
    assert set(diag["method"]) == {ARM_FIXED}
    assert "not_robustness_evidence" in TIER_DIAGNOSTIC


def test_external_sets_are_carried_as_a_control(sensitivity):
    """GDC and CIMAC run at allowance 1.0 and are never masked, so their curves
    must be flat. Movement here would mean the mask leaked onto them."""
    ext = sensitivity[sensitivity["tier"] == TIER_EXTERNAL]
    assert set(ext["dataset"]) == set(EXTERNAL_DATASETS)
    for dataset, g in ext.groupby("dataset"):
        assert g["recall@5"].nunique() == 1, f"{dataset} moved with the allowance rate"


# --------------------------------------------------------------------------
# Table S6
# --------------------------------------------------------------------------

def test_table_s6_is_72_rows_of_three_methods(sensitivity):
    """'72 reported rows = 3 methods x 4 datasets x 6 rates.'"""
    t = table_s6(sensitivity)
    assert len(t) == 72
    assert set(t["method"]) == set(TABLE_S6_METHODS)
    assert ARM_RETRAINED not in set(t["method"]), "the retrained arm is Figure S7 only"


def test_table_s6_carries_all_four_metrics(sensitivity):
    t = table_s6(sensitivity)
    for col in ("recall@1", "recall@5", "recall@10", "mrr@100"):
        assert col in t.columns
        assert t[col].notna().all()


# --------------------------------------------------------------------------
# Figures S6 and S7
# --------------------------------------------------------------------------

def test_figure_s6_plots_three_curves_on_four_panels(sensitivity):
    series = figure_s6_series(sensitivity)
    assert set(series) == set(PRIMARY_DATASETS)
    for dataset, curves in series.items():
        assert set(curves) == set(TABLE_S6_METHODS), dataset
        for method, points in curves.items():
            assert [r for r, _ in points] == list(ALLOWANCE_RATES)


def test_figure_s7_contrasts_fixed_against_retrained(sensitivity):
    series = figure_s7_series(sensitivity)
    assert set(series) == set(PRIMARY_DATASETS)
    for dataset, curves in series.items():
        assert set(curves) == set(FIGURE_S7_METHODS) == {ARM_FIXED, ARM_RETRAINED}


def test_at_the_operating_rate_both_arms_agree(sensitivity):
    """At 70% the retrained model IS the shipped model — same protocol, same data.
    A divergence here would mean the two arms are not the same pipeline."""
    series = figure_s7_series(sensitivity)
    for dataset, curves in series.items():
        fixed = dict(curves[ARM_FIXED])[OPERATING_RATE]
        retrained = dict(curves[ARM_RETRAINED])[OPERATING_RATE]
        assert abs(fixed - retrained) < 0.005, dataset


# --------------------------------------------------------------------------
# S6.2 magnitudes
# --------------------------------------------------------------------------

def _row(summary, method, dataset):
    m = summary[(summary["method"] == method) & (summary["dataset"] == dataset)]
    assert len(m) == 1
    return m.iloc[0]


def test_fixed_hgbc_variation_on_test_and_cdash(summary):
    """S6.2: 'varied by 0.075 Recall@5 on Test and 0.164 on CDASH'."""
    assert round(_row(summary, ARM_FIXED, "test")["range"], 3) == 0.075
    assert round(_row(summary, ARM_FIXED, "cdash")["range"], 3) == 0.164


def test_lexical_methods_vary_far_more(summary):
    """S6.2: 'the two lexical methods varied by 0.40-0.96 on every dataset'.

    The manuscript quotes the bounds to two decimals, so compare at that precision.
    """
    lexical = summary[summary["method"].isin(
        ["python_cde_match_approx", "cde_match_fuzzy"])]
    assert len(lexical) == 8, "2 lexical methods x 4 primary datasets"
    assert round(lexical["range"].min(), 2) == 0.40
    assert round(lexical["range"].max(), 2) == 0.96


def test_fixed_reranker_at_zero_allowance(summary):
    """S6.2: 'At 0% allowance the fixed reranker still reached 0.920 on Test and
    0.818 on CDASH'."""
    assert round(_row(summary, ARM_FIXED, "test")["at_0.0"], 3) == 0.920
    assert round(_row(summary, ARM_FIXED, "cdash")["at_0.0"], 3) == 0.818


def test_lexical_methods_at_zero_allowance(summary):
    """S6.2: 'compared with 0.151-0.556 for the lexical methods' (Test and CDASH)."""
    values = [
        _row(summary, m, d)["at_0.0"]
        for m in ("python_cde_match_approx", "cde_match_fuzzy")
        for d in ("test", "cdash")
    ]
    assert round(min(values), 3) == 0.151
    assert round(max(values), 3) == 0.556


def test_fixed_reranker_is_more_robust_than_both_lexical_methods(summary):
    """The actual claim of S6.2, stated as a comparison rather than as numbers."""
    for dataset in PRIMARY_DATASETS:
        fixed_range = _row(summary, ARM_FIXED, dataset)["range"]
        for method in ("python_cde_match_approx", "cde_match_fuzzy"):
            assert fixed_range < _row(summary, method, dataset)["range"], dataset


def test_operating_point_reproduces_the_headline_test_recall(summary):
    """The 70% column of this sweep must agree with the headline Recall@5 of
    0.971 on Test — it is the same pipeline, rebuilt."""
    assert round(_row(summary, ARM_FIXED, "test")["at_operating_rate"], 3) == 0.971
