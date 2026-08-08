"""Regression tests for the fixed-K (K = 30) candidate feature table.

This is the table the final HGBC was trained and evaluated on. The structural
checks come from ``validate_stepG_v2.py``, the validator that gated the final
build in the research repository. The routing checks are new and are the ones
that matter most: they read the published table and confirm that the 0.70
exact-match allowance actually fired on the caDSR-derived splits and actually did
not fire on GDC and CIMAC.

The table itself is 54 MB and is not distributed, so these run against the
committed summary in ``tests/fixtures/fixed_k_feature_table.json``. Set
``DEMAP_ARTIFACT_ROOT`` to also verify the summary against the real parquet.
"""
from __future__ import annotations

import pytest

from demap_repro.reranker.split_routing import (
    PAPER_FEATURE_TABLE_SPLITS,
    is_cadsr_derived,
)

pytestmark = [pytest.mark.tier3, pytest.mark.parity]

CE_FEATURES = ["crossenc_score", "crossenc_rank", "crossenc_margin_to_top1",
               "crossenc_margin_to_next", "crossenc_score_z_by_query", "crossenc_missing"]

# Query counts reported in the manuscript (query-level, reachable gold).
PAPER_QUERY_COUNTS = {
    "val_train": 3946, "val_dev": 3934, "test": 3959, "cctg": 1097,
    "oid_alt": 1766, "cdash": 324, "gdc_combined": 72, "cimac_v2": 131,
}


# --------------------------------------------------------------------------
# structure (ported from validate_stepG_v2.py)
# --------------------------------------------------------------------------

def test_exactly_the_eight_paper_splits_are_present(fixed_k_summary):
    assert set(fixed_k_summary["per_split"]) == set(PAPER_FEATURE_TABLE_SPLITS)


def test_all_six_crossencoder_features_present(fixed_k_summary):
    assert fixed_k_summary["ce_feature_columns_present"] == CE_FEATURES


def test_crossencoder_coverage_is_complete(fixed_k_summary):
    """Every candidate row carries a cross-encoder score. A coverage gap would
    make crossenc_* missingness correlate with split, which the HGBC would learn."""
    for split, s in fixed_k_summary["per_split"].items():
        assert s["n_rows_with_ce"] == s["n_rows"], split


def test_no_duplicate_candidate_keys(fixed_k_summary):
    assert fixed_k_summary["n_duplicate_candidate_keys"] == 0


def test_training_and_tuning_splits_have_positives(fixed_k_summary):
    for split in ("val_train", "val_dev"):
        assert fixed_k_summary["per_split"][split]["n_positive_rows"] > 0


def test_query_counts_match_the_manuscript(fixed_k_summary):
    for split, expected in PAPER_QUERY_COUNTS.items():
        assert fixed_k_summary["per_split"][split]["n_queries"] == expected, split


def test_hgbc_training_set_matches_s55(fixed_k_summary):
    """S5.5: trained on Validation Training — 3,946 queries, 109,141
    query-candidate pairs, 4,109 positive pairs (about 3.7%)."""
    s = fixed_k_summary["per_split"]["val_train"]
    assert s["n_queries"] == 3946
    assert s["n_rows"] == 109141
    assert s["n_positive_rows"] == 4109
    assert round(100 * s["n_positive_rows"] / s["n_rows"], 1) == 3.8


def test_realized_pool_stays_under_nominal_k(fixed_k_summary):
    """S5.3/S5.5: nominal K = 30, realized mean about 27.6 because the two arms
    overlap. A mean at or above 30 would mean the dedup-by-public-id step broke."""
    for split, s in fixed_k_summary["per_split"].items():
        assert 26.0 <= s["mean_candidates_per_query"] < 30.0, split


# --------------------------------------------------------------------------
# routing — the checks that catch a silent allowance regression
# --------------------------------------------------------------------------

def test_routing_label_matches_split_routing_module(fixed_k_summary):
    for split, s in fixed_k_summary["per_split"].items():
        expected = "cadsr_a070" if is_cadsr_derived(split) else "external_a10"
        assert s["routing"] == expected, split


@pytest.mark.parametrize(
    "split", [s for s in PAPER_FEATURE_TABLE_SPLITS if is_cadsr_derived(s)]
)
def test_blocked_queries_never_keep_an_exact_gold_candidate(fixed_k_summary, split):
    """The 0.70 allowance withholds the gold CDE's exact-match evidence for a
    deterministic 30% of caDSR-derived queries. In the published table not one of
    those queries carries an exact gold candidate.

    If CCTG, OID ALT or CDASH were ever rerouted to the external a1.0 tables this
    count would jump to roughly the allowed-query rate, and their reported recall
    would rise with it.
    """
    s = fixed_k_summary["per_split"][split]
    assert s["n_queries_gold_blocked_at_070"] > 0, "no blocked queries to test"
    assert s["n_blocked_queries_with_exact_gold"] == 0


@pytest.mark.parametrize("split", ["gdc_combined", "cimac_v2"])
def test_externally_curated_sets_keep_their_exact_evidence(fixed_k_summary, split):
    """The mirror image: GDC and CIMAC run at allowance 1.0, so queries that the
    0.70 mask *would* have blocked still carry exact gold candidates. Seeing zero
    here would mean the mask leaked onto sets it must never touch."""
    s = fixed_k_summary["per_split"][split]
    assert s["n_queries_gold_blocked_at_070"] > 0
    assert s["n_blocked_queries_with_exact_gold"] > 0


def test_allowed_queries_retain_exact_evidence_on_cadsr_splits(fixed_k_summary):
    """Sanity in the other direction: the mask must suppress only the blocked
    fraction, not all exact evidence. Allowed queries keep theirs at a high rate."""
    for split in PAPER_FEATURE_TABLE_SPLITS:
        if not is_cadsr_derived(split):
            continue
        s = fixed_k_summary["per_split"][split]
        rate = s["n_allowed_queries_with_exact_gold"] / s["n_queries_gold_allowed_at_070"]
        assert rate > 0.9, f"{split}: only {rate:.1%} of allowed queries kept exact gold"


def test_mask_parameters_are_the_paper_operating_point(fixed_k_summary):
    assert fixed_k_summary["allowance_mask"] == {
        "version": "exact_match_mask_v1", "rate": 0.70, "seed": 42,
    }


# --------------------------------------------------------------------------
# optional: verify the committed summary against the real artifact
# --------------------------------------------------------------------------

@pytest.mark.needs_artifacts
@pytest.mark.slow
def test_summary_matches_the_real_feature_table(fixed_k_summary, artifact_root):
    import pandas as pd

    table = artifact_root / fixed_k_summary["source_artifact"]
    if not table.exists():
        pytest.skip(f"{table} not present")
    df = pd.read_parquet(table, columns=["split", "query_id", "cde_id", "is_label",
                                         "is_exact_candidate", "crossenc_score"])
    assert len(df) == fixed_k_summary["n_rows"]
    for split, s in fixed_k_summary["per_split"].items():
        g = df[df["split"] == split]
        assert len(g) == s["n_rows"], split
        assert g["query_id"].nunique() == s["n_queries"], split
        assert int(g["is_label"].sum()) == s["n_positive_rows"], split
        assert int(g["crossenc_score"].notna().sum()) == s["n_rows_with_ce"], split
