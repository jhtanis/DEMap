"""Table S3 — permissible-value overlap, and the denominator correction.

Two claims are checked. The grouping logic reproduces the corrected values from a
per-pair diagnostics file, and the corrected values are the ones v21 prints.

Why the correction happened is worth keeping visible. The originally published
row counts came from a `pairs.parquet` build of 69,844 rows that was later
overwritten by the 69,102-row paper-era build — the one every model stage
consumed, and the one §3.1 and S1.3 quote. Table S3's denominators were therefore
inconsistent with the rest of the manuscript. The PV generation code never
drifted; only the input to the frozen diagnostic had been replaced.

The superseded artifact is kept as a fixture so the discrepancy stays explicable,
and a test asserts the two are actually different — if they ever coincided, one of
the two fixtures would be mislabelled.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from demap_repro.reporting.dataset_tables_pv_overlap import (
    MIN_GROUP_ROWS,
    REPORTED_GROUPS,
    build_pv_overlap_table,
    parse_items,
    set_jaccard,
)

pytestmark = [pytest.mark.tier3, pytest.mark.parity]

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

# v21 Table S3 after the 2026-08-08 correction.
PAPER_TABLE_S3 = {
    "overall":             {"rows": 38964, "block_pct": 0.4, "shorter_pct": 30.1, "jaccard": 0.168},
    "ENUM":                {"rows": 24158, "block_pct": 0.4, "shorter_pct": 45.3, "jaccard": 0.153},
    "BINARY_WITH_UNKNOWN": {"rows": 3040,  "block_pct": 0.1, "shorter_pct": 0.0,  "jaccard": 0.333},
    "BINARY_WITH_NA":      {"rows": 8815,  "block_pct": 0.0, "shorter_pct": 0.0,  "jaccard": 0.123},
}

# The values printed before the correction, from the 69,844-row input.
SUPERSEDED_ROWS = {"overall": 39391, "ENUM": 24543,
                   "BINARY_WITH_UNKNOWN": 3074, "BINARY_WITH_NA": 8816}


@pytest.fixture(scope="module")
def corrected():
    return pd.read_csv(FIXTURES / "table_s3_corrected.csv").set_index("group")


@pytest.fixture(scope="module")
def superseded():
    return pd.read_csv(FIXTURES / "table_s3_pv_overlap_SUPERSEDED.csv").set_index("group")


# --------------------------------------------------------------------------
# the corrected table matches the manuscript
# --------------------------------------------------------------------------

@pytest.mark.parametrize("group", list(PAPER_TABLE_S3))
def test_corrected_values_match_the_manuscript(corrected, group):
    want = PAPER_TABLE_S3[group]
    row = corrected.loc[group]
    assert int(row["n_rows_both_present"]) == want["rows"]
    assert round(100 * row["normalized_block_match_rate"], 1) == want["block_pct"]
    assert round(100 * row["query_shorter_than_cde_rate"], 1) == want["shorter_pct"]
    assert round(row["mean_jaccard_overlap"], 3) == want["jaccard"]


def test_prose_denominator_is_the_overall_row(corrected):
    """S3.5 reads 'over the 38,964 rows in which both sides carry a
    permissible-value summary'. It must be the same number as the table."""
    assert int(corrected.loc["overall", "n_rows_both_present"]) == 38964


def test_table_shows_pv_evidence_is_partial_not_duplicated(corrected):
    """The point of the table: whole-block agreement is rare and item overlap is
    low, so query-side PV is not a copy of the gold CDE's values."""
    assert corrected.loc["overall", "normalized_block_match_rate"] < 0.01
    assert corrected.loc["overall", "mean_jaccard_overlap"] < 0.25


# --------------------------------------------------------------------------
# the correction is real and documented
# --------------------------------------------------------------------------

def test_superseded_row_counts_differ_from_the_corrected_ones(corrected, superseded):
    for group, stale in SUPERSEDED_ROWS.items():
        assert int(superseded.loc[group, "n_rows_both_present"]) == stale
        assert int(corrected.loc[group, "n_rows_both_present"]) != stale


def test_only_one_rate_cell_moved(corrected, superseded):
    """11 of 12 rate cells round identically; only ENUM query-shorter changed,
    45.2% -> 45.3%. That is why the correction is a denominator issue rather than
    a change in what the table says."""
    moved = []
    for group in PAPER_TABLE_S3:
        for col, places in (("normalized_block_match_rate", 3),
                            ("query_shorter_than_cde_rate", 3),
                            ("mean_jaccard_overlap", 3)):
            a = round(float(superseded.loc[group, col]), places)
            b = round(float(corrected.loc[group, col]), places)
            if a != b:
                moved.append((group, col, a, b))
    assert len(moved) == 1, moved
    group, col, _a, _b = moved[0]
    assert group == "ENUM" and col == "query_shorter_than_cde_rate"


# --------------------------------------------------------------------------
# the grouping logic
# --------------------------------------------------------------------------

def test_reported_groups_are_the_four_v21_prints():
    assert REPORTED_GROUPS == ("overall", "ENUM", "BINARY_WITH_UNKNOWN", "BINARY_WITH_NA")


def test_computation_carries_six_groups_but_the_paper_prints_four(corrected):
    """BINARY_WITH_UNKNOWN_NA and BINARY clear the 100-row threshold but are
    omitted from the table."""
    assert len(corrected) == 6
    assert set(corrected.index) - set(REPORTED_GROUPS) == {"BINARY_WITH_UNKNOWN_NA", "BINARY"}


def test_small_pv_types_are_excluded_by_the_row_threshold(tmp_path):
    records = pd.DataFrame({
        "query_source": ["REF"] * 12,
        "both_present": [True] * 12,
        "cde_pv_type": ["ENUM"] * 10 + ["RARE"] * 2,
        "normalized_block_match": [False] * 12,
        "normalized_item_match": [False] * 12,
        "query_shorter_than_cde": [True] * 12,
        "query_pv_items_cf": ["['a']"] * 12,
        "cde_pv_items_cf": ["['a', 'b']"] * 12,
    })
    csv = tmp_path / "records.csv"
    records.to_csv(csv, index=False)
    table = build_pv_overlap_table(csv, min_group_rows=5)
    assert set(table["group"]) == {"overall", "ENUM"}, "RARE has 2 rows, below the threshold"


def test_rows_without_both_sides_are_excluded(tmp_path):
    """A pair with PV on only one side cannot be an overlap observation."""
    records = pd.DataFrame({
        "query_source": ["REF"] * 4,
        "both_present": [True, True, False, False],
        "cde_pv_type": ["ENUM"] * 4,
        "normalized_block_match": [True, False, True, True],
        "normalized_item_match": [True, False, True, True],
        "query_shorter_than_cde": [True, False, True, True],
        "query_pv_items_cf": ["['a']"] * 4,
        "cde_pv_items_cf": ["['a']"] * 4,
    })
    csv = tmp_path / "records.csv"
    records.to_csv(csv, index=False)
    table = build_pv_overlap_table(csv, min_group_rows=1).set_index("group")
    assert int(table.loc["overall", "n_rows_both_present"]) == 2
    assert table.loc["overall", "normalized_block_match_rate"] == 0.5


def test_string_booleans_survive_the_csv_round_trip(tmp_path):
    """The diagnostics are CSV, so booleans arrive as strings. A naive
    astype(bool) would make 'False' truthy and report every row as a match."""
    records = pd.DataFrame({
        "query_source": ["REF"] * 2,
        "both_present": ["True", "True"],
        "cde_pv_type": ["ENUM"] * 2,
        "normalized_block_match": ["False", "False"],
        "normalized_item_match": ["False", "False"],
        "query_shorter_than_cde": ["False", "True"],
        "query_pv_items_cf": ["['a']"] * 2,
        "cde_pv_items_cf": ["['a', 'b']"] * 2,
    })
    csv = tmp_path / "records.csv"
    records.to_csv(csv, index=False)
    table = build_pv_overlap_table(csv, min_group_rows=1).set_index("group")
    assert table.loc["overall", "normalized_block_match_rate"] == 0.0
    assert table.loc["overall", "query_shorter_than_cde_rate"] == 0.5


@pytest.mark.parametrize("raw,expected", [
    ("['yes', 'no']", ["yes", "no"]),
    ("[]", []),
    ("", []),
    ("nan", []),
    (None, []),
])
def test_item_parsing(raw, expected):
    assert parse_items(raw) == expected


def test_jaccard_overlap():
    assert set_jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert set_jaccard(["a"], ["a", "b"]) == 0.5
    assert set_jaccard(["a"], ["b"]) == 0.0


def test_default_row_threshold_is_100():
    assert MIN_GROUP_ROWS == 100


# --------------------------------------------------------------------------
# full chain
# --------------------------------------------------------------------------

@pytest.mark.needs_artifacts
@pytest.mark.slow
def test_full_chain_reproduces_the_corrected_table(artifact_root, tmp_path, corrected):
    """pairs.parquet -> pv_frozen_diagnostics -> this grouping -> Table S3.

    This is the claim the correction rests on: the canonical 69,102-pair benchmark
    yields the corrected numbers when run through the current code.
    """
    from demap_repro.data import pv_frozen_diagnostics

    pairs = artifact_root / "data/processed/pairs.parquet"
    catalog = artifact_root / "data/processed/cde_master_enriched.parquet"
    if not (pairs.exists() and catalog.exists()):
        pytest.skip("frozen benchmark inputs not present")

    out = tmp_path / "diag"
    pv_frozen_diagnostics.main([
        "--pairs-parquet", str(pairs),
        "--cde-parquet", str(catalog),
        "--out-dir", str(out),
    ])
    table = build_pv_overlap_table(out / "query_cde_pv_pair_records.csv").set_index("group")

    for group, want in PAPER_TABLE_S3.items():
        assert int(table.loc[group, "n_rows_both_present"]) == want["rows"], group
        assert round(100 * table.loc[group, "query_shorter_than_cde_rate"], 1) == want["shorter_pct"], group
        assert round(table.loc[group, "mean_jaccard_overlap"], 3) == want["jaccard"], group
