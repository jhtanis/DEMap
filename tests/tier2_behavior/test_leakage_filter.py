"""The train/evaluation overlap filter.

S1.5 defines it, S6.1 and Table S5 report every external result with and without
it. The filtered counts are load-bearing: 1,089 / 1,759 / 310 / 68 / 113, from
1,097 / 1,766 / 324 / 72 / 131.

The rule itself is small and text-based, so most of it is tested directly. The
published removal set is committed as a fixture, and the full replay against the
real evaluation parquets runs when ``DEMAP_ARTIFACT_ROOT`` is set.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from demap_repro.data.leakage_filter import (
    EXTERNAL_DATASETS,
    PAPER_COUNTS,
    PASSTHROUGH_DATASETS,
    build_leakage_filtered_sets,
    norm,
    publicid,
)

pytestmark = pytest.mark.tier2

FIXTURE = (Path(__file__).resolve().parents[1] / "fixtures"
           / "eval_canonical_v2_removed_queries.csv")


@pytest.fixture(scope="module")
def removed_fixture():
    with FIXTURE.open() as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------
# the rule's primitives
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Tumor Size (mm)", "tumor size mm"),
    ("TUMOR_SIZE", "tumor size"),
    ("  tumor   size  ", "tumor size"),
    ("Tumor-Size/Value", "tumor size value"),
    ("tumor size", "tumor size"),
])
def test_normalization_collapses_punctuation_and_case(raw, expected):
    assert norm(raw) == expected


def test_normalization_does_not_merge_distinct_queries():
    assert norm("tumor size") != norm("tumour size")
    assert norm("stage 1") != norm("stage 2")


@pytest.mark.parametrize("raw,expected", [
    ("2002587::5", "2002587"),
    ("2002587", "2002587"),
    (" 2002587::1 ", "2002587"),
])
def test_publicid_strips_the_version_suffix(raw, expected):
    assert publicid(raw) == expected


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------

def _write(tmp_path, name, rows):
    d = tmp_path / "src"
    d.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(d / f"{name}.parquet", index=False)
    return d


def _train(tmp_path, rows):
    p = tmp_path / "train.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


def _row(qid, text, cde, family="F", source="REF"):
    return dict(query_id=qid, query_text_raw=text, cde_publicid=cde,
                family=family, query_source=source)


def test_removes_query_matching_train_on_text_and_gold(tmp_path):
    src = _write(tmp_path, "cctg", [_row("e1", "Tumor Size (mm)", "111")])
    train = _train(tmp_path, [_row("t1", "tumor size mm", "111::3")])
    summary, removed = build_leakage_filtered_sets(
        src, train, datasets=["cctg"], passthrough=[])
    assert summary[0]["removed_queries"] == 1
    assert summary[0]["v2_queries"] == 0
    assert "norm_text+gold_in_train" in removed.iloc[0]["reason"]


def test_keeps_query_when_only_the_text_matches(tmp_path):
    """Same wording, different gold CDE: not the leakage the paper controls for."""
    src = _write(tmp_path, "cctg", [_row("e1", "Tumor Size", "111")])
    train = _train(tmp_path, [_row("t1", "tumor size", "999")])
    summary, removed = build_leakage_filtered_sets(
        src, train, datasets=["cctg"], passthrough=[])
    assert summary[0]["removed_queries"] == 0
    assert removed.empty


def test_keeps_query_when_only_the_gold_matches(tmp_path):
    """Same CDE reached from different wording is generalization, not memorization."""
    src = _write(tmp_path, "cctg", [_row("e1", "Size of primary lesion", "111")])
    train = _train(tmp_path, [_row("t1", "tumor size", "111")])
    summary, _ = build_leakage_filtered_sets(src, train, datasets=["cctg"], passthrough=[])
    assert summary[0]["removed_queries"] == 0


def test_gold_matching_ignores_the_version_suffix(tmp_path):
    src = _write(tmp_path, "cctg", [_row("e1", "tumor size", "111::7")])
    train = _train(tmp_path, [_row("t1", "tumor size", "111::2")])
    summary, _ = build_leakage_filtered_sets(src, train, datasets=["cctg"], passthrough=[])
    assert summary[0]["removed_queries"] == 1


def test_removal_is_whole_query_across_all_its_gold_rows(tmp_path):
    """A multi-gold query must not be left with a partial gold set."""
    src = _write(tmp_path, "cctg", [
        _row("e1", "tumor size", "111"),
        _row("e1", "tumor size", "222"),
        _row("e2", "other", "333"),
    ])
    train = _train(tmp_path, [_row("t1", "tumor size", "111")])
    summary, removed = build_leakage_filtered_sets(
        src, train, datasets=["cctg"], passthrough=[])
    assert summary[0]["removed_queries"] == 1
    assert summary[0]["removed_rows"] == 2
    assert summary[0]["v2_queries"] == 1
    assert set(removed["cde_publicid"]) == {"111", "222"}


def test_test_split_passes_through_untouched(tmp_path):
    src = _write(tmp_path, "test", [_row("e1", "tumor size", "111")])
    train = _train(tmp_path, [_row("t1", "tumor size", "111")])
    summary, removed = build_leakage_filtered_sets(
        src, train, datasets=[], passthrough=["test"])
    assert summary[0]["removed_queries"] == 0
    assert summary[0]["rule"].startswith("passthrough")
    assert removed.empty


def test_refuses_to_overwrite_existing_outputs(tmp_path):
    src = _write(tmp_path, "cctg", [_row("e1", "a", "1")])
    train = _train(tmp_path, [_row("t1", "b", "2")])
    out = tmp_path / "out"
    build_leakage_filtered_sets(src, train, datasets=["cctg"], passthrough=[], out_dir=out)
    with pytest.raises(FileExistsError):
        build_leakage_filtered_sets(src, train, datasets=["cctg"], passthrough=[], out_dir=out)


# --------------------------------------------------------------------------
# the published removal set
# --------------------------------------------------------------------------

def test_dataset_lists_match_the_paper():
    assert EXTERNAL_DATASETS == ("cctg", "oid_alt", "cdash", "gdc_combined", "cimac_v2")
    assert PASSTHROUGH_DATASETS == ("test",)


def test_fixture_holds_51_removed_queries(removed_fixture):
    queries = {(r["dataset"], r["query_id"]) for r in removed_fixture}
    assert len(queries) == 51
    assert len(removed_fixture) == 60, "60 rows because some removed queries are multi-gold"


def test_per_dataset_removals_match_the_paper(removed_fixture):
    counts = {}
    for r in removed_fixture:
        counts.setdefault(r["dataset"], set()).add(r["query_id"])
    for dataset, (_before, removed, _after) in PAPER_COUNTS.items():
        got = len(counts.get(dataset, set()))
        assert got == removed, f"{dataset}: removed {got}, paper says {removed}"


def test_paper_counts_are_internally_consistent():
    for dataset, (before, removed, after) in PAPER_COUNTS.items():
        assert before - removed == after, dataset


def test_every_removal_fired_on_the_text_and_gold_clause(removed_fixture):
    """S1.5 states the criterion as text+gold recurrence. The query-identifier
    clause is evaluated but never fires on the canonical inputs — so the stated
    criterion and the executed one coincide."""
    for r in removed_fixture:
        assert "norm_text+gold_in_train" in r["reason"]
    assert not any("query_id_in_train" in r["reason"] for r in removed_fixture)


def test_cimac_removal_is_the_18_reported_in_section_3_8(removed_fixture):
    cimac = {r["query_id"] for r in removed_fixture if r["dataset"] == "cimac_v2"}
    assert len(cimac) == 18


# --------------------------------------------------------------------------
# full replay
# --------------------------------------------------------------------------

@pytest.mark.needs_artifacts
@pytest.mark.slow
def test_rule_reproduces_the_published_removal_set(artifact_root, removed_fixture):
    src = artifact_root / "data/processed/eval_canonical"
    train = artifact_root / "data/processed/splits/train.parquet"
    if not (src.exists() and train.exists()):
        pytest.skip("canonical evaluation inputs not present")

    summary, removed = build_leakage_filtered_sets(src, train)

    for row in summary:
        before, n_removed, after = PAPER_COUNTS[row["dataset"]]
        assert row["v1_queries"] == before, row["dataset"]
        assert row["removed_queries"] == n_removed, row["dataset"]
        assert row["v2_queries"] == after, row["dataset"]

    replayed = {(r["dataset"], str(r["query_id"])) for _, r in removed.iterrows()}
    published = {(r["dataset"], r["query_id"]) for r in removed_fixture}
    assert replayed == published
