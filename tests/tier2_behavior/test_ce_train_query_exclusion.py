"""The 172-query cross-encoder training exclusion.

S5.4 calls the CE training set the "corrected, production-eligible candidate
pool". That correction is this filter: 172 of the 47,645 train queries have no
obtainable positive and are removed before pair construction.

In the research repository the removal happened inside a Slurm chain job and left
only a log line behind. These tests pin the behaviour and the count.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from demap_repro.crossencoder.eligibility import (
    PAPER_N_DROPPED,
    PAPER_N_TRAIN_QUERIES_AFTER,
    PAPER_N_TRAIN_QUERIES_BEFORE,
    drop_train_queries_without_obtainable_positive as drop_queries,
)

pytestmark = pytest.mark.tier2

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "ce_train_query_exclusion.json"


@pytest.fixture(scope="module")
def exclusion():
    return json.loads(FIXTURE.read_text())


# --------------------------------------------------------------------------
# the published numbers
# --------------------------------------------------------------------------

def test_exactly_172_queries_removed_for_canonical_inputs(exclusion):
    assert exclusion["n_dropped"] == PAPER_N_DROPPED == 172
    assert exclusion["n_train_queries_before"] == PAPER_N_TRAIN_QUERIES_BEFORE == 47645
    assert exclusion["n_train_queries_after"] == PAPER_N_TRAIN_QUERIES_AFTER == 47473
    assert (exclusion["n_train_queries_before"] - exclusion["n_dropped"]
            == exclusion["n_train_queries_after"])


def test_dropped_query_set_is_exactly_the_recorded_one(exclusion):
    ids = exclusion["dropped_query_ids"]
    assert len(ids) == 172
    assert len(set(ids)) == 172, "duplicate ids in the dropped set"
    assert ids == sorted(ids), "fixture must stay sorted for a stable diff"
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    assert digest == exclusion["dropped_query_ids_sha256"]


def test_resulting_pool_size_matches_the_paper(exclusion):
    assert exclusion["n_train_pool_rows_after"] == 1311461


# --------------------------------------------------------------------------
# the behaviour
# --------------------------------------------------------------------------

def _pool():
    """Two train queries with a positive, one without; plus eval rows."""
    return pd.DataFrame({
        "split": ["train"] * 5 + ["val_dev"] * 3 + ["test"] * 2,
        "query_id": ["q1", "q1", "q2", "q3", "q3", "d1", "d1", "d2", "t1", "t2"],
        "cde_id": list("abcdefghij"),
        "is_label": [True, False, False, False, True,
                     False, False, False, False, False],
    })


def test_removes_only_queries_with_no_positive():
    out, dropped = drop_queries(_pool(), verbose=False)
    assert dropped == ["q2"]
    assert set(out.loc[out["split"] == "train", "query_id"]) == {"q1", "q3"}


def test_every_retained_train_query_has_an_obtainable_positive():
    out, _ = drop_queries(_pool(), verbose=False)
    train = out[out["split"] == "train"]
    assert train.groupby("query_id")["is_label"].any().all()


def test_evaluation_rows_are_never_altered():
    """A query with no positive in val_dev or test must survive: those splits are
    reachability-filtered upstream and their denominators are reported."""
    before = _pool()
    out, dropped = drop_queries(before, verbose=False)
    assert "d2" not in dropped and "t1" not in dropped
    for split in ("val_dev", "test"):
        pd.testing.assert_frame_equal(
            before[before["split"] == split].reset_index(drop=True),
            out[out["split"] == split].reset_index(drop=True),
        )


def test_is_train_only_even_when_an_eval_query_shares_an_id():
    """Same query_id in train and val_dev: dropping it from train must not touch
    the val_dev rows."""
    pool = pd.DataFrame({
        "split": ["train", "val_dev"],
        "query_id": ["shared", "shared"],
        "cde_id": ["a", "b"],
        "is_label": [False, False],
    })
    out, dropped = drop_queries(pool, verbose=False)
    assert dropped == ["shared"]
    assert list(out["split"]) == ["val_dev"]


def test_noop_when_every_train_query_has_a_positive():
    pool = _pool()
    pool.loc[pool["query_id"] == "q2", "is_label"] = True
    out, dropped = drop_queries(pool, verbose=False)
    assert dropped == []
    pd.testing.assert_frame_equal(pool, out)


def test_noop_when_there_is_no_train_split():
    pool = _pool()
    pool = pool[pool["split"] != "train"]
    out, dropped = drop_queries(pool, verbose=False)
    assert dropped == []
    pd.testing.assert_frame_equal(pool, out)


def test_dropped_ids_are_returned_sorted():
    pool = pd.DataFrame({
        "split": ["train"] * 3,
        "query_id": ["zz", "aa", "mm"],
        "cde_id": list("abc"),
        "is_label": [False, False, False],
    })
    _, dropped = drop_queries(pool, verbose=False)
    assert dropped == ["aa", "mm", "zz"]


def test_missing_required_column_raises():
    with pytest.raises(KeyError, match="is_label"):
        drop_queries(pd.DataFrame({"split": ["train"], "query_id": ["q"]}), verbose=False)


# --------------------------------------------------------------------------
# against the real pool
# --------------------------------------------------------------------------

@pytest.mark.needs_artifacts
@pytest.mark.slow
def test_published_pool_contains_no_positive_free_train_query(exclusion, artifact_root):
    """Re-running the filter on the published pool must be a no-op — it already ran."""
    pool_path = artifact_root / exclusion["source_pool"]
    if not pool_path.exists():
        pytest.skip(f"{pool_path} not present")
    pool = pd.read_parquet(pool_path, columns=["split", "query_id", "is_label"])
    out, dropped = drop_queries(pool, verbose=False)
    assert dropped == []
    assert len(out) == len(pool)

    train = pool[pool["split"] == "train"]
    assert train["query_id"].nunique() == exclusion["n_train_queries_after"]
    assert len(train) == exclusion["n_train_pool_rows_after"]


@pytest.mark.needs_artifacts
@pytest.mark.slow
def test_dropped_set_is_exactly_split_minus_pool(exclusion, artifact_root):
    pool_path = artifact_root / exclusion["source_pool"]
    split_path = artifact_root / exclusion["source_split"]
    if not (pool_path.exists() and split_path.exists()):
        pytest.skip("artifacts not present")
    pool = pd.read_parquet(pool_path, columns=["split", "query_id"])
    in_pool = set(pool.loc[pool["split"] == "train", "query_id"].astype(str))
    in_split = set(pd.read_parquet(split_path, columns=["query_id"])["query_id"].astype(str))
    assert sorted(in_split - in_pool) == exclusion["dropped_query_ids"]
