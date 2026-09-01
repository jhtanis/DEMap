"""Candidate-pool size selection reproduces K = 30.

S5.3 and Figure S5. The grid is a small public CSV and is committed, so the whole
decision replays offline.

The branch matters as much as the number: the grid also contains ``clone`` and
``clone_or_fuzzy`` arms explored during development, and selecting on those would
give different ceilings. The paper pool is ``kwfuzzy`` only, so that the Python
approximation to NCI CDE Match stays an independent comparison method in Table 4
rather than an ingredient of the system it is compared against.

SUPERSEDED FOR COVERAGE. ``tests/fixtures/k_selection_grid.csv`` also carries
union gold-coverage counts, and they are NOT the canonical ones. The grid
predates the production-catalog eligibility correction and disagrees with the
shipped pool on exactly three queries: OID ALT 1,528 vs 1,527, GDC 71 vs 72,
CIMAC 114 vs 113. The GDC cell was a false positive - a retired version sharing
the gold's public identifier sat at rank 1, and public-id matching credited it.

The grid remains correct for what it is used for here, which is the K = 20/30/40/60
ceiling comparison behind Figure S5 and the choice of K. For candidate-pool
coverage use :mod:`demap_repro.pool.coverage` and Table S5 instead.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from demap_repro.pool.select_k import (
    PAPER_BRANCH,
    PAPER_KEYWORD_DEPTH,
    PAPER_SELECTED_K,
    PAPER_SELECTION_SPLIT,
    SE_DEPTHS,
    compute_k_grid,
    nominal_k,
    select_k,
)

pytestmark = [pytest.mark.tier3, pytest.mark.parity]

GRID_CSV = Path(__file__).resolve().parents[1] / "fixtures" / "k_selection_grid.csv"


@pytest.fixture(scope="module")
def grid():
    return pd.read_csv(GRID_CSV)


@pytest.fixture(scope="module")
def selection(grid):
    return select_k(grid)


# --------------------------------------------------------------------------
# the decision
# --------------------------------------------------------------------------

def test_selects_k_30(selection):
    assert selection["selected_k"] == PAPER_SELECTED_K == 30
    assert selection["selected_se_k"] == 20
    assert selection["keyword_depth"] == PAPER_KEYWORD_DEPTH == 10


def test_selected_on_val_train(selection):
    assert selection["selection_split"] == PAPER_SELECTION_SPLIT == "val_train"


def test_val_dev_is_descriptive_and_agrees(selection):
    """S5.3: 'Validation Dev gives the same ceiling Recall@5 of 0.9888 at K = 30.'"""
    assert selection["descriptive_split"] == "val_dev"
    assert round(selection["descriptive_ceiling_recall"], 4) == 0.9888


def test_paper_branch_is_the_keyword_fuzzy_arm(selection):
    assert selection["branch"] == PAPER_BRANCH == "kwfuzzy"


# --------------------------------------------------------------------------
# the published ceiling curve
# --------------------------------------------------------------------------

def test_ceiling_at_k20_and_k30(selection):
    """S5.3: 'rises from 0.9828 at K = 20 to 0.9888 at K = 30'."""
    by_k = {r["k_nominal"]: r for r in selection["grid"]}
    assert round(by_k[20]["pool_recall"], 4) == 0.9828
    assert round(by_k[30]["pool_recall"], 4) == 0.9888


def test_curve_flattens_after_k30(selection):
    """S5.3: 'gaining only 0.0036 between K = 30 and K = 60'."""
    by_k = {r["k_nominal"]: r for r in selection["grid"]}
    gain = by_k[60]["pool_recall"] - by_k[30]["pool_recall"]
    assert round(gain, 4) == 0.0036


def test_realized_pool_at_k30_is_about_27_6(selection):
    """S5.3: 'the mean realized pool contains 27.6 candidates'."""
    by_k = {r["k_nominal"]: r for r in selection["grid"]}
    assert by_k[30]["mean_pool_size"] == 27.6


def test_swept_depths_give_the_four_nominal_ks(selection):
    ks = [r["k_nominal"] for r in selection["grid"]]
    assert ks == [20, 30, 40, 60]
    assert tuple(r["se_k"] for r in selection["grid"]) == SE_DEPTHS


def test_ceiling_is_monotone_in_k(selection):
    recalls = [r["pool_recall"] for r in selection["grid"]]
    assert recalls == sorted(recalls), "a deeper pool cannot lower the ceiling"


def test_val_train_denominator(selection):
    assert all(r["n_queries"] == 3946 for r in selection["grid"])


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------

def test_nominal_k_is_the_sum_of_depths():
    assert nominal_k(20, 10) == 30
    assert nominal_k(50, 10) == 60


def test_unknown_branch_raises_rather_than_falling_back(grid):
    with pytest.raises(ValueError, match="no rows for dataset"):
        select_k(grid, branch="not_a_branch")


def test_clone_branches_are_present_but_not_selected(grid):
    """They exist in the grid; the paper path must never pick them by default."""
    assert {"clone", "clone_or_fuzzy"} <= set(grid["branch"].unique())
    assert PAPER_BRANCH == "kwfuzzy"


def test_a_tighter_tolerance_would_pick_a_larger_pool(grid):
    """The plateau rule is doing real work: with no tolerance the rule degenerates
    to 'take the deepest pool', which is K = 60."""
    strict = select_k(grid, plateau_tolerance=0.0)
    assert strict["selected_k"] == 60


# --------------------------------------------------------------------------
# the grid computation itself
# --------------------------------------------------------------------------

def test_compute_k_grid_counts_any_gold_in_pool():
    gold = {"q1": {"111"}, "q2": {"222", "333"}, "q3": {"999"}}
    be = {"q1": {"111": 5}, "q2": {"444": 1}, "q3": {"888": 1}}
    kw = {"q2": {"333": 3}}
    out = compute_k_grid(gold, be, kw, dataset="d", se_depths=[10], kw_depths=[10])
    row = out.iloc[0]
    assert row["n_queries"] == 3
    assert row["gold_in_pool"] == 2       # q1 via bi-encoder, q2 via keyword
    assert row["pool_recall"] == round(2 / 3, 4)


def test_compute_k_grid_respects_depth_cutoffs():
    gold = {"q1": {"111"}}
    be = {"q1": {"111": 15}}
    shallow = compute_k_grid(gold, be, {}, dataset="d", se_depths=[10], kw_depths=[10])
    deep = compute_k_grid(gold, be, {}, dataset="d", se_depths=[20], kw_depths=[10])
    assert shallow.iloc[0]["gold_in_pool"] == 0
    assert deep.iloc[0]["gold_in_pool"] == 1


def test_compute_k_grid_dedups_the_union_by_public_id():
    """A candidate found by both arms occupies one pool slot, which is why the
    realized pool is smaller than nominal K."""
    gold = {"q1": {"111"}}
    be = {"q1": {"111": 1, "222": 2}}
    kw = {"q1": {"111": 1, "333": 2}}
    out = compute_k_grid(gold, be, kw, dataset="d", se_depths=[20], kw_depths=[10])
    assert out.iloc[0]["mean_pool_size"] == 3.0   # 111, 222, 333 — not 4


def test_compute_k_grid_labels_the_keyword_free_baseline_none():
    gold = {"q1": {"111"}}
    out = compute_k_grid(gold, {"q1": {"111": 1}}, {}, dataset="d",
                         se_depths=[10], kw_depths=[0])
    assert out.iloc[0]["branch"] == "none"
