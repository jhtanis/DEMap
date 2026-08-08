"""Cross-encoder selection reproduces the published winner.

Figure 4 and S5.4 report a three-way bake-off decided on Validation Dev Recall@5.
The per-backbone evaluation CSVs are small and public, so they are committed as a
fixture and the selection replays here offline — no checkpoints, no GPU.

What this pins is the *decision*, not the training: given those evaluation
results, the pipeline must still choose FT-MedCPT and must still report
0.912 / 0.904 / 0.874.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from demap_repro.crossencoder.select import MODELS, SEL_METRIC, SEL_SPLIT, select

pytestmark = [pytest.mark.tier3, pytest.mark.parity]

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "ce_bakeoff"

# S5.4, to three decimals as printed in the manuscript.
PAPER_VAL_DEV_RECALL5 = {
    "ncbi_MedCPT-Cross-Encoder": 0.912,
    "BAAI_bge-reranker-base": 0.904,
    "cross-encoder_ms-marco-MiniLM-L-6-v2": 0.874,
}


@pytest.fixture(scope="module")
def published():
    return json.loads((FIXTURE / "CE_WINNER_published.json").read_text())


@pytest.fixture
def replayed(tmp_path):
    """Run the selection over a writable copy of the fixture bake-off."""
    root = tmp_path / "crossencoder_fulltrain_v2_eligible"
    shutil.copytree(FIXTURE, root)
    return select(root, None)


def test_selection_is_on_val_dev_recall_at_5():
    assert SEL_SPLIT == "val_dev"
    assert SEL_METRIC == "recall@5"


def test_three_backbones_compared():
    assert set(MODELS) == {
        "ncbi_MedCPT-Cross-Encoder",
        "BAAI_bge-reranker-base",
        "cross-encoder_ms-marco-MiniLM-L-6-v2",
    }


def test_ft_medcpt_is_selected(replayed):
    assert replayed["winner_tag"] == "ncbi_MedCPT-Cross-Encoder"
    assert replayed["winner_label"] == "MedCPT-Cross-Encoder"


def test_ranking_matches_the_manuscript(replayed):
    order = [r["model_tag"] for r in replayed["ranking"]]
    assert order == [
        "ncbi_MedCPT-Cross-Encoder",
        "BAAI_bge-reranker-base",
        "cross-encoder_ms-marco-MiniLM-L-6-v2",
    ]
    for row in replayed["ranking"]:
        expected = PAPER_VAL_DEV_RECALL5[row["model_tag"]]
        assert round(row["recall@5"], 3) == expected, row["model_tag"]


def test_replay_matches_the_published_winner_record(replayed, published):
    assert replayed["winner_tag"] == published["winner_tag"]
    assert replayed["val_dev_metrics"] == published["val_dev_metrics"]
    assert ([r["model_tag"] for r in replayed["ranking"]]
            == [r["model_tag"] for r in published["ranking"]])
    for a, b in zip(replayed["ranking"], published["ranking"]):
        assert a == b


def test_winner_checkpoint_points_into_the_runs_tree(replayed):
    assert replayed["winner_checkpoint"].endswith("runs/ncbi_MedCPT-Cross-Encoder")


def test_protocol_string_records_the_frozen_settings(replayed):
    protocol = replayed["protocol"]
    for token in ("ep2", "lr2e-5", "bs32", "len512", "seed 20260527"):
        assert token in protocol


def test_missing_backbone_fails_loudly(tmp_path):
    """A partial bake-off must not quietly select from whatever survived."""
    root = tmp_path / "ce"
    shutil.copytree(FIXTURE, root)
    victim = next((root / "comparison").glob("*BAAI*"))
    victim.unlink()
    with pytest.raises(FileNotFoundError, match="missing corrected eval"):
        select(root, None)


def test_write_false_leaves_no_output(tmp_path):
    root = tmp_path / "ce"
    shutil.copytree(FIXTURE, root)
    before = {p.name for p in root.rglob("*")}
    select(root, None, write=False)
    assert {p.name for p in root.rglob("*")} == before
