"""Figure 4 inputs: off-the-shelf and fine-tuned cross-encoder Validation Dev Recall@5.

Figure 4 plots, per backbone, the epoch-0 (untrained) score, the fine-tuned score,
and the delta between them.

Why this is written fresh
-------------------------
The research repository's builder,
``.scratch/demap/paper_v10_revision/data2/_build_data2.py``, reads the
**pre-correction** roots ``artifacts/final_reranker/{crossencoder,
crossencoder_fulltrain}/``. Its own ``provenance.json`` records that. The CSVs it
once wrote were later overwritten by the v13 eligibility correction, but the
builder never was — so re-running it would silently restore the superseded Figure 4
values (MedCPT 0.9128 / BGE 0.8940 / MiniLM 0.8699).

That file is retained as historical provenance and is classified
``historical_reference_stale``. It must never become reproduction code. This module
reads the corrected ``*_v2_eligible`` roots instead and is parity-tested against
the CSVs that back the figure in v21.

Corrected sources
-----------------
fine-tuned (fulltrain, canonical)
    ``crossencoder_fulltrain_v2_eligible/comparison/BAKEOFF_ce_selection_val_dev.csv``
fine-tuned (val_train slice, backbone-selection evidence)
    ``crossencoder_v2_eligible/comparison/BAKEOFF_ce_selection_val_dev.csv``
off-the-shelf epoch 0
    ``crossencoder_v2_eligible/offtheshelf_epoch0/comparison/
    crossenc_eval_by_split_<recipe>_<recipe>_<backbone>__epoch0.csv``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from demap_repro.utils.paths import data_root

__all__ = ["BACKBONES", "build_finetuned", "build_offtheshelf", "build_figure4_inputs"]

RECIPE = "SN_DEC_DEF_PQT_PV"
SELECTION_SPLIT = "val_dev"
TRAINING_SEED = 20260527

#: display name -> HuggingFace id and the artifact tag used in file names.
BACKBONES: Dict[str, Dict[str, str]] = {
    "MedCPT": {"hf_model_id": "ncbi/MedCPT-Cross-Encoder",
               "tag": "ncbi_MedCPT-Cross-Encoder"},
    "BGE": {"hf_model_id": "BAAI/bge-reranker-base",
            "tag": "BAAI_bge-reranker-base"},
    "MiniLM": {"hf_model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
               "tag": "cross-encoder_ms-marco-MiniLM-L-6-v2"},
}

ERROR_BAR_NOTE = ("single training run (one seed=20260527, no replicate); "
                  "no seed SD available -> no error bars possible")

# The six canonical datasets were never scored with an untrained cross-encoder.
# The gap is deliberate and is recorded rather than imputed.
CANONICAL_SPLITS = {
    "test": 3959, "cctg": 1097, "oid_alt": 1766,
    "cdash": 324, "gdc_combined": 72, "cimac_v2": 131,
}
OTS_GAP_NOTE = ("documented raw-CE canonical-score gap: no off-the-shelf CE was "
                "ever scored on the six canonical datasets; intentionally not "
                "filled, needs an approved eval job")


def _bakeoff(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "model_tag" not in df.columns:
        raise ValueError(f"{path}: expected a BAKEOFF selection CSV with model_tag")
    return df


def build_finetuned(fulltrain_root: Path, slice_root: Optional[Path] = None) -> pd.DataFrame:
    """Fine-tuned Validation Dev Recall@5 per backbone.

    Two stages are emitted. ``fulltrain`` is canonical — the models trained on the
    full train split, one of which ships. ``val_train`` is the earlier slice
    bake-off, retained as backbone-selection evidence because it is what first
    ordered the three backbones; it is not the reported result.
    """
    rows: List[Dict] = []
    stages = [("fulltrain", "canonical", fulltrain_root)]
    if slice_root is not None:
        stages.append(("val_train", "backbone_selection_evidence", slice_root))

    for stage, status, root in stages:
        src = root / "comparison" / "BAKEOFF_ce_selection_val_dev.csv"
        df = _bakeoff(src)
        df = df.sort_values("recall@5", ascending=False).reset_index(drop=True)
        for rank, (_, r) in enumerate(df.iterrows(), start=1):
            name = next(n for n, meta in BACKBONES.items() if meta["tag"] == r["model_tag"])
            rows.append({
                "backbone": name,
                "hf_model_id": BACKBONES[name]["hf_model_id"],
                "val_dev_recall_at_5": float(r["recall@5"]),
                "n_runs": 1,
                "selected": bool(stage == "fulltrain" and rank == 1),
                "run_dir": str(root / "runs" / r["model_tag"]),
                "stage": stage,
                "stage_status": status,
                "n_queries": int(r["n_queries"]) if "n_queries" in r else 3934,
                "rank_within_stage": rank,
                "training_seed": TRAINING_SEED,
                "source_file": str(src),
                "error_bar_note": ERROR_BAR_NOTE,
            })
    return pd.DataFrame(rows)


def build_offtheshelf(ots_root: Path) -> pd.DataFrame:
    """Epoch-0 Validation Dev Recall@5 per backbone, plus the recorded canonical gap."""
    rows: List[Dict] = []
    for name, meta in BACKBONES.items():
        src = (ots_root / "comparison"
               / f"crossenc_eval_by_split_{RECIPE}_{RECIPE}_{meta['tag']}__epoch0.csv")
        df = pd.read_csv(src)
        row = df[(df["split"] == SELECTION_SPLIT) & (df["method"] == "crossencoder")]
        if not len(row):
            raise ValueError(f"{src}: no val_dev crossencoder row")
        rows.append({
            "backbone": name, "hf_model_id": meta["hf_model_id"],
            "split": SELECTION_SPLIT,
            "recall_at_5": float(row.iloc[0]["recall@5"]),
            "n_queries": int(row.iloc[0]["n_queries"]),
            "n_runs": 1,
            "exists_finetuned_stage": True, "exists_offtheshelf_stage": True,
            "source_file": str(src),
            "note": ("epoch-0 (untrained backbone) scored on the CORRECTED "
                     "production-eligible val_dev pool; like-for-like with the "
                     "corrected fine-tuned values"),
        })
    for name, meta in BACKBONES.items():
        for split, n in CANONICAL_SPLITS.items():
            rows.append({
                "backbone": name, "hf_model_id": meta["hf_model_id"],
                "split": split, "recall_at_5": "MISSING", "n_queries": n, "n_runs": 0,
                "exists_finetuned_stage": True, "exists_offtheshelf_stage": False,
                "source_file": "MISSING", "note": OTS_GAP_NOTE,
            })
    return pd.DataFrame(rows)


def build_figure4_inputs(root: Path, out_dir: Optional[Path] = None):
    """Both Figure 4 tables from the corrected artifact roots."""
    fulltrain = root / "artifacts/final_reranker/crossencoder_fulltrain_v2_eligible"
    slice_root = root / "artifacts/final_reranker/crossencoder_v2_eligible"
    ots = slice_root / "offtheshelf_epoch0"

    finetuned = build_finetuned(fulltrain, slice_root if slice_root.exists() else None)
    offtheshelf = build_offtheshelf(ots)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        finetuned.to_csv(out_dir / "crossencoder_finetuned.csv", index=False)
        offtheshelf.to_csv(out_dir / "crossencoder_offtheshelf.csv", index=False)
    return finetuned, offtheshelf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(data_root()))
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    ft, ots = build_figure4_inputs(Path(args.root), Path(args.out_dir))
    print(ft[["backbone", "stage", "val_dev_recall_at_5", "selected"]].to_string(index=False))
    print()
    print(ots[ots["split"] == SELECTION_SPLIT][
        ["backbone", "recall_at_5"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
