#!/usr/bin/env python3
"""Aggregate the cross-encoder bake-off and select the winner.

Three backbones are fine-tuned under one frozen protocol and compared on
Validation Dev Recall@5 while reranking the same fixed candidate pool. The
highest score wins; there is no tie-break rule because the margins are not close
(FT-MedCPT 0.912 > FT-BGE 0.904 > FT-MiniLM 0.874) and each backbone was trained
once, so the comparison carries no error bars.

Writes ``BAKEOFF_ce_by_split_all_models.csv``,
``BAKEOFF_ce_selection_val_dev.csv``, an old-vs-corrected comparison per backbone
when the pre-correction results are present, and ``CE_WINNER.json`` — the file
every downstream stage reads to find the selected checkpoint.

Migrated from ``.scratch/demap/paper_v13_scientific_audit/chain_F4_ce_select_fulltrain_v2.py``,
the executed producer of the published selection. The logic is unchanged; the
straight-line script body is wrapped in ``main()`` so importing the module has no
side effects, and the roots come from ``DEMAP_DATA_ROOT`` instead of an absolute
path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from demap_repro.utils.paths import data_root

RECIPE = "SN_DEC_DEF_PQT_PV"
SEL_SPLIT = "val_dev"
SEL_METRIC = "recall@5"
MODELS = {
    "cross-encoder_ms-marco-MiniLM-L-6-v2": "MiniLM (ms-marco-MiniLM-L-6-v2)",
    "ncbi_MedCPT-Cross-Encoder": "MedCPT-Cross-Encoder",
    "BAAI_bge-reranker-base": "BGE-reranker-base",
}
MK = ["recall@1", "recall@5", "recall@10", "mrr@100"]

DEFAULT_ROOT = "artifacts/final_reranker/crossencoder_fulltrain_v2_eligible"
DEFAULT_OLD_ROOT = "artifacts/final_reranker/crossencoder_fulltrain"


def select(ce_root: Path, old_root: Path | None = None, *, write: bool = True) -> dict:
    """Aggregate the bake-off under ``ce_root`` and return the winner record.

    ``write=False`` computes the selection without touching disk, which is how the
    regression test replays it against the published artifacts.
    """
    cmp_dir = ce_root / "comparison"
    runs_dir = ce_root / "runs"

    rows = []
    for tag, label in MODELS.items():
        f = cmp_dir / f"crossenc_eval_by_split_{RECIPE}_{tag}.csv"
        if not f.is_file():
            raise FileNotFoundError(f"missing corrected eval: {f}")
        d = pd.read_csv(f)
        d = d[d["method"] == "crossencoder"].copy()   # the reranked-pool rows only
        if not len(d):
            raise ValueError(f"no crossencoder rows in {f}")
        d["model_tag"], d["model"] = tag, label
        rows.append(d)

    allm = pd.concat(rows, ignore_index=True)
    if write:
        allm.to_csv(cmp_dir / "BAKEOFF_ce_by_split_all_models.csv", index=False)

    sel = (allm[allm["split"] == SEL_SPLIT]
           .sort_values(SEL_METRIC, ascending=False).reset_index(drop=True))
    if write:
        sel.to_csv(cmp_dir / "BAKEOFF_ce_selection_val_dev.csv", index=False)
    win = sel.iloc[0]
    winner_tag = win["model_tag"]

    # old-vs-corrected per backbone on val_dev, when the pre-correction run is present
    old_rows = []
    if old_root is not None:
        for tag in MODELS:
            fo = old_root / "comparison" / f"crossenc_eval_by_split_{RECIPE}_{tag}.csv"
            if fo.is_file():
                do = pd.read_csv(fo)
                do = do[(do["split"] == SEL_SPLIT) & (do["method"] == "crossencoder")] \
                    .assign(model_tag=tag, which="old")
                old_rows.append(do)
    comp = (pd.concat(old_rows + [sel.assign(which="corrected")], ignore_index=True)
            if old_rows else sel.assign(which="corrected"))
    if write:
        comp.to_csv(cmp_dir / "BAKEOFF_old_vs_corrected_val_dev.csv", index=False)

    winner = {
        "selection_split": SEL_SPLIT, "selection_metric": SEL_METRIC,
        "winner_tag": winner_tag, "winner_label": MODELS[winner_tag],
        "winner_checkpoint": str(runs_dir / winner_tag),
        "val_dev_metrics": {k: float(win[k]) for k in MK if k in win},
        "ranking": [{"model_tag": r["model_tag"],
                     **{k: float(r[k]) for k in MK if k in r}}
                    for _, r in sel.iterrows()],
        "pairs": ("corrected FULLTRAIN pairs (eligibility-enforced; 0 ineligible, "
                  "0 empty-text)"),
        "protocol": ("frozen fulltrain: BCE pointwise, ep2, lr2e-5, bs32, len512, "
                     "seed 20260527, TRAIN=full train split"),
    }
    if write:
        (ce_root / "CE_WINNER.json").write_text(json.dumps(winner, indent=2))
    return winner


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ce-root", default=None,
                    help=f"cross-encoder artifact root (default: <data root>/{DEFAULT_ROOT})")
    ap.add_argument("--old-ce-root", default=None,
                    help="pre-correction root for the old-vs-corrected comparison; "
                         "omit to skip that comparison")
    args = ap.parse_args(argv)

    root = data_root()
    ce_root = Path(args.ce_root) if args.ce_root else root / DEFAULT_ROOT
    if args.old_ce_root:
        old_root = Path(args.old_ce_root)
    else:
        candidate = root / DEFAULT_OLD_ROOT
        old_root = candidate if candidate.exists() else None

    winner = select(ce_root, old_root)
    print(json.dumps(winner, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
