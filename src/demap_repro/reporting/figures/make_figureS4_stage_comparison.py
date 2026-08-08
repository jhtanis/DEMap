#!/usr/bin/env python3
"""Figure S4 — Validation Dev Recall@5 by fine-tuning stage.

Three backbones across off-the-shelf inference, Phase 1 fine-tuning and Phase 2
refinement. The fine-tuned stages show a two-seed mean with sample SD; the
off-the-shelf bars carry no error bar because inference is deterministic and
there is nothing to average.

Migrated from ``notebooks/paper_figures/figure_stage_comparison_v10.ipynb``, the
only producer of this figure. The plotting logic is unchanged; the notebook's
repository discovery and fixed output directory were replaced with the standard
data-root and figure-directory resolution.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from demap_repro.utils.paths import data_root


def build(repo: Path):
    """Render the figure. Returns the written paths."""
    import json
    import textwrap

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    from demap_repro.reporting.figures import style as S
    S.apply_style()
    REPO = repo

    DATA = REPO / ".scratch/demap/paper_v10_revision/data"
    SRC_P0 = DATA / "phase0_heatmaps.csv"
    SRC_P1 = DATA / "phase1_winners.csv"
    SRC_P2 = DATA / "phase2_winners.csv"
    NOTEBOOK = "notebooks/paper_figures/figure_stage_comparison_v10.ipynb"

    p0 = pd.read_csv(SRC_P0)
    p1 = pd.read_csv(SRC_P1)
    p2 = pd.read_csv(SRC_P2)
    print("phase 0 grid cells:", len(p0), "| models:", sorted(p0.model.unique()))
    assert set(p0.n_queries) == {4234}, "off-the-shelf grid must be the n=4234 Validation Dev split"

    FIG_W, FIG_H = 9.2, 5.6
    BAR_W = 0.26
    CAPSIZE = 4.0
    ERR_LW = 1.4
    VALUE_FMT = "{:.3f}"

    MODEL_ORDER = ["all-MPNet", "BioSimCSE", "PubMedBERT"]
    STAGE_ORDER = ["Off the shelf", "Phase 1", "Phase 2"]
    STAGE_COLOR = {"Off the shelf": S.PALETTE["bm25"],       # #999999
                   "Phase 1": S.PALETTE["crossencoder"],     # #56B4E9
                   "Phase 2": S.PALETTE["all-MPNet"]}        # #0072B2
    NO_ERRORBAR_STAGES = {"Off the shelf"}       # deterministic: no uncertainty to draw

    YLIM = (0.0, 1.06)
    YTICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    YLABEL = "Validation Dev Recall@5"

    rows = []
    best_cells = {}
    for m in MODEL_ORDER:
        g = p0[p0.model == m]
        b = g.loc[g.recall_at_5.idxmax()]
        best_cells[m] = f"{b.query_variant}/{b.query_label} × {b.cde_recipe_label}"
        rows.append(dict(model=m, stage="Off the shelf", mean=float(b.recall_at_5),
                         sd=np.nan, n_seeds=0, detail=best_cells[m]))
        a = p1[p1.model == m].iloc[0]
        rows.append(dict(model=m, stage="Phase 1", mean=float(a.val_dev_recall5_mean),
                         sd=float(a.val_dev_recall5_sd), n_seeds=2,
                         detail=f"{a.query_rep}/{a.cde_recipe}, {a.loss}, lr={a.lr}, t={a.temperature}"))
        c = p2[p2.model == m].iloc[0]
        rows.append(dict(model=m, stage="Phase 2", mean=float(c.val_dev_recall5_mean),
                         sd=float(c.val_dev_recall5_sd), n_seeds=2,
                         detail=f"negatives={c.negative_strategy}, lr={c.lr}, t={c.temperature}"))
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    print()
    print("off-the-shelf value = best cell of the 4 x 10 representation grid, per model:")
    for m in MODEL_ORDER:
        print(f"  {m:11} {best_cells[m]}  ({float(p0[p0.model == m].recall_at_5.max()):.6f}) "
              f"[{len(p0[p0.model == m])} grid cells searched]")
    print()
    print("stages plotted WITHOUT error bars (deterministic, no replicate):",
          sorted(NO_ERRORBAR_STAGES))
    assert tab[tab.stage.isin(NO_ERRORBAR_STAGES)].sd.isna().all()
    assert tab[~tab.stage.isin(NO_ERRORBAR_STAGES)].sd.notna().all()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    xs = np.arange(len(MODEL_ORDER), dtype=float)
    offsets = (np.arange(len(STAGE_ORDER)) - (len(STAGE_ORDER) - 1) / 2)
    ti = tab.set_index(["model", "stage"])

    for j, stage in enumerate(STAGE_ORDER):
        for i, m in enumerate(MODEL_ORDER):
            v = float(ti.loc[(m, stage), "mean"])
            sd = ti.loc[(m, stage), "sd"]
            x = xs[i] + offsets[j] * BAR_W
            ax.bar(x, v, BAR_W, color=STAGE_COLOR[stage], edgecolor="black", linewidth=0.5,
                   label=stage if i == 0 else None)
            top = v
            if stage not in NO_ERRORBAR_STAGES and pd.notna(sd):
                ax.errorbar(x, v, yerr=float(sd), fmt="none", ecolor="black",
                            elinewidth=ERR_LW, capsize=CAPSIZE, capthick=ERR_LW)
                top = v + float(sd)
            ax.text(x, top + 0.013, VALUE_FMT.format(v), ha="center", va="bottom",
                    fontsize=S.ANNOTATION_PT)

    ax.set_xticks(xs)
    ax.set_xticklabels(MODEL_ORDER, fontsize=S.TICK_LABEL_PT + 1)
    ax.set_xlim(-0.55, len(MODEL_ORDER) - 0.45)
    ax.set_ylim(*YLIM)
    ax.set_yticks(YTICKS)
    ax.set_ylabel(YLABEL, fontsize=S.AXIS_LABEL_PT)
    ax.set_xlabel("Bi-encoder backbone", fontsize=S.AXIS_LABEL_PT, labelpad=8)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncols=3, fontsize=S.LEGEND_PT)
    fig.tight_layout()

    fig.text(0.5, -0.055,
             "\n".join(textwrap.wrap(
                 "Off the shelf is deterministic (one forward pass, no training seed) and is "
                 "plotted without an error bar; the value is each model's best cell of the "
                 "4 x 10 representation grid. Phase 1 and Phase 2 bars are the mean over seeds "
                 "{0, 1} with sample SD (ddof = 1). n = 4234 Validation Dev pairs.", 96)),
             ha="center", va="top", fontsize=S.ANNOTATION_PT, linespacing=1.35)

    paths = S.save_figure(
        fig, "figS4_stage_comparison", sources=[SRC_P0, SRC_P1, SRC_P2], notebook=NOTEBOOK,
        description=("Validation Dev Recall@5 by training stage (off the shelf -> Phase 1 -> "
                     "Phase 2) for all-MPNet, BioSimCSE and PubMedBERT; off-the-shelf has no "
                     "error bar (deterministic), Phase 1/2 are two-seed mean +/- SD."),
        extra={"models": MODEL_ORDER, "stages": STAGE_ORDER, "split": "val_dev",
               "n_queries": 4234, "seeds_phase1_phase2": [0, 1],
               "off_the_shelf_definition": "best cell of the 4x10 representation grid per model",
               "off_the_shelf_best_cells": best_cells,
               "stages_without_error_bars": sorted(NO_ERRORBAR_STAGES),
               "supersedes": "notebooks/paper_figures/figure_validation_stage_comparison_phase2_winners.ipynb"})
    print({k: str(v) for k, v in paths.items()})

    plt.close("all")
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None,
                    help="data root holding the figure input tables")
    args = ap.parse_args(argv)
    paths = build(Path(args.root) if args.root else data_root())
    print({k: str(v) for k, v in paths.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
