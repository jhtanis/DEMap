#!/usr/bin/env python3
"""Figure S5 — ceiling Recall@5 against nominal candidate-pool size K.

Pools combine fine-tuned MPNet retrieval with a fixed keyword/fuzzy depth of 10.
K is selected on Validation Training; Validation Dev is plotted for description
only. K = 30 is the selected operating point.

Migrated from ``notebooks/paper_figures/figure_k_ceiling_v10.ipynb``, the only
producer of this figure. The selection logic itself lives in
:mod:`demap_repro.pool.select_k` and is tested there; this module only draws it.
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

    SRC = REPO / "notebooks/data/reranker_k_ceiling_recall5.csv"
    AUDIT_CSV = REPO / ".scratch/demap/reranker_k_ceiling_audit/reranker_k_ceiling_recall5.csv"
    AUDIT_PROV = REPO / ".scratch/demap/reranker_k_ceiling_audit/provenance.json"
    NOTEBOOK = "notebooks/paper_figures/figure_k_ceiling_v10.ipynb"

    df = pd.read_csv(SRC)
    df["K"] = df["K"].astype(int)
    print("source:", SRC.relative_to(REPO), "| sha256:", S.sha256(SRC))
    if AUDIT_CSV.exists():
        same = S.sha256(AUDIT_CSV) == S.sha256(SRC)
        print("matches the audited scratch copy:", same)
        assert same, "the durable copy has drifted from the audited CSV"
    prov = json.loads(AUDIT_PROV.read_text()) if AUDIT_PROV.exists() else {}
    print("audit provenance available:", bool(prov))

    FIG_W, FIG_H = 9.6, 4.6
    LINE_WIDTH = 2.2
    MARKER_SIZE = 7.5
    MARKER_SEL_SIZE = 13.0
    VLINE_WIDTH = 1.2
    VLINE_STYLE = (0, (4, 3))

    COLOR_SELECTION = S.PALETTE["all-MPNet"]        # val_train (selection split)
    COLOR_DESCRIPTIVE = S.PALETTE["cdematch_python"]  # val_dev (descriptive)
    COLOR_GUIDE = "#5a5a5a"

    XLIM = (14, 66)
    YLIM_FULL = (0.0, 1.03)
    YLIM_ZOOM = (0.980, 1.000)
    YTICKS_ZOOM = [0.980, 0.985, 0.990, 0.995, 1.000]
    ANNOT_BOX = dict(boxstyle="round,pad=0.32", facecolor="white",
                     edgecolor=COLOR_GUIDE, linewidth=0.8, alpha=0.95)
    ANNOT_OFFSET_FULL = (16, -54)      # points, relative to the selected point
    ANNOT_OFFSET_ZOOM = (18, -40)

    LABEL_X = "Candidate-pool size, K"
    LABEL_Y = "Ceiling Recall@5"
    LABEL_SELECTION = "val_train (selection split)"
    LABEL_DESCRIPTIVE = "val_dev (descriptive)"

    K_VALUES = [int(k) for k in sorted(df.K.unique())]
    SPLIT_SEL, SPLIT_DESC = "val_train", "val_dev"
    sel = df[df.split == SPLIT_SEL].sort_values("K").set_index("K")
    dev = df[df.split == SPLIT_DESC].sort_values("K").set_index("K")

    print(df[["split", "K", "ceiling_recall5", "gold_in_pool", "denominator",
              "gain_vs_prev_K", "add_queries_vs_prev_K", "realized_mean_pool_size",
              "pct_queries_lt_K", "selected"]].to_string(index=False))

    # ---- fact 1: the plotted K grid ---------------------------------------------------
    print("\nFACT 1  K values present in the CSV:", K_VALUES)
    assert K_VALUES == [20, 30, 40, 60], "unexpected K grid"

    # ---- fact 2: the keyword arm is fixed at depth 10 ----------------------------------
    print("FACT 2  keyword arm (audit record):", prov.get("keyword_arm", "unavailable"))
    print("        K sweep      (audit record):", prov.get("figure_k_sweep", "unavailable"))
    assert "depth 10" in prov.get("keyword_arm", ""), "keyword depth is not fixed at 10"

    # ---- fact 3: K = 30 selected, on val_train ----------------------------------------
    SELECTED_K = int(sel[sel.selected].index[0])
    print(f"FACT 3  selected K (CSV, {SPLIT_SEL}): {SELECTED_K}"
          f" | audit record: selected_K={prov.get('selected_K')},"
          f" selection_split={prov.get('selection_split')!r}")
    assert SELECTED_K == 30 == prov.get("selected_K")
    assert prov.get("selection_split") == SPLIT_SEL
    print(f"        {SPLIT_DESC} marks the same K but is DESCRIPTIVE ONLY - it did not select K.")

    # ---- fact 4: the ceiling at the selected K ----------------------------------------
    CEILING = {SPLIT_SEL: float(sel.loc[SELECTED_K, "ceiling_recall5"]),
               SPLIT_DESC: float(dev.loc[SELECTED_K, "ceiling_recall5"])}
    print("FACT 4  ceiling Recall@5 at K =", SELECTED_K, "->", CEILING)
    print("        realized mean pool size at the selected K:",
          float(sel.loc[SELECTED_K, "realized_mean_pool_size"]),
          "| denominators:", {SPLIT_SEL: int(sel.loc[SELECTED_K, "denominator"]),
                              SPLIT_DESC: int(dev.loc[SELECTED_K, "denominator"])})

    # ---- fact 5: K = 70 is excluded ---------------------------------------------------
    print("FACT 5  excluded larger grid point:", prov.get("full_grid_largest_K", "unavailable"))

    ANNOT_TEXT = f"K = {SELECTED_K}\nceiling Recall@5 = {CEILING[SPLIT_SEL]:.4f}"
    print("\nannotation text (built from the CSV):", repr(ANNOT_TEXT))

    def draw(ax, *, ylim, yticks=None, annot_offset, annotate=True):
        ax.axvline(SELECTED_K, color=COLOR_GUIDE, linestyle=VLINE_STYLE,
                   linewidth=VLINE_WIDTH, zorder=1)
        for frame, colour, label in [(sel, COLOR_SELECTION, LABEL_SELECTION),
                                     (dev, COLOR_DESCRIPTIVE, LABEL_DESCRIPTIVE)]:
            ax.plot(frame.index, frame["ceiling_recall5"], color=colour,
                    linewidth=LINE_WIDTH, marker="o", markersize=MARKER_SIZE,
                    markerfacecolor="white", markeredgewidth=1.8, label=label, zorder=3)
        ax.plot([SELECTED_K], [CEILING[SPLIT_SEL]], marker="o",
                markersize=MARKER_SEL_SIZE, color=COLOR_SELECTION, zorder=4)
        if annotate:
            ax.annotate(ANNOT_TEXT, xy=(SELECTED_K, CEILING[SPLIT_SEL]),
                        xytext=annot_offset, textcoords="offset points",
                        fontsize=S.ANNOTATION_PT, bbox=ANNOT_BOX, zorder=5)
        ax.set_xlabel(LABEL_X, fontsize=S.AXIS_LABEL_PT)
        ax.set_ylabel(LABEL_Y, fontsize=S.AXIS_LABEL_PT)
        ax.set_xlim(*XLIM)
        ax.set_xticks(K_VALUES)
        ax.set_ylim(*ylim)
        if yticks is not None:
            ax.set_yticks(yticks)
        ax.tick_params(labelsize=S.TICK_LABEL_PT)
        return ax

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
    draw(axA, ylim=YLIM_FULL, annot_offset=ANNOT_OFFSET_FULL)
    axA.set_title("Full range", fontsize=S.TITLE_PT, pad=8)
    S.panel_letter(axA, "A", dx=-0.16, dy=1.04)

    draw(axB, ylim=YLIM_ZOOM, yticks=YTICKS_ZOOM, annot_offset=ANNOT_OFFSET_ZOOM)
    axB.set_title("Zoom on the plateau (expanded axis)", fontsize=S.TITLE_PT, pad=8)
    S.panel_letter(axB, "B", dx=-0.20, dy=1.04)

    fig.tight_layout()
    handles, labels = axA.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.07),
               ncols=2, fontsize=S.LEGEND_PT)

    fig.text(0.5, -0.135,
             "\n".join(textwrap.wrap(
                 "Pools combine fine-tuned MPNet retrieval with a keyword/fuzzy arm held at "
                 f"depth 10 and are deduplicated by CDE public identifier. K was selected on "
                 f"{SPLIT_SEL} (n = {int(sel.loc[SELECTED_K, 'denominator'])}); {SPLIT_DESC} "
                 f"(n = {int(dev.loc[SELECTED_K, 'denominator'])}) is descriptive. Realized "
                 f"pools are smaller than nominal K after deduplication (mean "
                 f"{float(sel.loc[SELECTED_K, 'realized_mean_pool_size'])} at the selected K).",
                 98)),
             ha="center", va="top", fontsize=S.ANNOTATION_PT, linespacing=1.35)

    paths = S.save_figure(
        fig, "figS5_k_ceiling", sources=[SRC, AUDIT_PROV], notebook=NOTEBOOK,
        description=("Ceiling Recall@5 versus nominal reranker candidate-pool size K "
                     "(bi-encoder depth swept, keyword depth fixed at 10); full range and "
                     "plateau zoom. K = 30 selected on val_train; val_dev descriptive."),
        extra={"k_values_plotted": K_VALUES, "selected_K": SELECTED_K,
               "selection_split": SPLIT_SEL, "descriptive_split": SPLIT_DESC,
               "keyword_depth_fixed": 10,
               "ceiling_recall5_at_selected_K": CEILING,
               "excluded_from_curve": ("K=70 (se50 U kw20) also varies the keyword depth and is "
                                       "not on the fixed-kw10 line"),
               "authoritative_primary_source": prov.get("authoritative_primary_source"),
               "supersedes": "notebooks/reranker_k_ceiling_recall5.ipynb (pre-v10 styling)"})
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
