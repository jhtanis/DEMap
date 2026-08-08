#!/usr/bin/env python3
"""Figures 2, S1 and S2 — the off-the-shelf 4 x 10 representation heatmaps.

One panel per bi-encoder backbone, showing Validation Dev Recall@5 for every
combination of four query representations and ten CDE text recipes, before any
fine-tuning. Figure 2 is all-MPNet; S1 is BioSimCSE; S2 is PubMedBERT.

Two cells are outlined on each panel: a solid box on the best cell, and a dashed
box on the prespecified anchor. The anchor matters more than the maximum. It was
fixed before the screen ran, so it is the honest basis for the downstream choice;
the best cell is reported for context and is not always the one carried forward.

The denominator is **pair-level** — 4,234 query-CDE pairs from 3,934 distinct
Validation Dev queries. The source CSV calls the column ``n_queries``, which is a
misnomer, and the assertion below pins the value so the mislabelling cannot
quietly become a different number.

Migrated from ``notebooks/paper_figures/figure_representation_heatmaps_v12.ipynb``
— the notebook was the only producer of these three figures. Converted to a
headless script: same data, same layout, no notebook runtime, and an explicit
output directory instead of a fixed one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from demap_repro.utils.paths import data_root

# Pair-level Validation Dev denominator.
VAL_DEV_PAIRS = 4234

FIG_W, FIG_H = 9.6, 4.5
CMAP = "YlOrRd"
ANNOT_FMT = "{:.3f}"

BEST_EDGE = {"edgecolor": "black", "linewidth": 2.6, "linestyle": "solid"}
ANCHOR_EDGE = {"edgecolor": "black", "linewidth": 2.2, "linestyle": (0, (4, 2))}

QUERY_ORDER = ["Q1", "Q2", "Q3", "Q4"]
RECIPE_ORDER = ["v3", "v2", "v1_v3", "v1_v2a_v3_v5", "v1_v2a_v2b_v5",
                "v1_v2a_v2b_v3", "v1_v2a_v2b_v3_v5", "v1_v6_v2b_v3_v5",
                "v1_v2a_v6_v2b_v3_v5", "v1_v2a_v2b_v3_v4_v5"]

#: The cell fixed before the screen ran, marked on every panel.
ANCHOR = ("Q3", "v1_v6_v2b_v3_v5")

PANELS = [
    ("all-MPNet", "fig2_heatmap_all_mpnet",
     "Figure 2: off-the-shelf all-MPNet 4x10 representation grid, Validation Dev Recall@5"),
    ("BioSimCSE", "figS1_heatmap_biosimcse",
     "Figure S1: off-the-shelf BioSimCSE 4x10 representation grid, Validation Dev Recall@5"),
    ("PubMedBERT", "figS2_heatmap_pubmedbert",
     "Figure S2: off-the-shelf PubMedBERT 4x10 representation grid, Validation Dev Recall@5"),
]

DEFAULT_SOURCE = ".scratch/demap/paper_v10_revision/data/phase0_heatmaps.csv"


def load_grid_table(source: Path) -> pd.DataFrame:
    df = pd.read_csv(source)
    if set(df["n_queries"]) != {VAL_DEV_PAIRS}:
        raise ValueError(
            f"expected the pair-level Validation Dev denominator {VAL_DEV_PAIRS}, "
            f"got {sorted(set(df['n_queries']))}")
    return df


def grid_for(df: pd.DataFrame, model: str) -> pd.DataFrame:
    g = (df[df["model"] == model]
         .pivot(index="query_variant", columns="cde_recipe", values="recall_at_5")
         .reindex(index=QUERY_ORDER, columns=RECIPE_ORDER))
    if not g.notna().all().all():
        raise ValueError(f"incomplete 4x10 grid for {model}")
    return g


def _heatmap(df, model, name, description, source, style):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    g = grid_for(df, model)
    sub = df[df["model"] == model]
    best = sub.nlargest(1, "recall_at_5").iloc[0]
    best_key = (best["query_variant"], best["cde_recipe"])

    qlabel = df.drop_duplicates("query_variant").set_index("query_variant")["query_label"].to_dict()
    rlabel = df.drop_duplicates("cde_recipe").set_index("cde_recipe")["cde_recipe_label"].to_dict()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    im = ax.imshow(g.values, cmap=CMAP, aspect="auto")
    ax.set_xticks(range(len(RECIPE_ORDER)))
    ax.set_xticklabels([rlabel[r] for r in RECIPE_ORDER], rotation=38,
                       ha="right", fontsize=style.TICK_LABEL_PT)
    ax.set_yticks(range(len(QUERY_ORDER)))
    ax.set_yticklabels([f"{q} / {qlabel[q]}" for q in QUERY_ORDER],
                       fontsize=style.TICK_LABEL_PT)
    ax.set_xlabel("CDE text representation", fontsize=style.AXIS_LABEL_PT)
    ax.set_ylabel("Query representation", fontsize=style.AXIS_LABEL_PT)
    ax.set_title(f"{model} (off the shelf)", fontsize=style.TITLE_PT, pad=10)
    ax.grid(False)

    for i, q in enumerate(QUERY_ORDER):
        for j, r in enumerate(RECIPE_ORDER):
            ax.text(j, i, ANNOT_FMT.format(g.loc[q, r]), ha="center", va="center",
                    fontsize=style.HEATMAP_CELL_PT, color="black")
            if (q, r) == ANCHOR:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       zorder=5, **ANCHOR_EDGE))
            if (q, r) == best_key:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       zorder=6, **BEST_EDGE))

    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.015)
    cb.set_label("Validation Dev Recall@5", fontsize=style.AXIS_LABEL_PT)
    cb.ax.tick_params(labelsize=style.TICK_LABEL_PT)

    handles = [Line2D([0], [0], color="black", lw=2.6, ls="solid", label="best cell"),
               Line2D([0], [0], color="black", lw=2.2, ls=(0, (4, 2)),
                      label="prespecified anchor")]
    fig.tight_layout()
    # Anchored below the figure box so bbox_inches="tight" expands the canvas
    # and the legend can never collide with the rotated recipe labels.
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06),
               ncol=2, fontsize=style.LEGEND_PT, frameon=False)

    paths = style.save_figure(
        fig, name, sources=[source],
        notebook="migrated from notebooks/paper_figures/figure_representation_heatmaps_v12.ipynb",
        description=description,
        extra={"model": model, "n_pairs": VAL_DEV_PAIRS, "n_queries_distinct": 3934,
               "split": "val_dev", "stage": "off_the_shelf",
               "anchor_cell": list(ANCHOR), "best_cell": list(best_key)})
    plt.close(fig)
    return paths, best


def build(source: Path):
    """Render all three heatmaps. Returns {figure name: written paths}."""
    from demap_repro.reporting.figures import style

    style.apply_style()
    df = load_grid_table(source)
    written = {}
    for model, name, description in PANELS:
        paths, best = _heatmap(df, model, name, description, source, style)
        written[name] = paths
        anchor = df[(df["model"] == model)
                    & (df["query_variant"] == ANCHOR[0])
                    & (df["cde_recipe"] == ANCHOR[1])].iloc[0]
        print(f"{model:11} best {best['query_label']} x {best['cde_recipe_label']} "
              f"= {best['recall_at_5']:.6f} (rank {int(best['rank_within_model'])})   "
              f"anchor = {anchor['recall_at_5']:.6f} "
              f"(rank {int(anchor['rank_within_model'])})")
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=None,
                    help=f"phase0_heatmaps.csv (default: <data root>/{DEFAULT_SOURCE})")
    args = ap.parse_args(argv)
    source = Path(args.source) if args.source else data_root() / DEFAULT_SOURCE
    for name, paths in build(source).items():
        print(name, {k: str(v) for k, v in paths.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
