#!/usr/bin/env python3
"""Figure S3 — representation-pair x loss screen for the biomedical backbones (v21).

Editable source, extracted verbatim from
`notebooks/paper_figures/figure_repxloss_v10.ipynb` (cells 1, 3, 5, 9), reading the
same CSV. Nothing is recomputed: every bar height, error bar, value label, selected
star, panel title, axis limit, axis label and category label is produced by the
notebook's own code from the same data.

Exactly two things differ from the notebook:

  1. The legend label for the second objective reads "Symmetric bidirectional"
     instead of "Symmetric MNRL", per the approved terminology. The internal
     dataframe key stays `symmetric_mnrl` because that is the value stored in the
     source CSV — only the DISPLAYED label changes.
  2. The embedded footer strip (`fig.text(...)`) is not drawn. The figure already
     in the manuscript is the notebook output with that footer cropped away, so
     omitting it reproduces what the document currently shows rather than
     re-introducing a duplicate of caption material.

Output is written to manuscript/figures/ and is sized to match the image it
replaces (3890 x 2132 px at 400 dpi, inserted at 6.40 x 3.51 in).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from demap_repro.utils.paths import data_root

#: Data and artifact tree. This was an absolute path into the research
#: repository, which made the module unusable anywhere else; see
#: ``demap_repro.utils.paths`` and ``DEMAP_DATA_ROOT``.
REPO = data_root()
from demap_repro.reporting.figures import style as S

# ``notebooks/paper_figures/paper_figure_style.py`` in the research repository;
# migrated here symbol-for-symbol. ``style.FIG_ROOT`` already honours
# DEMAP_FIGURE_DIR, so the output location no longer needs overriding.

def main(argv=None) -> int:
    """Render Figure S3."""
    S.apply_style()

    import matplotlib.pyplot as plt  # noqa: E402

    SRC = REPO / ".scratch/demap/paper_v10_revision/data/section4_3_repxloss.csv"

    df = pd.read_csv(SRC)
    assert set(df.n_seeds) == {2}, "every configuration must be a two-seed mean"

    # ------------------------------------------------------------------ knobs (verbatim)
    FIG_W, FIG_H = 9.6, 5.2
    BAR_W = 0.34
    CAPSIZE = 4.0
    ERR_LW = 1.4
    VALUE_FMT = "{:.3f}"

    PANEL_MODELS = ["BioSimCSE", "PubMedBERT"]
    REP_ORDER = ["anchor", "best"]
    LOSS_ORDER = ["mnrl", "symmetric_mnrl"]

    REP_TITLE = {"anchor": "Anchor pair", "best": "Best off-the-shelf pair"}

    # ---- the only display change: approved reader-facing loss terminology ----------
    LOSS_LABEL = {"mnrl": "MNRL", "symmetric_mnrl": "Symmetric bidirectional"}

    LOSS_COLOR = {"mnrl": S.PALETTE["cdematch_python"],
                  "symmetric_mnrl": S.PALETTE["all-MPNet"]}

    SELECTED_MARK = "★"
    Y_PAD_LOW, Y_PAD_HIGH = 0.048, 0.040
    YLABEL = "Validation Dev Recall@5"


    def rep_ticklabel(row):
        c = row.cde_recipe.split("/")[-1]
        return f"{REP_TITLE[row.rep_pair_label]}\n{row.query_rep}\n× {c}"


    plot_df = (df[df.model.isin(PANEL_MODELS)]
               .assign(rep_pair_label=lambda d: pd.Categorical(d.rep_pair_label, REP_ORDER, ordered=True),
                       loss=lambda d: pd.Categorical(d.loss, LOSS_ORDER, ordered=True))
               .sort_values(["model", "rep_pair_label", "loss"]))

    lo = float((plot_df.seed_mean_recall_at_5 - plot_df.sd).min()) - Y_PAD_LOW
    hi = float((plot_df.seed_mean_recall_at_5 + plot_df.sd).max()) + Y_PAD_HIGH
    YLIM = (lo, hi)
    print(f"shared y-limits (data-derived): {YLIM[0]:.6f} to {YLIM[1]:.6f}")
    for m in PANEL_MODELS:
        sel = plot_df[(plot_df.model == m) & (plot_df.selected)]
        assert len(sel) == 1, f"{m}: expected exactly one selected configuration"
        r = sel.iloc[0]
        print(f"selected {m:11}: {r.rep_pair_label} x {r.loss} = {r.seed_mean_recall_at_5:.6f}")

    # ------------------------------------------------------------------ draw (verbatim)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=True)

    for ax, model, letter in zip(axes, PANEL_MODELS, "AB"):
        sub = plot_df[plot_df.model == model]
        xs = np.arange(len(REP_ORDER), dtype=float)
        ticklabels = []
        for i, rep in enumerate(REP_ORDER):
            rows = sub[sub.rep_pair_label == rep]
            ticklabels.append(rep_ticklabel(rows.iloc[0]))
            for j, loss in enumerate(LOSS_ORDER):
                r = rows[rows.loss == loss].iloc[0]
                x = xs[i] + (j - 0.5) * BAR_W
                ax.bar(x, r.seed_mean_recall_at_5, BAR_W,
                       color=LOSS_COLOR[loss], edgecolor="black", linewidth=0.5,
                       label=LOSS_LABEL[loss] if (i == 0 and ax is axes[0]) else None)
                ax.errorbar(x, r.seed_mean_recall_at_5, yerr=r.sd, fmt="none",
                            ecolor="black", elinewidth=ERR_LW, capsize=CAPSIZE, capthick=ERR_LW)
                top = r.seed_mean_recall_at_5 + r.sd
                ax.text(x, top + 0.004, VALUE_FMT.format(r.seed_mean_recall_at_5),
                        ha="center", va="bottom", fontsize=S.ANNOTATION_PT)
                if bool(r.selected):
                    ax.text(x, top + 0.018, SELECTED_MARK, ha="center", va="bottom",
                            fontsize=S.ANNOTATION_PT + 4, color="black")

        ax.set_xticks(xs)
        ax.set_xticklabels(ticklabels, fontsize=S.TICK_LABEL_PT)
        ax.set_xlim(-0.55, len(REP_ORDER) - 0.45)
        ax.set_ylim(*YLIM)
        ax.set_title(model, fontsize=S.TITLE_PT, pad=8)
        ax.grid(axis="x", visible=False)
        S.panel_letter(ax, letter, dx=-0.09 if ax is axes[0] else -0.04, dy=1.03)

    axes[0].set_ylabel(YLABEL, fontsize=S.AXIS_LABEL_PT)
    fig.tight_layout()

    handles, labels = axes[0].get_legend_handles_labels()
    star = plt.Line2D([], [], linestyle="none", marker="*", markersize=14, color="black")
    fig.legend(handles=handles + [star], labels=labels + ["selected configuration"],
               loc="lower center", bbox_to_anchor=(0.5, -0.055), ncols=3,
               fontsize=S.LEGEND_PT)

    # No fig.text() footer — see module docstring.

    paths = S.save_figure(
        fig, "figureS3_repxloss_v21", sources=[SRC],
        notebook="manuscript/figures/make_figureS3_repxloss_v21.py",
        description=("Figure S3: representation-pair x loss screen for BioSimCSE and "
                     "PubMedBERT, Validation Dev Recall@5, two-seed mean +/- SD, shared "
                     "y-axis. Identical to the v10 notebook output except that the "
                     "legend reads 'Symmetric bidirectional' and the embedded footer "
                     "strip is omitted."),
        extra={"models": PANEL_MODELS, "split": "val_dev", "n_queries": 4234,
               "seeds": [0, 1], "rep_pairs": REP_ORDER, "losses": LOSS_ORDER,
               "loss_display_labels": LOSS_LABEL,
               "shared_ylim": list(YLIM), "recomputed": False})
    print({k: str(v) for k, v in paths.items()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
