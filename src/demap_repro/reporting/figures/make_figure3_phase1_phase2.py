#!/usr/bin/env python3
"""Figure 3 — Validation Dev Recall@5 after Phase 1 and Phase 2 fine-tuning.

Regenerated from the winner CSVs for manuscript v17. Changes relative to the
previously embedded raster:

  * the code-style strategy labels ``hard_top25`` / ``semihard_1_50`` / ``hard``
    are replaced by reader-facing labels ("Hard negatives, ranks 1-25",
    "Semi-hard negatives, ranks 1-50");
  * the embedded multi-line footer (which duplicated caption material and
    carried a code-level footnote about the historical ``hard`` label) is
    removed -- the formal numbered caption lives in Word, outside the image;
  * axes, legend, bar value labels, error bars and model names are retained.

Source of record for the numbers:
  .scratch/demap/paper_v10_revision/data/phase1_winners.csv
  .scratch/demap/paper_v10_revision/data/phase2_winners.csv

The all-MPNet Phase 2 artifact records the historical strategy string "hard";
its run_config has hardneg.hard_band == [1, 25], i.e. the same ranks-1-25 hard
strategy recorded as ``hard_top25`` for BioSimCSE. Both therefore carry the
identical reader-facing label and the code-level footnote is unnecessary.

Usage:  python manuscript/figures/make_figure3_phase1_phase2.py
Outputs: manuscript/figures/figure3_phase1_phase2_v17.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent

# --------------------------------------------------------------------- values
# (model, P1 mean, P1 sd, P2 mean, P2 sd, reader-facing Phase 2 strategy)
ROWS = [
    ("all-MPNet",
     0.8549834671705243, 0.004342176738509745,
     0.9013934813415210, 0.0008350339881749451,
     "Hard negatives,\nranks 1–25"),
    ("BioSimCSE",
     0.8436466698157771, 0.0016700679763498900,
     0.8940717997165801, 0.0001670067976350204,
     "Hard negatives,\nranks 1–25"),
    ("PubMedBERT",
     0.8186112423240435, 0.0227129244783586160,
     0.8965517241379310, 0.0000000000000000000,
     "Semi-hard negatives,\nranks 1–50"),
]

# --------------------------------------------------------------------- styling
WIDTH_IN, HEIGHT_IN = 7.4, 4.05          # house v10 full width + right room
AXIS_LABEL_PT, TICK_PT, LEGEND_PT, ANNOT_PT = 13, 11.5, 11.5, 11
STRATEGY_PT = 10.5

C_P1, C_P2 = "#6BAED6", "#08519C"       # light / dark blue, as previously used


def main(argv=None) -> int:
    """Render Figure 3."""
    matplotlib.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "font.family": "DejaVu Sans",
        "font.size": TICK_PT,
        "axes.labelsize": AXIS_LABEL_PT,
        "xtick.labelsize": TICK_PT,
        "ytick.labelsize": TICK_PT,
        "legend.fontsize": LEGEND_PT,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN))

    x = np.arange(len(ROWS))
    w = 0.34
    p1 = [r[1] for r in ROWS]
    p1e = [r[2] for r in ROWS]
    p2 = [r[3] for r in ROWS]
    p2e = [r[4] for r in ROWS]

    b1 = ax.bar(x - w / 2, p1, w, yerr=p1e, capsize=3.5, label="Phase 1",
                color=C_P1, edgecolor="black", linewidth=0.6,
                error_kw={"elinewidth": 1.2, "ecolor": "black"})
    b2 = ax.bar(x + w / 2, p2, w, yerr=p2e, capsize=3.5, label="Phase 2",
                color=C_P2, edgecolor="black", linewidth=0.6,
                error_kw={"elinewidth": 1.2, "ecolor": "black"})


    def label_bars(bars, vals, errs):
        for bar, v, e in zip(bars, vals, errs):
            ax.text(bar.get_x() + bar.get_width() / 2, v + e + 0.006,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=ANNOT_PT)


    label_bars(b1, p1, p1e)
    label_bars(b2, p2, p2e)

    # reader-facing Phase 2 negative-mining strategy, above each Phase 2 bar
    for xi, r in zip(x, ROWS):
        ax.text(xi + w / 2, r[3] + r[4] + 0.030, r[5],
                ha="center", va="bottom", fontsize=STRATEGY_PT,
                linespacing=1.25,
                bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                          edgecolor="0.55", linewidth=0.7))

    # The Phase-2 strategy annotation sits above the right-most bar and is the widest
    # element in the figure. With the default x-limits its box overflows the axes and
    # is clipped at the right edge. Reserve explicit room instead of cropping in Word.
    ax.set_xlim(-0.62, len(ROWS) - 1 + 0.92)

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in ROWS])
    ax.set_xlabel("Bi-encoder backbone")
    ax.set_ylabel("Validation Dev Recall@5")
    ax.set_ylim(0.75, 0.985)
    ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2)

    pdf = OUT / "figure3_phase1_phase2_v19.pdf"
    png = OUT / "figure3_phase1_phase2_v19.png"
    fig.savefig(pdf)
    fig.savefig(png)
    print("wrote", pdf)
    print("wrote", png)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
