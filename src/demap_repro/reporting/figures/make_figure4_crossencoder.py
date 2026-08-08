#!/usr/bin/env python3
"""Figure 4 — Off-the-shelf and fine-tuned cross-encoder Validation Dev Recall@5.

Regenerated for manuscript v17. Changes relative to the previously embedded
raster:

  * the embedded four-line footer (which duplicated caption material) is
    removed -- the formal numbered caption lives in Word, outside the image;
  * axes, legend, backbone names, bar value labels and the Delta annotations
    are retained.

Delta is computed here from the full-precision values, exactly as before, and
asserted against the plotted rounding at build time.

Source of record for the numbers:
  .scratch/demap/paper_v10_revision/data2/crossencoder_selection.csv
  (see notebooks/paper_figures/figure_crossencoder_v10.ipynb)

Usage:  python manuscript/figures/make_figure4_crossencoder.py
Outputs: manuscript/figures/figure4_crossencoder_v17.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent

# --------------------------------------------------------------------- values
# (backbone, off-the-shelf Recall@5, fine-tuned Recall@5)
ROWS = [
    ("MedCPT", 0.548805287239451,  0.9123029994916116),
    ("BGE",    0.6014234875444839, 0.9036603965429588),
    ("MiniLM", 0.6077783426537875, 0.8741738688357905),
]

# Delta must be fine-tuned minus off-the-shelf, computed at full precision.
DELTAS = [ft - ots for _, ots, ft in ROWS]
EXPECTED = [0.3634977122521606, 0.3022369089984749, 0.2663955261820030]

def main(argv=None) -> int:
    """Render Figure 4."""
    for got, exp in zip(DELTAS, EXPECTED):
        assert abs(got - exp) < 1e-12, f"delta mismatch: {got!r} vs {exp!r}"

    # --------------------------------------------------------------------- styling
    WIDTH_IN, HEIGHT_IN = 7.0, 3.95
    AXIS_LABEL_PT, TICK_PT, LEGEND_PT, ANNOT_PT = 13, 11.5, 11.5, 11

    C_OTS, C_FT = "#9E9E9E", "#3FA9F5"      # grey / blue, as previously used

    matplotlib.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
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
    w = 0.36
    ots = [r[1] for r in ROWS]
    ft = [r[2] for r in ROWS]

    b1 = ax.bar(x - w / 2, ots, w, label="Off the shelf",
                color=C_OTS, edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + w / 2, ft, w, label="Fine-tuned",
                color=C_FT, edgecolor="black", linewidth=0.6)

    for bars, vals in ((b1, ots), (b2, ft)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=ANNOT_PT)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r[0]}\nΔ +{d:.3f}" for r, d in zip(ROWS, DELTAS)])
    ax.set_xlabel("Cross-encoder backbone")
    ax.set_ylabel("Validation Dev Recall@5")
    ax.set_ylim(0, 1.06)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)

    pdf = OUT / "figure4_crossencoder_v17.pdf"
    png = OUT / "figure4_crossencoder_v17.png"
    fig.savefig(pdf)
    fig.savefig(png)
    print("wrote", pdf)
    print("wrote", png)
    for (name, o, f), d in zip(ROWS, DELTAS):
        print(f"  {name:8s} ots={o:.6f} ft={f:.6f} delta={d:.6f} -> plotted +{d:.3f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
