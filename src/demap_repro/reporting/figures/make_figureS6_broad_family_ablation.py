#!/usr/bin/env python3
"""Figure S6 - broad evidence-family ablation of the final HGBC (manuscript S5.7).

Two-panel diverging heatmap of the change in Recall@5 (A) and Recall@1 (B) when one of
the three broad evidence families is removed and the HGBC is retrained on the IDENTICAL
fixed Stage-1 candidate pool.

The three families are an exhaustive, mutually exclusive partition of all 117 final
features (98 / 13 / 6), validated in
`artifacts/final_reranker/broad_family_ablation_v1/summary/broad_family_map.json`.

Values are read from the validated canonical artifacts (never hardcoded):
    broad_family_recall5_delta.csv
    broad_family_recall1_delta.csv
committed as fixtures under tests/fixtures/, byte-identical to
artifacts/final_reranker/broad_family_ablation_v1/summary/.

The fine-grained nine-condition companion this was styled to match is retired: it
was v25 Figure S7 and is not part of the current manuscript, so it is deliberately
not migrated. The visual language is kept as-is.

House style: demap_repro.reporting.figures.style
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from demap_repro.reporting.figures import style as pfs  # noqa: E402
from demap_repro.utils.paths import artifact_root, repo_root  # noqa: E402

#: Delta tables. Committed as fixtures, so the figure regenerates offline; point
#: at the canonical artifact tree with --summary-dir to use it instead.
SUMM = repo_root() / "tests" / "fixtures"

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import AsinhNorm  # noqa: E402

# Only three rows, so the axes is short and wide: the 15 pt house title would
# overrun the axes and collide with the colorbar label. 13 pt still exceeds the
# >=12 pt house minimum for titles.
TITLE_PT = 13

DATASETS = ["Test", "CCTG", "OID ALT", "CDASH", "GDC", "CIMAC"]

# (condition key in the CSV, display label) - shared row order for both panels.
ROWS = [
    ("minus_lexical_keyword", "− Lexical/keyword-oriented evidence"),
    ("minus_ft_mpnet", "− FT-MPNet evidence"),
    ("minus_ft_medcpt", "− FT-MedCPT evidence"),
]

R5_NAME = "broad_family_recall5_delta.csv"
R1_NAME = "broad_family_recall1_delta.csv"


def delta_matrix(csv_path: Path) -> np.ndarray:
    d = pd.read_csv(csv_path).set_index("condition")
    return np.array([[float(d.loc[k, ds]) for ds in DATASETS] for k, _ in ROWS])


def cell_text(v: float) -> str:
    """Signed value to three decimals; exact zero prints unsigned."""
    if v == 0:
        return "0.000"
    if abs(v) < 0.0005:                      # would print as +/-0.000
        return f"{v:+.4f}"
    return f"{v:+.3f}"


def draw(ax, M, title):
    """One panel.

    Colour scale: diverging RdBu centred at zero (losses red, gains blue, near-zero
    white), on an ASINH norm. A linear norm is unusable here because the
    lexical/keyword-oriented row is an order of magnitude larger than the two neural
    rows and would flatten them to white. The asinh norm is near-linear within
    ``linear_width`` -- set from the largest NEURAL-family effect -- and compresses
    beyond it, so both scales stay readable without clipping. Exact values are printed
    in every cell regardless.
    """
    lim = float(np.abs(M).max())
    neural_max = float(np.abs(M[1:]).max())      # the two smaller families
    norm = AsinhNorm(linear_width=neural_max / 2.0, vmin=-lim, vmax=lim)
    im = ax.imshow(M, cmap="RdBu", norm=norm, aspect="auto")

    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS, rotation=30, ha="right")
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([lab for _, lab in ROWS])
    ax.set_title(title, pad=8, loc="left", fontsize=TITLE_PT)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(DATASETS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ROWS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    rgba = im.cmap(im.norm(M))
    lum = 0.299 * rgba[..., 0] + 0.587 * rgba[..., 1] + 0.114 * rgba[..., 2]
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, cell_text(M[i, j]), ha="center", va="center",
                    fontsize=pfs.HEATMAP_CELL_PT,
                    color="black" if lum[i, j] > 0.55 else "white")

    # Only three rows here, so the colorbar is short: keep the tick set sparse
    # (extreme, one intermediate, zero) or the labels collide once the figure is
    # scaled down to the manuscript column width.
    mid = -round(lim / 4, 2)
    ticks = sorted({-lim, mid, 0.0})
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.028, pad=0.02, ticks=ticks)
    cb.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda t, _: "0" if abs(t) < 1e-12 else f"{t:+.3f}".rstrip("0").rstrip(".")))
    cb.set_label("change vs full model", fontsize=pfs.ANNOTATION_PT)
    cb.ax.tick_params(labelsize=pfs.ANNOTATION_PT)
    cb.outline.set_visible(False)
    return im


def build(summary_dir: Path | None = None, fig_dir: Path | None = None):
    """Render Figure S6. Returns the written paths."""
    pfs.apply_style()
    mpl.rcParams["savefig.dpi"] = 600
    summ = Path(summary_dir) if summary_dir else SUMM
    if fig_dir:
        pfs.FIG_ROOT = Path(fig_dir)
    pfs.FIG_ROOT.mkdir(parents=True, exist_ok=True)
    R5, R1 = summ / R5_NAME, summ / R1_NAME
    for src in (R5, R1):
        if not src.is_file():
            raise SystemExit(f"ERROR: missing delta table: {src}")

    M5, M1 = delta_matrix(R5), delta_matrix(R1)
    assert M5.shape == (3, 6) and M1.shape == (3, 6), (M5.shape, M1.shape)

    fig, axes = plt.subplots(2, 1, figsize=(pfs.WIDTH_FULL, 4.9))
    draw(axes[0], M5, "(A)  Change in Recall@5 relative to the full HGBC")
    draw(axes[1], M1, "(B)  Change in Recall@1 relative to the full HGBC")
    fig.subplots_adjust(left=0.335, right=0.905, top=0.910, bottom=0.150, hspace=0.75)

    paths = pfs.save_figure(
        fig, "figureS6_broad_family_ablation",
        sources=[R5, R1],
        notebook="manuscript/figures/make_figureS6_broad_family_ablation.py",
        description=("Two-panel diverging heatmap of the change in Recall@5 (A) and Recall@1 (B) "
                     "when one of three broad evidence families is removed from the final HGBC "
                     "and the model is retrained on the identical fixed Stage-1 candidate pool."),
        extra={"datasets": DATASETS,
               "rows": [lab for _, lab in ROWS],
               "family_sizes": {"lexical_keyword": 98, "ft_mpnet": 13, "ft_medcpt": 6},
               "colour_scale": ("RdBu diverging, AsinhNorm centred at 0, symmetric limits and "
                                "linear_width chosen independently per panel from that panel's "
                                "largest neural-family effect"),
               "companion_figure": "figureS7_feature_group_ablation (fine-grained, 9 rows)"})
    fig.savefig(pfs.FIG_ROOT / "figureS6_broad_family_ablation.svg")
    print("wrote:", {k: str(v) for k, v in paths.items()})
    print("       ", pfs.FIG_ROOT / "figureS6_broad_family_ablation.svg")
    print(f"panel A limit +/-{np.abs(M5).max():.4f} lw={np.abs(M5[1:]).max()/2:.4f}   "
          f"panel B limit +/-{np.abs(M1).max():.4f} lw={np.abs(M1[1:]).max()/2:.4f}")

    return paths


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-dir", default=None,
                    help=f"directory holding the two delta CSVs (default: {SUMM})")
    ap.add_argument("--fig-dir", default=None,
                    help="output directory (default: $DEMAP_FIGURE_DIR or ./figures)")
    args = ap.parse_args(argv)
    build(args.summary_dir, args.fig_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
