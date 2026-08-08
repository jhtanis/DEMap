"""Shared style + provenance helpers for DEMAP manuscript figures (v10).

Font sizes are specified at the FINAL INSERTED WIDTH in the Word document, so a
figure exported at ``width_in`` inches and inserted at that same width keeps the
point sizes below. The v10 standard (author comment 4) is:

    panel letters   >= 15-16 pt
    figure titles   14-16 pt (only when needed)
    axis labels     >= 12-13 pt
    tick labels     >= 11 pt
    legends         >= 11 pt
    annotations     >= 10-11 pt
    heatmap cells   >= 9.5-10 pt

Nothing important should render below 10 pt after insertion.

Every figure exports a high-resolution PNG and a vector PDF, and writes a
provenance JSON recording the source artifacts and their SHA256 values.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_ROOT = REPO_ROOT / "notebooks" / "figures" / "paper_v10"

# ---------------------------------------------------------------- style knobs
# EDIT THESE to restyle every v10 figure at once.
PANEL_LETTER_PT = 16
TITLE_PT = 15
AXIS_LABEL_PT = 13
TICK_LABEL_PT = 11.5
LEGEND_PT = 11.5
ANNOTATION_PT = 11
HEATMAP_CELL_PT = 10

# Manuscript text column width (inches) used for single-column figures.
WIDTH_FULL = 7.0
WIDTH_WIDE = 9.0

# Colour-blind-safe qualitative palette (Okabe-Ito), used consistently for models
# and methods across every figure.
PALETTE = {
    "all-MPNet": "#0072B2",
    "BioSimCSE": "#009E73",
    "PubMedBERT": "#D55E00",
    "FT-MPNet": "#0072B2",
    "keyword": "#CC79A7",
    "cdematch_python": "#E69F00",
    "bm25": "#999999",
    "reranker": "#000000",
    "crossencoder": "#56B4E9",
}
SEQUENTIAL_CMAP = "viridis"


def apply_style() -> None:
    """Apply the v10 manuscript rcParams. Call once at the top of a notebook."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "font.family": "DejaVu Sans",
        "font.size": TICK_LABEL_PT,
        "axes.titlesize": TITLE_PT,
        "axes.labelsize": AXIS_LABEL_PT,
        "xtick.labelsize": TICK_LABEL_PT,
        "ytick.labelsize": TICK_LABEL_PT,
        "legend.fontsize": LEGEND_PT,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,      # embed TrueType so text stays editable
        "ps.fonttype": 42,
    })


def panel_letter(ax, letter: str, dx: float = -0.10, dy: float = 1.04) -> None:
    """Place a bold panel letter (A, B, C…) in axes coordinates."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=PANEL_LETTER_PT, fontweight="bold", va="bottom", ha="left")


def bar_value_labels(ax, bars, fmt: str = "{:.3f}", pad: float = 0.004,
                     fontsize: Optional[float] = None) -> None:
    """Print the exact value above each bar (author comment 30)."""
    for b in bars:
        h = b.get_height()
        if h != h:      # NaN
            continue
        ax.text(b.get_x() + b.get_width() / 2, h + pad, fmt.format(h),
                ha="center", va="bottom",
                fontsize=fontsize or ANNOTATION_PT)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        return "unknown"


def save_figure(fig, name: str, sources: Iterable[Path],
                notebook: str, description: str,
                extra: Optional[Dict] = None) -> Dict[str, Path]:
    """Export PNG + vector PDF and write a provenance JSON.

    Returns the written paths. ``sources`` are the authoritative artifacts the
    figure was built from; each is hashed into the provenance record.
    """
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    png = FIG_ROOT / f"{name}.png"
    pdf = FIG_ROOT / f"{name}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    prov = {
        "figure": name,
        "description": description,
        "generated": str(date.today()),
        "git_head": _git_head(),
        "notebook": notebook,
        "png": str(png.relative_to(REPO_ROOT)),
        "pdf": str(pdf.relative_to(REPO_ROOT)),
        "figure_size_inches": list(fig.get_size_inches()),
        "font_standard_pt": {
            "panel_letter": PANEL_LETTER_PT, "title": TITLE_PT,
            "axis_label": AXIS_LABEL_PT, "tick_label": TICK_LABEL_PT,
            "legend": LEGEND_PT, "annotation": ANNOTATION_PT,
            "heatmap_cell": HEATMAP_CELL_PT,
        },
        "sources": [],
    }
    for s in sources:
        s = Path(s)
        rec = {"path": str(s.relative_to(REPO_ROOT)) if s.is_absolute() and
               str(s).startswith(str(REPO_ROOT)) else str(s)}
        if s.is_file():
            rec["sha256"] = sha256(s)
            rec["size_bytes"] = s.stat().st_size
        else:
            rec["note"] = "directory or missing at export time"
        prov["sources"].append(rec)
    if extra:
        prov.update(extra)
    (FIG_ROOT / f"{name}_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    return {"png": png, "pdf": pdf,
            "provenance": FIG_ROOT / f"{name}_provenance.json"}
