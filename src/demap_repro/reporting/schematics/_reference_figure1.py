#!/usr/bin/env python3
"""Figure 1 — "Source-to-CDE mapping workflow" (v20).

v20 change: the band-2 label "Phase 2 hard-negative refinement" overflowed its
box, so it is shortened to "Phase 2 refinement". The hard-negative detail is
carried by Supplementary S4.4. Nothing else changed.

Editable, deterministic source. Renders a vector PDF and a 600 dpi PNG with
matplotlib, so it regenerates on the cluster without PowerPoint or LibreOffice.

What changed from v17
---------------------
The v17 figure drew band 1 as a single four-box chain:

    caDSR export -> Benchmark construction -> Dedup/filtering -> Production catalog

which reads as though the June production catalog were *produced by* benchmark
filtering. It is not: it is a separate, later caDSR snapshot that supplies the
corpus for fine-tuning and for candidate retrieval. v19 therefore splits band 1
into two independent groups with no arrow between them:

  * the January benchmark-construction chain, ending at 68,142 analysis-ready
    pairs; and
  * the June production catalog, drawn as a separate input whose arrows point
    DOWN into the bi-encoder-development and candidate-generation stages.

Light group panels sit behind bands 2 and 3 so those arrows land on a stage
rather than on one arbitrary box inside it.

Also: the overflowing box title "Canonical deduplication and reachable-gold
filtering" is shortened to "Deduplication and reachability filtering".

Everything else is preserved from the approved v17 layout: FT-MPNet top 20,
CDE Match-Fuzzy top 10, union with public-identifier deduplication, ~30
candidates, three PARALLEL feature branches into the HGBC, final ranking, and
evaluation on the internal test set plus five holdouts. No DEMAP/DEMap wording
and no exact-match pinning appear anywhere.

Usage:  python manuscript/figures/make_figure1_workflow_v19.py
Outputs: manuscript/figures/figure1_workflow_v19.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent

# ----------------------------------------------------------------- style knobs
W, H = 6.80, 5.10                 # inches; 6.80 is the manuscript insertion width

TITLE_PT = 11.5
BAND_PT = 8.5
BOX_TITLE_PT = 7.4
BOX_BODY_PT = 6.8

C_DATA, L_DATA = "#DDEBF7", "#1F4E79"     # data / inputs        (light blue)
C_CORPUS, L_CORPUS = "#D6E9F8", "#12557F"  # production catalog  (deeper blue)
C_MODEL, L_MODEL = "#E2F0D9", "#375623"   # models / training    (light green)
C_RETR, L_RETR = "#FFF2CC", "#7F6000"     # retrieval            (light amber)
C_FEAT, L_FEAT = "#FBE4D5", "#833C0B"     # features             (light orange)
C_RANK, L_RANK = "#E6E0EC", "#403051"     # reranking/evaluation (light purple)
C_LINE = "#595959"
C_BODY = "#333333"
C_PANEL = "#F4F4F6"

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])

def main(argv=None) -> int:
    """Render the Figure 1 schematic (reference only)."""
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)                 # y grows downward, like a slide canvas
    ax.axis("off")


    def panel(x, y, w, h):
        """Light background marking one workflow stage, so a stage-level arrow can
        terminate on the stage rather than on one arbitrary box inside it."""
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07",
            linewidth=0, facecolor=C_PANEL, zorder=0))


    def box(x, y, w, h, title, body="", fill=C_DATA, edge=L_DATA,
            title_pt=BOX_TITLE_PT, body_pt=BOX_BODY_PT):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=0.055",
            linewidth=0.9, edgecolor=edge, facecolor=fill,
            mutation_aspect=1.0, zorder=2))
        cx = x + w / 2
        if body:
            ax.text(cx, y + h * 0.34, title, ha="center", va="center",
                    fontsize=title_pt, fontweight="bold", color=edge, zorder=3)
            ax.text(cx, y + h * 0.70, body, ha="center", va="center",
                    fontsize=body_pt, color=C_BODY, linespacing=1.30, zorder=3)
        else:
            ax.text(cx, y + h / 2, title, ha="center", va="center",
                    fontsize=title_pt, fontweight="bold", color=edge, zorder=3)


    def arrow(x1, y1, x2, y2, lw=1.0, rad=0.0, color=C_LINE, ls="-"):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=7, linewidth=lw, linestyle=ls,
            color=color, shrinkA=0, shrinkB=0, zorder=1,
            connectionstyle=f"arc3,rad={rad}"))


    def band(y, text, color):
        ax.text(0.12, y, text, ha="left", va="center",
                fontsize=BAND_PT, fontweight="bold", color=color)


    # --------------------------------------------------------------- graphic title
    ax.text(W / 2, 0.20, "Source-to-CDE mapping workflow",
            ha="center", va="center", fontsize=TITLE_PT, fontweight="bold",
            color="#202020")

    # --------------------------------------- 1  Data sources and benchmark construction
    band(0.50, "1   Data sources and benchmark construction", L_DATA)
    y1, bh1 = 0.66, 0.78

    # --- January benchmark-construction chain (three boxes, left group) ---------
    jan = [
        ("caDSR export", "January 12, 2026\n79,479-record benchmark-\nconstruction catalog"),
        ("Benchmark construction", "69,102 constructed\nquery–CDE pairs"),
        ("Deduplication and\nreachability filtering", "68,142 analysis-ready\npairs"),
    ]
    bw1, gap1 = 1.45, 0.12
    for i, (t, b) in enumerate(jan):
        x = 0.12 + i * (bw1 + gap1)
        box(x, y1, bw1, bh1, t, b, C_DATA, L_DATA)
        if i:
            arrow(x - gap1, y1 + bh1 / 2, x, y1 + bh1 / 2)

    # --- separate June production catalog (right group, NO incoming arrow) -----
    ax.plot([4.86, 4.86], [y1 - 0.05, y1 + bh1 + 0.05], color="#B4B4B4",
            linewidth=0.8, linestyle=(0, (2.5, 2.0)), zorder=1)
    jx, jw = 5.00, 1.68
    box(jx, y1, jw, bh1, "Production catalog",
        "June 18, 2026\n62,976 records;\n62,858 unique CDE public IDs",
        C_CORPUS, L_CORPUS)

    # ------------------------------------------------------ 2  Bi-encoder development
    band(1.80, "2   Bi-encoder development", L_MODEL)
    y2, bh2 = 1.96, 0.44
    panel(0.06, y2 - 0.06, 6.21, bh2 + 0.12)
    cols2 = ["Text-representation\nscreening", "Representation × loss\nscreening",
             "Phase 1\nfine-tuning", "Phase 2\nrefinement", "FT-MPNet\nselected"]
    bw2, gap2 = 1.15, 0.08
    for i, t in enumerate(cols2):
        x = 0.12 + i * (bw2 + gap2)
        box(x, y2, bw2, bh2, t, "", C_MODEL, L_MODEL, title_pt=6.6)
        if i:
            arrow(x - gap2, y2 + bh2 / 2, x, y2 + bh2 / 2)

    # ------------------------------------------ 3  Candidate generation and reranking
    band(2.62, "3   Candidate generation and reranking", L_FEAT)
    panel(0.06, 2.74, 6.46, 1.86)

    rx, rw, rh = 0.12, 1.34, 0.50
    box(rx, 2.84, rw, rh, "FT-MPNet", "semantic retrieval, top 20", C_RETR, L_RETR,
        title_pt=7.6, body_pt=6.6)
    box(rx, 3.52, rw, rh, "CDE Match-Fuzzy", "keyword retrieval, top 10", C_RETR, L_RETR,
        title_pt=7.6, body_pt=6.6)

    px, pw, py, ph = 1.62, 1.38, 3.00, 0.86
    box(px, py, pw, ph, "Candidate pool",
        "union, deduplicated by\nCDE public identifier\n≈ 30 candidates",
        C_RETR, L_RETR, title_pt=7.6, body_pt=6.6)
    arrow(rx + rw, 3.09, px, py + 0.26)
    arrow(rx + rw, 3.77, px, py + 0.60)

    fx, fw, fh = 3.24, 1.74, 0.44
    fys = [2.84, 3.43, 4.02]
    feats = [
        ("Bi-encoder features", "similarity, rank, margins"),
        ("Keyword features", "rule, rank, fuzzy and\ntoken-overlap evidence"),
        ("FT-MedCPT cross-encoder\nfeatures", "score, rank, within-query z"),
    ]
    for (t, b), fy in zip(feats, fys):
        box(fx, fy, fw, fh, t, b, C_FEAT, L_FEAT, title_pt=6.6, body_pt=6.1)
        arrow(px + pw, py + ph / 2, fx, fy + fh / 2)

    hx, hw = 5.18, 1.12
    box(hx, 3.16, hw, 0.54, "HGBC reranker", "", C_RANK, L_RANK, title_pt=7.6)
    box(hx, 4.00, hw, 0.46, "Final ranking", "", C_RANK, L_RANK, title_pt=7.6)
    for fy in fys:
        arrow(fx + fw, fy + fh / 2, hx, 3.16 + 0.27)
    arrow(hx + hw / 2, 3.70, hx + hw / 2, 4.00)

    # --- the production catalog feeds stages 2 and 3, not the benchmark chain ---
    arrow(jx + jw * 0.32, y1 + bh1, 3.30, y2 - 0.06, lw=1.0, rad=-0.12,
          color=L_CORPUS)
    arrow(jx + jw * 0.86, y1 + bh1, 6.45, 2.74, lw=1.0, rad=-0.06, color=L_CORPUS)
    ax.text(6.62, 2.58, "retrieval and training corpus", ha="right", va="center",
            fontsize=6.2, style="italic", color=L_CORPUS)

    # ------------------------------------------------------------------ 4 Evaluation
    band(4.72, "4   Evaluation", L_RANK)
    box(0.12, 4.84, W - 0.24, 0.30,
        "Internal test set and five distribution-shifted holdouts "
        "(CCTG, OID ALT, CDASH, GDC, CIMAC)",
        "", C_RANK, L_RANK, title_pt=7.4)

    pdf = OUT / "figure1_workflow_v20.pdf"
    png = OUT / "figure1_workflow_v20.png"
    matplotlib.rcParams["pdf.fonttype"] = 42      # embed TrueType; text stays editable
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    print("wrote", pdf)
    print("wrote", png)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
