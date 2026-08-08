#!/usr/bin/env python3
"""Figure 5 — keyword-method selection and evaluation (three panels).

Editable source, extracted verbatim from
`notebooks/paper_figures/figure_keyword_threepanel_v10.ipynb` so the figure has a
runnable source of record instead of only a notebook.

Nothing is recomputed. The same three validated CSVs are read, and every plotted
value, subset count and axis is unchanged. Exactly two things differ from the
notebook:

  1. The panel C title becomes "Queries without available exact-match evidence
     for the gold CDE". The previous title, "Queries with no exact lexical match
     to the gold CDE", was inaccurate for the four caDSR-derived sets: that
     subset is defined post-policy and is dominated by queries whose exact-match
     strings were withheld by the 70% allowance mask, not by queries whose gold
     CDE shares no identical string.
  2. The embedded footer strip is dropped. Its content ("GDC is deliberately
     omitted from panel C...") is carried by the Word caption instead, so keeping
     it duplicates the caption inside the image.

Outputs vector PDF + 600 dpi PNG + provenance JSON into manuscript/figures/.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/vf/users/nextgen2/james/tasks/cde_project/demap")  # same tree as /data/...
sys.path.insert(0, str(REPO / "notebooks" / "paper_figures"))
import paper_figure_style as S  # noqa: E402

S.FIG_ROOT = REPO / "manuscript" / "figures"

def main(argv=None) -> int:
    """Render Figure 5."""
    S.apply_style()

    import matplotlib as mpl  # noqa: E402
    import matplotlib.pyplot as plt  # noqa: E402

    mpl.rcParams["savefig.dpi"] = 600

    DATA2 = REPO / ".scratch/demap/paper_v10_revision/data2"
    SRC_SEL = DATA2 / "keyword_selection.csv"
    SRC_FULL = DATA2 / "keyword_fullset_recall5.csv"
    SRC_NE = DATA2 / "nonexact_subset_recall5.csv"

    sel_all = pd.read_csv(SRC_SEL)
    full_all = pd.read_csv(SRC_FULL)
    ne_all = pd.read_csv(SRC_NE)

    FIG_W, FIG_H = 9.6, 9.0
    BAR_W_SINGLE = 0.6
    BAR_W_GROUP = 0.26
    VALUE_FMT = "{:.3f}"
    VALUE_PT = S.ANNOTATION_PT
    YLIM = (0.0, 1.06)
    YTICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    YLABEL = "Recall@5"
    SELECTION_VARIANT = "canonical_val_dev_n3934"

    PANEL_C_TITLE = "Queries without available exact-match evidence for the gold CDE"

    METHOD_LABEL_FLAT = {
        "python_approximation_to_nci_cde_match": "Python approximation to NCI CDE Match",
        "cdematch_fuzzy_keyword_v1": "CDE Match-Fuzzy",
        "ft_mpnet": "FT-MPNet",
    }
    METHOD_COLOR = {
        "python_approximation_to_nci_cde_match": S.PALETTE["cdematch_python"],
        "cdematch_fuzzy_keyword_v1": S.PALETTE["keyword"],
        "ft_mpnet": S.PALETTE["FT-MPNet"],
    }
    DATASET_ORDER = ["test", "cctg", "oid_alt", "cdash", "gdc_combined", "cimac_v2"]
    DATASET_LABEL = {"test": "Test", "cctg": "CCTG", "oid_alt": "OID ALT",
                     "cdash": "CDASH", "gdc_combined": "GDC", "cimac_v2": "CIMAC"}
    PANEL_C_METHOD_ORDER = ["python_approximation_to_nci_cde_match", "ft_mpnet",
                            "cdematch_fuzzy_keyword_v1"]

    # ---------------------------------------------------------------- panel A
    panelA = (sel_all[sel_all.variant == SELECTION_VARIANT]
              .sort_values("recall_at_5", ascending=False).reset_index(drop=True))
    assert set(panelA.n_queries) == {3934}
    SELECTED_METHOD = panelA.iloc[0].method

    # ---------------------------------------------------------------- panel B
    assert set(full_all.method) == {SELECTED_METHOD}
    panelB = (full_all.assign(_o=lambda d: d.dataset.map(
                  {d_: i for i, d_ in enumerate(DATASET_ORDER)}))
              .sort_values("_o").reset_index(drop=True))
    assert list(panelB.dataset) == DATASET_ORDER

    # ---------------------------------------------------------------- panel C
    panelC_all = ne_all.copy()
    excluded = sorted(set(panelC_all[~panelC_all.included_in_panel_c].dataset))
    panelC = panelC_all[panelC_all.included_in_panel_c].copy()
    DATASET_ORDER_C = [d for d in DATASET_ORDER if d in set(panelC.dataset)]
    assert set(panelC.method) == set(PANEL_C_METHOD_ORDER)
    nsub = panelC.drop_duplicates("dataset").set_index("dataset")["n_non_exact"].astype(int)

    print("=== values plotted (must match the shipped figure) ===")
    print("panel A:", {METHOD_LABEL_FLAT[r.method]: round(r.recall_at_5, 4)
                       for r in panelA.itertuples()})
    print("panel B:", {DATASET_LABEL[r.dataset]: round(r.recall_at_5_full_precision, 4)
                       for r in panelB.itertuples()})
    print("panel C subset sizes:", {DATASET_LABEL[d]: int(nsub[d]) for d in DATASET_ORDER_C})

    # ---------------------------------------------------------------- draw
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.48, wspace=0.24)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    xa = np.arange(len(panelA), dtype=float)
    for x, r in zip(xa, panelA.itertuples()):
        axA.bar(x, r.recall_at_5, BAR_W_SINGLE, color=METHOD_COLOR[r.method],
                edgecolor="black", linewidth=0.5)
        axA.text(x, r.recall_at_5 + 0.014, VALUE_FMT.format(r.recall_at_5),
                 ha="center", va="bottom", fontsize=VALUE_PT)
    axA.set_xticks(xa)
    axA.set_xticklabels([textwrap.fill(METHOD_LABEL_FLAT[m], 16) for m in panelA.method],
                        fontsize=S.TICK_LABEL_PT)
    axA.set_xlim(-0.6, len(panelA) - 0.4)
    axA.set_ylim(*YLIM); axA.set_yticks(YTICKS)
    axA.set_ylabel("Validation Dev Recall@5", fontsize=S.AXIS_LABEL_PT)
    axA.set_title("Keyword method selection\n(Validation Dev, n = 3934)",
                  fontsize=S.TITLE_PT, pad=10)
    axA.grid(axis="x", visible=False)
    S.panel_letter(axA, "A", dx=-0.16, dy=1.06)

    xb = np.arange(len(panelB), dtype=float)
    for x, r in zip(xb, panelB.itertuples()):
        axB.bar(x, r.recall_at_5_full_precision, BAR_W_SINGLE,
                color=METHOD_COLOR[r.method], edgecolor="black", linewidth=0.5)
        axB.text(x, r.recall_at_5_full_precision + 0.014,
                 VALUE_FMT.format(r.recall_at_5_full_precision),
                 ha="center", va="bottom", fontsize=VALUE_PT)
    axB.set_xticks(xb)
    axB.set_xticklabels([DATASET_LABEL[d] for d in panelB.dataset],
                        fontsize=S.TICK_LABEL_PT, rotation=30, ha="right")
    axB.set_xlim(-0.6, len(panelB) - 0.4)
    axB.set_ylim(*YLIM); axB.set_yticks(YTICKS)
    axB.set_ylabel(YLABEL, fontsize=S.AXIS_LABEL_PT)
    axB.set_xlabel("Evaluation dataset", fontsize=S.AXIS_LABEL_PT, labelpad=6)
    axB.set_title("Full-set Recall@5\n(selected keyword method)",
                  fontsize=S.TITLE_PT, pad=10)
    axB.grid(axis="x", visible=False)
    S.panel_letter(axB, "B", dx=-0.15, dy=1.06)

    xc = np.arange(len(DATASET_ORDER_C), dtype=float)
    pc = panelC.set_index(["method", "dataset"])
    offsets = (np.arange(len(PANEL_C_METHOD_ORDER)) - (len(PANEL_C_METHOD_ORDER) - 1) / 2)
    for j, m in enumerate(PANEL_C_METHOD_ORDER):
        for i, d in enumerate(DATASET_ORDER_C):
            v = float(pc.loc[(m, d), "recall_at_5"])
            x = xc[i] + offsets[j] * BAR_W_GROUP
            axC.bar(x, v, BAR_W_GROUP, color=METHOD_COLOR[m], edgecolor="black",
                    linewidth=0.5, label=METHOD_LABEL_FLAT[m] if i == 0 else None)
            axC.text(x, v + 0.014, VALUE_FMT.format(v), ha="center", va="bottom",
                     fontsize=VALUE_PT - 0.5)
    axC.set_xticks(xc)
    axC.set_xticklabels([f"{DATASET_LABEL[d]}\n({int(nsub[d])} queries)"
                         for d in DATASET_ORDER_C], fontsize=S.TICK_LABEL_PT)
    axC.set_xlim(-0.6, len(DATASET_ORDER_C) - 0.4)
    axC.set_ylim(*YLIM); axC.set_yticks(YTICKS)
    axC.set_ylabel("Subset Recall@5", fontsize=S.AXIS_LABEL_PT)
    axC.set_title(PANEL_C_TITLE, fontsize=S.TITLE_PT, pad=10)
    axC.grid(axis="x", visible=False)
    axC.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3,
               fontsize=S.LEGEND_PT)
    S.panel_letter(axC, "C", dx=-0.075, dy=1.04)

    # No fig.text() footer: its content lives in the Word caption.
    fig.subplots_adjust(bottom=0.10)

    paths = S.save_figure(
        fig, "figure5_keyword_v19", sources=[SRC_SEL, SRC_FULL, SRC_NE],
        notebook="manuscript/figures/make_figure5_keyword.py",
        description=("Figure 5: (A) Validation Dev keyword-method selection; (B) full-set "
                     "Recall@5 of the selected keyword method on the six canonical datasets; "
                     "(C) Recall@5 among queries without available exact-match evidence for "
                     "the gold CDE under the applicable allowance policy, on five datasets "
                     "(GDC omitted: only 2 such queries). Values identical to the v10/v13 "
                     "figure; only the panel C title changed and the embedded footer removed."),
        extra={"selection_variant": SELECTION_VARIANT,
               "panel_c_title": PANEL_C_TITLE,
               "panel_c_excluded_datasets": excluded,
               "recall_basis": "versioned_cde_id",
               "recomputed": False})
    print({k: str(v) for k, v in paths.items()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
