#!/usr/bin/env python3
"""Figures S7 and S8 — exact-match allowance sensitivity.

Figure S7 (primary, production-oriented): Recall@5 for the Python approximation
to NCI CDE Match, CDE Match-Fuzzy, and the FIXED HGBC trained at the 70%
allowance, across six inference allowances on the four caDSR-derived sets.

Figure S8 (incremental effect of retraining): the same fixed HGBC against HGBC
models RETRAINED separately at each matching allowance. No lexical methods.

Numbering note. These two figures were Figures S6 and S7 up to manuscript v21.
The v22-v24 revision inserted the candidate-pool coverage table and the broad
evidence-family ablation into the supplement, which pushed them to S7 and S8.

The OUTPUT STEMS below still read `figureS6_...` and `figureS7_...`, and are
deliberately left that way: those stems name the exact PNG/PDF bytes embedded
in the manuscript and recorded in the figure provenance JSON. Renaming them
would break the correspondence between this generator and the files a reader
can check it against. The stem is a historical artifact name; the caption and
the module name carry the current numbering.

Both read the single validated grid
`manuscript/v17_claude_reports/K_fixed070/allowance_sensitivity_four_methods.csv`,
which merges:
  * the fixed-model run  (Slurm array 26716780, Task K), and
  * the allowance-specific retraining run (Slurm 26664824/26664836/26664838/
    26664857/26672434/26672435, Task J).

House style: notebooks/paper_figures/paper_figure_style.py (Okabe-Ito palette,
panel letters, vector PDF + 600 dpi PNG + provenance JSON).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from demap_repro.utils.paths import data_root

#: Data and artifact tree. This was an absolute path into the research
#: repository, which made the module unusable anywhere else; see
#: ``demap_repro.utils.paths`` and ``DEMAP_DATA_ROOT``.
REPO = data_root()
K = REPO / "manuscript/v17_claude_reports/K_fixed070"
from demap_repro.reporting.figures import style as pfs

# ``notebooks/paper_figures/paper_figure_style.py`` in the research repository;
# migrated here symbol-for-symbol. ``style.FIG_ROOT`` already honours
# DEMAP_FIGURE_DIR, so the output location no longer needs overriding.

def main(argv=None) -> int:
    """Render Figures S6 and S7."""
    pfs.apply_style()

    import matplotlib as mpl  # noqa: E402
    import matplotlib.pyplot as plt  # noqa: E402

    mpl.rcParams["savefig.dpi"] = 600

    LINE_W = 2.2
    MS = 7
    C_FUZZY = pfs.PALETTE["keyword"]
    C_CLONE = pfs.PALETTE["cdematch_python"]
    C_FIXED = pfs.PALETTE["reranker"]          # black
    C_RETRAIN = pfs.PALETTE["crossencoder"]    # light blue, clearly distinct

    CSV = K / "allowance_sensitivity_four_methods.csv"
    t = pd.read_csv(CSV)
    t = t[t["tier"] == "primary"].copy()

    DATASETS = [("test", "Test"), ("cctg", "CCTG"),
                ("oid_alt", "OID ALT"), ("cdash", "CDASH")]
    LETTERS = "ABCD"
    RATES = [0.0, 0.5, 0.6, 0.7, 0.8, 1.0]

    # label wording must not be a bare "HGBC" — it would be ambiguous between the
    # fixed and the retrained analyses.
    LBL_FIXED = "Fixed HGBC trained at 70%"
    LBL_RETRAIN = "HGBC retrained at each allowance"

    assert sorted(t["allow_rate"].round(2).unique()) == RATES, "unexpected allowance grid"


    def build(series, name, description):
        fig, axes = plt.subplots(2, 2, figsize=(pfs.WIDTH_FULL, 6.4), sharex=True)
        for ax, (ds, label), letter in zip(axes.ravel(), DATASETS, LETTERS):
            for method, color, nm, marker, ls in series:
                d = (t[(t["method"] == method) & (t["dataset"] == ds)]
                     .sort_values("allow_rate"))
                assert len(d) == 6, f"{method}/{ds}: expected 6 rates, got {len(d)}"
                ax.plot(d["allow_rate"] * 100, d["recall@5"], marker=marker, ms=MS,
                        lw=LINE_W, ls=ls, color=color, label=nm,
                        markeredgecolor="white", markeredgewidth=1.2)
            ax.set_ylim(0.0, 1.02)
            ax.axvline(70, color="0.45", lw=1.4, ls="--", zorder=0)
            ax.annotate("70% primary", xy=(70, 0.0), xytext=(3, 6),
                        textcoords="offset points", fontsize=pfs.ANNOTATION_PT,
                        color="0.30")
            ax.set_title(label, fontsize=pfs.TITLE_PT)
            pfs.panel_letter(ax, letter)
            ax.set_xticks([0, 50, 60, 70, 80, 100])
            ax.tick_params(labelsize=pfs.TICK_LABEL_PT)
            ax.grid(True, axis="y", color="0.9", lw=0.8, zorder=0)
            ax.set_axisbelow(True)
        for ax in axes[1]:
            ax.set_xlabel("Exact-match allowance (%)", fontsize=pfs.AXIS_LABEL_PT)
        for ax in axes[:, 0]:
            ax.set_ylabel("Recall@5", fontsize=pfs.AXIS_LABEL_PT)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=1,
                   fontsize=pfs.LEGEND_PT, frameon=False,
                   bbox_to_anchor=(0.5, -0.02 - 0.030 * len(series)))
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        # The recorded producer path is the historical one, so the provenance
        # JSON keeps matching the figures already embedded in the manuscript.
        paths = pfs.save_figure(fig, name, sources=[CSV],
                                notebook="manuscript/figures/make_figureS6_S7_allowance.py",
                                description=description)
        plt.close(fig)
        print(name, "->", paths["pdf"])


    build(
        [("hgbc_fixed_070", C_FIXED, LBL_FIXED, "s", "-"),
         ("cde_match_fuzzy", C_FUZZY, "CDE Match-Fuzzy", "o", "-"),
         ("python_cde_match_approx", C_CLONE,
          "Python approximation to NCI CDE Match", "o", "-")],
        "figureS6_allowance_fixed_hgbc_v19",          # manuscript Figure S7
        "Figure S7: Recall@5 vs exact-match allowance on the four caDSR-derived "
        "evaluation sets, for the Python approximation to NCI CDE Match, CDE "
        "Match-Fuzzy, and the unchanged HGBC trained at the 70% allowance. At each "
        "inference allowance the CDE Match-Fuzzy candidates, merged pools and "
        "allowance-dependent features were regenerated; FT-MPNet, FT-MedCPT and the "
        "HGBC weights were held fixed.")

    build(
        [("hgbc_fixed_070", C_FIXED, LBL_FIXED, "s", "-"),
         ("hgbc_retrained_per_rate", C_RETRAIN, LBL_RETRAIN, "^", "--")],
        "figureS7_allowance_fixed_vs_retrained_v19",  # manuscript Figure S8
        "Figure S8: Recall@5 for the fixed HGBC trained at the 70% allowance versus "
        "HGBC models retrained separately at each allowance, on the four "
        "caDSR-derived evaluation sets. Candidate pools and allowance-dependent "
        "features were generated at the indicated allowance for both analyses; "
        "FT-MPNet and FT-MedCPT were held fixed.")

    # ---- echo the exact plotted numbers so they can be checked against the CSV ----
    print("\n=== plotted Recall@5 values ===")
    for meth in ["python_cde_match_approx", "cde_match_fuzzy",
                 "hgbc_fixed_070", "hgbc_retrained_per_rate"]:
        print(f"\n{meth}")
        for ds, lab in DATASETS:
            d = t[(t.method == meth) & (t.dataset == ds)].sort_values("allow_rate")
            print(f"  {lab:>8}: " + "  ".join(f"{v:.4f}" for v in d["recall@5"]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
