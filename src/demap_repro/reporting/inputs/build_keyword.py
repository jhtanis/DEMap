"""Figure 5 inputs: keyword-method selection, full-set recall, and the non-exact subset.

Figure 5 has three panels:

A. Validation Dev comparison of the two keyword methods — the Python approximation
   to NCI CDE Match against CDE Match-Fuzzy — which is how the keyword arm was
   chosen.
B. CDE Match-Fuzzy Recall@5 across the evaluation sets.
C. Recall@5 restricted to queries with **no** available exact-match evidence for
   the gold CDE under the applicable allowance policy. This is the panel that
   shows what each method contributes beyond string equality, and it is where the
   bi-encoder separates sharply from both lexical methods. GDC is omitted: only two
   qualifying queries remain there, too few to plot.

Written fresh against the corrected roots
-----------------------------------------
As with Figure 4, the research repository's ``_build_data2.py`` reads
pre-correction inputs and is retained only as historical provenance. This module
reads the eligibility-corrected artifacts and is parity-tested against the CSVs
that back Figure 5 in v21.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from demap_repro.utils.paths import data_root

__all__ = [
    "PANEL_B_DATASETS", "PANEL_C_DATASETS", "METHOD_LABELS",
    "build_keyword_selection", "build_fullset_recall",
    "build_non_exact_subset", "build_figure5_inputs",
]

CLONE = "python_approximation_to_nci_cde_match"
FUZZY = "cdematch_fuzzy_keyword_v1"
MPNET = "ft_mpnet"

#: artifact method name -> reporting name
METHOD_LABELS = {"clone": CLONE, "keyword_v1_fuzzy": FUZZY, "ft_mpnet": MPNET}

PANEL_B_DATASETS = ("test", "cctg", "oid_alt", "cdash", "gdc_combined", "cimac_v2")

#: GDC is excluded from panel C: only two queries there lack exact evidence.
PANEL_C_DATASETS = ("test", "cctg", "oid_alt", "cdash", "cimac_v2")

#: Allowance in force per dataset (see demap_repro.reranker.split_routing).
ALLOW_RATE = {"test": 0.7, "cctg": 0.7, "oid_alt": 0.7, "cdash": 0.7,
              "gdc_combined": 1.0, "cimac_v2": 1.0, "val_dev": 0.7}


def build_keyword_selection(selection_json: Path) -> pd.DataFrame:
    """Panel A: the Validation Dev head-to-head that selected the keyword arm."""
    sel = json.loads(Path(selection_json).read_text())
    rows: List[Dict] = []
    for key, method in ((("clone_val_dev"), CLONE),
                        (("kwfuzzy_val_dev_corrected"), FUZZY)):
        block = sel[key]
        rows.append({
            "method": method,
            "split": "val_dev",
            "recall_at_5": float(block["recall@5"]),
            "recall_at_1": float(block["recall@1"]),
            "recall_at_10": float(block["recall@10"]),
            "mrr_at_100": float(block["mrr@100"]),
            "n_queries": int(block["n_queries"]),
            "allow_rate": ALLOW_RATE["val_dev"],
            "recall_basis": "versioned_cde_id",
            "variant": "canonical_val_dev_n3934",
            "source_file": str(selection_json),
        })
    return pd.DataFrame(rows)


def selection_margin(selection_json: Path) -> float:
    """Recall@5 margin of CDE Match-Fuzzy over the Python approximation."""
    sel = json.loads(Path(selection_json).read_text())
    return float(sel["margin_recall@5"])


def build_fullset_recall(metrics_csv: Path,
                         datasets=PANEL_B_DATASETS) -> pd.DataFrame:
    """Panel B: CDE Match-Fuzzy Recall@5 on each evaluation set."""
    src = pd.read_csv(metrics_csv).set_index("dataset")
    rows: List[Dict] = []
    for dataset in datasets:
        if dataset not in src.index:
            raise KeyError(f"{metrics_csv}: no row for {dataset!r}")
        r = src.loc[dataset]
        rows.append({
            "method": FUZZY,
            "dataset": dataset,
            "recall_at_5": round(float(r["recall@5"]), 4),
            "n_queries": int(r["n_queries"]),
            "recall_at_5_full_precision": float(r["recall@5"]),
            "recall_basis": "versioned_cde_id",
            "allow_rate": ALLOW_RATE[dataset],
            "source_file": str(metrics_csv),
            "note": "corrected production-eligible keyword index (62,976 CDEs)",
        })
    return pd.DataFrame(rows)


def build_non_exact_subset(metrics_csv: Path,
                           datasets=PANEL_C_DATASETS) -> pd.DataFrame:
    """Panel C: Recall@5 among queries with no allowed exact-match evidence.

    "Non-exact" means the Python approximation's post-policy output contains no
    exact-to-gold candidate — i.e. under the allowance in force, string equality
    was not available for that query.
    """
    src = pd.read_csv(metrics_csv)
    rows: List[Dict] = []
    for dataset in datasets:
        block = src[src["dataset"] == dataset]
        if block.empty:
            raise KeyError(f"{metrics_csv}: no rows for {dataset!r}")
        for _, r in block.iterrows():
            method = METHOD_LABELS.get(r["method"])
            if method is None:
                continue
            rows.append({
                "method": method,
                "dataset": dataset,
                "recall_at_5": float(r["recall@5"]),
                "n_non_exact": int(r["n_non_exact"]),
                "allow_rate": ALLOW_RATE[dataset],
                "coverage": float(r["coverage"]),
                "depth_cap": int(r["depth_cap"]),
                "included_in_panel_c": True,
                "source_file": str(metrics_csv),
            })
    order = {CLONE: 0, FUZZY: 1, MPNET: 2}
    out = pd.DataFrame(rows)
    out["_d"] = out["dataset"].map({d: i for i, d in enumerate(datasets)})
    out["_m"] = out["method"].map(order)
    return out.sort_values(["_d", "_m"]).drop(columns=["_d", "_m"]).reset_index(drop=True)


def build_figure5_inputs(root: Path, out_dir: Optional[Path] = None):
    """All three Figure 5 tables from the corrected artifact roots."""
    audit = root / ".scratch/demap/paper_v13_scientific_audit"
    selection = build_keyword_selection(audit / "keyword_selection_val_dev_v13.json")
    fullset = build_fullset_recall(audit / "kwfuzzy_corrected_metrics.csv")
    nonexact = build_non_exact_subset(
        root / "artifacts/final_reranker/non_exact_subset_eval_v2_eligible"
             / "method_metrics_non_exact.csv")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        selection.to_csv(out_dir / "keyword_selection.csv", index=False)
        fullset.to_csv(out_dir / "keyword_fullset_recall5.csv", index=False)
        nonexact.to_csv(out_dir / "nonexact_subset_recall5.csv", index=False)
    return selection, fullset, nonexact


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(data_root()))
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    sel, full, nonexact = build_figure5_inputs(Path(args.root), Path(args.out_dir))
    print("panel A\n", sel[["method", "recall_at_5"]].to_string(index=False))
    print("\npanel B\n", full[["dataset", "recall_at_5"]].to_string(index=False))
    print("\npanel C\n", nonexact[["dataset", "method", "recall_at_5"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
