#!/usr/bin/env python3
"""Assemble the corrected Table 4 (final Recall@5) for manuscript v13.

Columns: Final reranker (corrected HGBC), FT-MPNet (unchanged), CDE Match-Fuzzy
(corrected), Python approximation (unchanged), BM25 (unchanged), FT-<CE winner>
(top-30 pool reranking, corrected).

Every value is loaded from a machine-readable artifact; fails fast if any input
is missing. Also writes the old-vs-corrected comparison for the handoff.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path("/vf/users/nextgen2/james/tasks/cde_project/demap")
OUT = REPO / ".scratch/demap/paper_v13_scientific_audit"
DATASETS = ["test", "cctg", "oid_alt", "cdash", "gdc_combined", "cimac_v2"]
DISPLAY = {"test": "Test", "cctg": "CCTG", "oid_alt": "OID ALT", "cdash": "CDASH",
           "gdc_combined": "GDC", "cimac_v2": "CIMAC"}

need = {
    "hgbc": REPO / "artifacts/final_reranker/hgbc_reranker_v2_eligible/with_ce_noprov/hgbc_eval_by_split.csv",
    "winner": REPO / "artifacts/final_reranker/crossencoder_fulltrain_v2_eligible/CE_WINNER.json",
    "kwfuzzy": OUT / "kwfuzzy_corrected_metrics.csv",
    "old_final": REPO / "artifacts/final_reranker/final_report/FINAL_METHOD_COMPARISON.csv",
    "clone": REPO / "artifacts/final_reranker/cde_match_clone/clone_canonical_metrics_combined.csv",
    "bm25": REPO / "artifacts/bm25_canonical_v1/bm25_canonical_metrics.csv",
}
def main(argv=None) -> int:
    """Assemble Table 4 from the per-method artifacts and write final_table4_v13.csv."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=str(REPO))
    ap.add_argument('--out-dir', default=str(OUT))
    args = ap.parse_args(argv)
    globals()['REPO'] = Path(args.root)
    globals()['OUT'] = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in need.values() if not p.is_file()]
    if missing:
        sys.exit("ERROR: missing required inputs:\n  " + "\n  ".join(missing))

    winner = json.loads(need["winner"].read_text())
    wtag = winner["winner_tag"]
    ce_eval = REPO / ("artifacts/final_reranker/hgbc_features_v2_eligible/ce_alone_fixedk30_eval/"
                      f"crossenc_eval_by_split_SN_DEC_DEF_PQT_PV_{wtag}_fulltrain_fixedk30.csv")
    if not ce_eval.is_file():
        sys.exit(f"ERROR: winner eval CSV missing: {ce_eval}")

    hgbc = pd.read_csv(need["hgbc"])
    hgbc = hgbc[hgbc["method"] == "hgbc"].set_index("split")
    ce = pd.read_csv(ce_eval)
    ce = ce[ce["method"] == "crossencoder"].set_index("split")
    kw = pd.read_csv(need["kwfuzzy"]).set_index("dataset")
    old = pd.read_csv(need["old_final"])
    clone = pd.read_csv(need["clone"]).set_index("dataset")
    bm25 = pd.read_csv(need["bm25"])
    b_col = "R5" if "R5" in bm25.columns else "recall@5"
    bm25 = bm25.set_index("dataset")

    WLABEL = {"ncbi_MedCPT-Cross-Encoder": "FT-MedCPT",
              "BAAI_bge-reranker-base": "FT-BGE",
              "cross-encoder_ms-marco-MiniLM-L-6-v2": "FT-MiniLM"}[wtag]


    def old_val(method, ds):
        m = old[(old["method"] == method) & (old["dataset"] == ds)]
        if len(m) == 0:
            return None
        return float(m.iloc[0]["recall@5"]) if "recall@5" in m.columns else float(m.iloc[0]["R5"])


    rows = []
    for ds in DATASETS:
        vals = {
            "final_reranker": float(hgbc.loc[ds, "recall@5"]),
            "ce_pool_rerank": float(ce.loc[ds, "recall@5"]),
            "ft_mpnet": old_val("ft_mpnet", ds),
            "cde_match_fuzzy": float(kw.loc[ds, "recall@5"]),
            "python_cde_match_approx": float(clone.loc[ds, "R5"]),
            "bm25": float(bm25.loc[ds, b_col]),
        }
        for method, v in vals.items():
            rows.append({"dataset": DISPLAY[ds], "method": method, "recall@5": v})
        # old-vs-corrected companion rows (extra methods; builder ignores unknown methods)
        rows.append({"dataset": DISPLAY[ds], "method": "old_final_reranker",
                     "recall@5": old_val("hgbc_with_ce_FINAL", ds)})
        rows.append({"dataset": DISPLAY[ds], "method": "old_ce_pool_rerank",
                     "recall@5": old_val("medcpt_ce_alone", ds)})
        rows.append({"dataset": DISPLAY[ds], "method": "old_cde_match_fuzzy",
                     "recall@5": float(kw.loc[ds, "old_recall@5"])})

    t = pd.DataFrame(rows)
    t["ce_winner_tag"] = wtag
    t["ce_winner_label"] = WLABEL
    t.to_csv(OUT / "final_table4_v13.csv", index=False)
    print(t.pivot_table(index="dataset", columns="method", values="recall@5").round(4).to_string())
    print("\nwrote", OUT / "final_table4_v13.csv")
    return 0


if __name__ == '__main__':
    sys.exit(main())
