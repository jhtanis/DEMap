#!/usr/bin/env python3
"""Assemble Table 4 — the six-method Recall@5 comparison.

Columns: Final reranker (HGBC), FT-MPNet, CDE Match-Fuzzy, the Python
approximation to NCI CDE Match, BM25, and FT-<CE winner> reranking the same
top-30 pool.

Every value is loaded from a machine-readable artifact; fails fast if any input
is missing. Also writes the old-vs-corrected comparison rows.

Precision
---------
Three methods used to be read from artifacts that store recall **already
rounded to 4 dp**. The manuscript prints Table 4 at 3 dp, so those cells were
rounded twice, and two OID ALT values sit exactly on a 4-dp boundary whose
float representation falls marginally below it:

    Python approximation  0.7695 -> 0.769   but  0.7695356738... -> 0.770
    BM25                  0.1755 -> 0.175   but  0.1755379388... -> 0.176

Table S7 already printed 0.770 and 0.176 for the same two quantities, so the
manuscript disagreed with itself until this was corrected. ``ft_mpnet`` came
from a 4-dp CSV too; it produced no wrong cell (CDASH 0.8395 rounds to 0.840
either way) but carried the same latent risk, so it is repointed as well.

All three now read full-precision sources. Two gates keep it that way: every
repointed value must equal its legacy 4-dp artifact when rounded to 4 dp
(proving only precision changed, not any result), and the 3-dp rendering must
equal the manuscript in all 36 cells.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from demap_repro.utils.paths import data_root

#: Data and artifact tree. This was an absolute path into the research
#: repository, which made the module unusable anywhere else; see
#: ``demap_repro.utils.paths`` and ``DEMAP_DATA_ROOT``.
REPO = data_root()
OUT = REPO / ".scratch/demap/paper_v13_scientific_audit"
DATASETS = ["test", "cctg", "oid_alt", "cdash", "gdc_combined", "cimac_v2"]
DISPLAY = {"test": "Test", "cctg": "CCTG", "oid_alt": "OID ALT", "cdash": "CDASH",
           "gdc_combined": "GDC", "cimac_v2": "CIMAC"}

need = {
    "hgbc": REPO / "artifacts/final_reranker/hgbc_reranker_v2_eligible/with_ce_noprov/hgbc_eval_by_split.csv",
    "winner": REPO / "artifacts/final_reranker/crossencoder_fulltrain_v2_eligible/CE_WINNER.json",
    "kwfuzzy": OUT / "kwfuzzy_corrected_metrics.csv",
    "old_final": REPO / "artifacts/final_reranker/final_report/FINAL_METHOD_COMPARISON.csv",
    # Full-precision display sources.
    "s7": OUT / "table_s7_v13_with_hgbc.csv",
    "ft_mpnet": REPO / ("artifacts/evaluation/phase2_canonical_v1_winners_canonical_eval/"
                        "phase2__all-mpnet-base-v2__hard_top25__seed1/canonical_metrics.csv"),
    # Retained ONLY as 4-dp cross-checks; no longer the display source.
    "clone_4dp": REPO / "artifacts/final_reranker/cde_match_clone/clone_canonical_metrics_combined.csv",
    "bm25_4dp": REPO / "artifacts/bm25_canonical_v1/bm25_canonical_metrics.csv",
}

#: Table 4 as the manuscript prints it, at 3 dp. The regression gate below: a
#: rebuild that no longer renders these values is a defect, not a new result.
MANUSCRIPT_TABLE4 = {
    "Test":    {"final_reranker": "0.971", "python_cde_match_approx": "0.719", "bm25": "0.361",
                "ft_mpnet": "0.912", "cde_match_fuzzy": "0.773", "ce_pool_rerank": "0.914"},
    "CCTG":    {"final_reranker": "0.909", "python_cde_match_approx": "0.739", "bm25": "0.392",
                "ft_mpnet": "0.705", "cde_match_fuzzy": "0.780", "ce_pool_rerank": "0.718"},
    "OID ALT": {"final_reranker": "0.832", "python_cde_match_approx": "0.770", "bm25": "0.176",
                "ft_mpnet": "0.428", "cde_match_fuzzy": "0.702", "ce_pool_rerank": "0.564"},
    "CDASH":   {"final_reranker": "0.920", "python_cde_match_approx": "0.750", "bm25": "0.710",
                "ft_mpnet": "0.840", "cde_match_fuzzy": "0.830", "ce_pool_rerank": "0.864"},
    "GDC":     {"final_reranker": "0.972", "python_cde_match_approx": "0.986", "bm25": "0.542",
                "ft_mpnet": "0.833", "cde_match_fuzzy": "0.972", "ce_pool_rerank": "0.903"},
    "CIMAC":   {"final_reranker": "0.802", "python_cde_match_approx": "0.679", "bm25": "0.160",
                "ft_mpnet": "0.565", "cde_match_fuzzy": "0.786", "ce_pool_rerank": "0.595"},
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

    # Full-precision display sources. eval_set v1 is the unfiltered canonical
    # evaluation Table 4 reports; v2 is the leakage-filtered one (Table S6).
    s7 = pd.read_csv(need["s7"])
    s7 = s7[s7["eval_set"] == "v1"]
    s7_clone = s7[s7["method"] == "python_cde_match_approx"].set_index("dataset")
    s7_bm25 = s7[s7["method"] == "bm25"].set_index("dataset")
    mpnet = pd.read_csv(need["ft_mpnet"]).set_index("dataset")

    # Legacy 4-dp artifacts, kept only to prove nothing scientific changed.
    clone4 = pd.read_csv(need["clone_4dp"]).set_index("dataset")
    bm254 = pd.read_csv(need["bm25_4dp"])
    b_col = "R5" if "R5" in bm254.columns else "recall@5"
    bm254 = bm254.set_index("dataset")

    WLABEL = {"ncbi_MedCPT-Cross-Encoder": "FT-MedCPT",
              "BAAI_bge-reranker-base": "FT-BGE",
              "cross-encoder_ms-marco-MiniLM-L-6-v2": "FT-MiniLM"}[wtag]


    def old_val(method, ds):
        m = old[(old["method"] == method) & (old["dataset"] == ds)]
        if len(m) == 0:
            return None
        return float(m.iloc[0]["recall@5"]) if "recall@5" in m.columns else float(m.iloc[0]["R5"])


    rows = []
    precision_checks = []
    for ds in DATASETS:
        vals = {
            "final_reranker": float(hgbc.loc[ds, "recall@5"]),
            "ce_pool_rerank": float(ce.loc[ds, "recall@5"]),
            "ft_mpnet": float(mpnet.loc[ds, "recall@5"]),
            "cde_match_fuzzy": float(kw.loc[ds, "recall@5"]),
            "python_cde_match_approx": float(s7_clone.loc[ds, "recall@5"]),
            "bm25": float(s7_bm25.loc[ds, "recall@5"]),
        }
        # Nothing scientific may change: each repointed full-precision value must
        # equal its legacy 4-dp artifact when rounded to 4 dp.
        for meth, legacy in (("python_cde_match_approx", float(clone4.loc[ds, "R5"])),
                             ("bm25", float(bm254.loc[ds, b_col])),
                             ("ft_mpnet", old_val("ft_mpnet", ds))):
            got = round(vals[meth], 4)
            if abs(got - legacy) > 1e-9:
                sys.exit(f"ERROR: {ds}/{meth} full precision {vals[meth]!r} rounds to "
                         f"{got} but the legacy 4-dp artifact says {legacy}")
            precision_checks.append((ds, meth))
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

    # Regression gate: the 3-dp rendering must equal the manuscript in all 36
    # cells, so a rebuild cannot silently reintroduce 0.769 / 0.175.
    bad = []
    for ds_disp, expected in MANUSCRIPT_TABLE4.items():
        for meth, want in expected.items():
            v = t[(t.dataset == ds_disp) & (t.method == meth)].iloc[0]["recall@5"]
            got = f"{float(v):.3f}"
            if got != want:
                bad.append(f"  {ds_disp}/{meth}: builder {got} != manuscript {want} (raw {v!r})")
    if bad:
        sys.exit("ERROR: Table 4 no longer reproduces the manuscript:\n" + "\n".join(bad))

    t.to_csv(OUT / "final_table4_v13.csv", index=False)

    # Singly-rounded display strings, so no consumer needs to round at all.
    disp = t[t.method.isin(MANUSCRIPT_TABLE4["Test"])].copy()
    disp["display_3dp"] = disp["recall@5"].map(lambda v: f"{float(v):.3f}")
    disp[["dataset", "method", "recall@5", "display_3dp"]].to_csv(
        OUT / "final_table4_v13_display3dp.csv", index=False)

    print(t.pivot_table(index="dataset", columns="method", values="recall@5").round(4).to_string())
    print(f"\nfull-precision vs legacy 4-dp cross-checks: {len(precision_checks)} passed")
    print(f"manuscript Table 4 regression gate: {6 * 6} cells passed")
    print("wrote", OUT / "final_table4_v13.csv")
    print("wrote", OUT / "final_table4_v13_display3dp.csv")
    return 0


if __name__ == '__main__':
    sys.exit(main())
