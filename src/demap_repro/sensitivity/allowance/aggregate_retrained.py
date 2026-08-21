#!/usr/bin/env python3
"""Task J stage 4: aggregate the full allowance grid.

Combines the (verified) lexical grid with the allowance-specific HGBC results
into machine-readable deliverables:

  results/allowance_sensitivity_with_hgbc.csv   -- 3 methods x 4 datasets x 6 rates
  results/pool_stats_by_rate.csv                -- ceiling + pool-size stats
  results/hgbc_training_summary.csv             -- selected config + val_dev R@5 per rate
  tables/table_S6_proposed.csv / .md            -- proposed updated Table S6
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from demap_repro.utils.paths import data_root

#: Data and artifact tree. This was an absolute path into the research
#: repository, which made the module unusable anywhere else; see
#: ``demap_repro.utils.paths`` and ``DEMAP_DATA_ROOT``.
REPO = data_root()
J = REPO / ".scratch/v17_claude/J_allowance"
RATES = ["0.0", "0.5", "0.6", "0.70", "0.8", "1.0"]
DATASETS = ["test", "cctg", "oid_alt", "cdash"]
HGBC_LABEL = "hgbc_reranker_allowance_specific"


def to_md(df: pd.DataFrame) -> str:
    """Minimal GitHub-flavoured markdown table (avoids the optional 'tabulate' dep)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main() -> None:
    (J / "tables").mkdir(exist_ok=True)

    lex = pd.read_csv(J / "results/lexical_metrics_recomputed.csv")

    hgbc_rows, pool_rows, train_rows = [], [], []
    missing = []
    for r in RATES:
        rd = J / f"rates/a{r}"
        mf = rd / f"metrics_hgbc_a{r}.json"
        pf = rd / "pool_stats.json"
        if not mf.exists() or not pf.exists():
            missing.append(r)
            continue
        m = json.loads(mf.read_text())
        # PRIMARY HGBC numbers = the trainer's own eval CSV (identical convention
        # to the manuscript's final-reranker numbers). post_metrics deploy-style
        # tie-break values remain in metrics_hgbc_a{r}.json as a cross-check.
        ev = pd.read_csv(rd / "hgbc_noprov/hgbc_eval_by_split.csv")
        for ds in DATASETS:
            row = ev[ev["split"] == ds].iloc[0]
            hgbc_rows.append({
                "method": HGBC_LABEL, "dataset": ds, "allow_rate": float(r),
                "n_queries": int(row["n_queries"]), "recall@1": float(row["recall@1"]),
                "recall@5": float(row["recall@5"]), "recall@10": float(row["recall@10"]),
                "mrr@100": float(row["mrr@100"])})
        p = json.loads(pf.read_text())
        for ds, s in p["splits"].items():
            pool_rows.append({"allow_rate": float(r), "dataset": ds, **s})
        cfg = json.loads((rd / "hgbc_noprov/selected_hgbc_config.json").read_text())
        grid = pd.read_csv(rd / "hgbc_noprov/hgbc_grid_results.csv")
        vd = m["splits"].get("val_dev", {})
        train_rows.append({"allow_rate": float(r), **{f"selected_{k}": v for k, v in cfg.items()},
                           "grid_best_val_dev_r5": float(grid["tune_recall@5"].max())
                           if "tune_recall@5" in grid.columns else None,
                           "val_dev_recall@5": vd.get("recall@5"),
                           "val_train_n_queries": m["splits"].get("val_train", {}).get("n_queries")})

    if missing:
        print(f"NOTE: rates not yet complete: {missing}")

    comb = pd.concat([lex, pd.DataFrame(hgbc_rows)], ignore_index=True)
    comb = comb.sort_values(["method", "dataset", "allow_rate"])
    comb.to_csv(J / "results/allowance_sensitivity_with_hgbc.csv", index=False)
    pd.DataFrame(pool_rows).sort_values(["dataset", "allow_rate"]).to_csv(
        J / "results/pool_stats_by_rate.csv", index=False)
    pd.DataFrame(train_rows).to_csv(J / "results/hgbc_training_summary.csv", index=False)

    # ---- proposed Table S6 ----
    name_map = {"python_cde_match_approx": "Python approximation to NCI CDE Match",
                "cde_match_fuzzy": "CDE Match-Fuzzy",
                HGBC_LABEL: "Final reranker (HGBC, retrained per allowance)"}
    ds_map = {"test": "Test", "cctg": "CCTG", "oid_alt": "OID ALT", "cdash": "CDASH"}
    tab = comb.copy()
    tab["Method"] = tab["method"].map(name_map)
    tab["Dataset"] = tab["dataset"].map(ds_map)
    tab["Allowance (%)"] = (tab["allow_rate"] * 100).round().astype(int)
    for c in ("recall@1", "recall@5", "recall@10", "mrr@100"):
        tab[c] = tab[c].round(3)
    tab = tab[["Dataset", "Allowance (%)", "Method",
               "recall@1", "recall@5", "recall@10", "mrr@100"]].rename(
        columns={"recall@1": "Recall@1", "recall@5": "Recall@5",
                 "recall@10": "Recall@10", "mrr@100": "MRR@100"})
    tab = tab.sort_values(["Dataset", "Allowance (%)", "Method"])
    tab.to_csv(J / "tables/table_S6_proposed.csv", index=False)
    with open(J / "tables/table_S6_proposed.md", "w") as fh:
        fh.write("# Proposed updated Table S6 - exact-match allowance sensitivity\n\n"
                 "All three methods, four caDSR-derived evaluation sets, allowance "
                 "rates 0/50/60/70/80/100% (shared nested SHA-1 seed-42 query mask).\n"
                 "The HGBC row at each rate is a reranker retrained from scratch on "
                 "candidates/features generated at that rate (train=val_train, "
                 "selection=val_dev only, frozen 16-config grid, seed 42).\n\n")
        fh.write(to_md(tab))
        fh.write("\n")
    print(f"wrote {J/'results/allowance_sensitivity_with_hgbc.csv'} ({len(comb)} rows)")
    print(f"wrote {J/'tables/table_S6_proposed.csv'}")


if __name__ == "__main__":
    main()
