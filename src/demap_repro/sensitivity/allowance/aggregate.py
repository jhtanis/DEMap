#!/usr/bin/env python3
"""Task K stage 2: aggregate the fixed-0.70 results and join them to the
existing Task J grid (Python approximation, CDE Match-Fuzzy, retrained HGBC).

Emits a single tidy CSV carrying all FOUR methods at every allowance rate, plus
pool statistics and the fixed-vs-retrained delta.
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
K = REPO / ".scratch/v17_claude/K_fixed070"
J_CSV = REPO / "manuscript/v17_claude_reports/J_allowance/allowance_sensitivity_with_hgbc.csv"
OUT = REPO / "manuscript/v17_claude_reports/K_fixed070"

RATES = ["0.0", "0.5", "0.6", "0.70", "0.8", "1.0"]
PRIMARY = ["test", "cctg", "oid_alt", "cdash"]
METRICS = ["recall@1", "recall@5", "recall@10", "mrr@100"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- fixed-model rows --------------------------------------------------
    fixed_rows, pool_rows = [], []
    for r in RATES:
        j = json.load(open(K / f"rates/a{r}/metrics_fixed070_a{r}.json"))
        for split, m in j["splits"].items():
            fixed_rows.append({
                "allow_rate": float(r), "dataset": split,
                "method": "hgbc_fixed_070", "tier": m["tier"],
                **{k: m[k] for k in METRICS},
                "n_queries": m["n_queries"],
            })
        for split, p in j["pool_stats"].items():
            pool_rows.append({"allow_rate": float(r), "dataset": split, **p})
    fixed = pd.DataFrame(fixed_rows)
    pools = pd.DataFrame(pool_rows)

    # ---- Task J rows (lexical arms + retrained HGBC) ------------------------
    jdf = pd.read_csv(J_CSV)
    jdf = jdf.rename(columns={"method": "method_raw"})
    jdf["method"] = jdf["method_raw"].replace(
        {"hgbc_reranker_allowance_specific": "hgbc_retrained_per_rate"})
    jdf["tier"] = jdf["dataset"].map(
        lambda d: "primary" if d in PRIMARY else "secondary")

    keep = ["allow_rate", "dataset", "method", "tier"] + METRICS
    combined = pd.concat(
        [jdf[[c for c in keep if c in jdf.columns]], fixed[keep]],
        ignore_index=True)
    combined = combined.sort_values(["dataset", "method", "allow_rate"])
    combined.to_csv(OUT / "allowance_sensitivity_four_methods.csv", index=False)
    pools.to_csv(OUT / "pool_stats_fixed070.csv", index=False)

    # ---- fixed vs retrained delta -----------------------------------------
    piv = (combined[combined.tier == "primary"]
           .pivot_table(index=["dataset", "allow_rate"], columns="method",
                        values=METRICS))
    rows = []
    for (ds, rate), row in piv.iterrows():
        rec = {"dataset": ds, "allow_rate": rate}
        for m in METRICS:
            f = row.get((m, "hgbc_fixed_070"))
            t = row.get((m, "hgbc_retrained_per_rate"))
            rec[f"{m}_fixed"] = f
            rec[f"{m}_retrained"] = t
            rec[f"{m}_fixed_minus_retrained"] = (
                None if pd.isna(f) or pd.isna(t) else f - t)
        rows.append(rec)
    delta = pd.DataFrame(rows).sort_values(["dataset", "allow_rate"])
    delta.to_csv(OUT / "fixed_vs_retrained_delta.csv", index=False)

    # ---- console summary ---------------------------------------------------
    print("=== Recall@5, four methods, primary datasets ===")
    for ds in PRIMARY:
        print(f"\n{ds}")
        sub = combined[(combined.dataset == ds)]
        for meth in ["python_cde_match_approx", "cde_match_fuzzy",
                     "hgbc_fixed_070", "hgbc_retrained_per_rate"]:
            s = sub[sub.method == meth].sort_values("allow_rate")
            if s.empty:
                continue
            vals = "  ".join(f"{v:.4f}" for v in s["recall@5"])
            span = s["recall@5"].iloc[-1] - s["recall@5"].iloc[0]
            print(f"  {meth:>26s}: {vals}   span={span:+.3f}")

    print("\n=== fixed minus retrained (Recall@5) ===")
    for ds in PRIMARY:
        s = delta[delta.dataset == ds].sort_values("allow_rate")
        vals = "  ".join(f"{v:+.4f}" for v in s["recall@5_fixed_minus_retrained"])
        print(f"  {ds:>8s}: {vals}")

    print(f"\nwrote {OUT/'allowance_sensitivity_four_methods.csv'}")
    print(f"wrote {OUT/'pool_stats_fixed070.csv'}")
    print(f"wrote {OUT/'fixed_vs_retrained_delta.csv'}")


if __name__ == "__main__":
    main()
