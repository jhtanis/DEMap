#!/usr/bin/env python3
"""Task J: (1) independently recompute the lexical allowance-grid metrics
(Recall@1/5/10, MRR@100; public-id level, every query in the denominator) from
the archived candidate parquets, and compare to the manuscript source CSV
allowance_sensitivity_v13.csv (pinned by the Figure S6 provenance JSON);
(2) verify the SHA-1 seed-42 allowance masks are NESTED across
0.0 < 0.5 < 0.6 < 0.70 < 0.8 < 1.0 on all four eval sets AND on val_train /
val_dev, recording realized admission rates.

Outputs: results/lexical_metrics_recomputed.csv, results/lexical_vs_manuscript.json,
results/mask_nesting.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from demap_repro.utils.paths import data_root

#: Data and artifact tree. This was an absolute path into the research
#: repository, which made the module unusable anywhere else; see
#: ``demap_repro.utils.paths`` and ``DEMAP_DATA_ROOT``.
REPO = data_root()
J = REPO / ".scratch/v17_claude/J_allowance"
from demap_repro.lexical.mask import query_allowed

DATASETS = ["test", "cctg", "oid_alt", "cdash"]
RATES = ["0.0", "0.5", "0.6", "0.70", "0.8", "1.0"]
SRC = {
    "clone": {r: REPO / "artifacts/paper_v13_lexical_sensitivity/clone" for r in RATES},
    "kwfuzzy": {r: REPO / "artifacts/paper_v13_lexical_sensitivity/kwfuzzy" for r in RATES},
}
SRC["clone"]["0.70"] = REPO / "artifacts/final_reranker/cde_match_clone"
SRC["kwfuzzy"]["0.70"] = REPO / "artifacts/final_reranker/keyword_fuzzy_v2_eligible"
LABEL = {"clone": "python_cde_match_approx", "kwfuzzy": "cde_match_fuzzy"}


def pub(c) -> str:
    return str(c).split("::")[0]


def gold_sets(ds: str) -> pd.Series:
    g = pd.read_parquet(REPO / f"data/processed/eval_canonical/{ds}.parquet",
                        columns=["query_id", "cde_id"])
    g["query_id"] = g["query_id"].astype(str)
    g["pub"] = g["cde_id"].map(pub)
    return g.groupby("query_id")["pub"].apply(set)


def metrics(f: Path, golds: pd.Series) -> dict:
    d = pd.read_parquet(f, columns=["query_id", "cde_id", "cdematch_rank"])
    d["query_id"] = d["query_id"].astype(str)
    d["pub"] = d["cde_id"].map(pub)
    d = d.sort_values("cdematch_rank", kind="stable").drop_duplicates(["query_id", "pub"])
    best = {}
    for q, sub in d.groupby("query_id"):
        if q not in golds.index:
            continue
        hit = sub[sub["pub"].isin(golds.loc[q])]
        if len(hit):
            best[q] = float(hit["cdematch_rank"].min())
    n = len(golds)
    out = {"n_queries": n}
    for k in (1, 5, 10):
        out[f"recall@{k}"] = sum(1 for q in golds.index if q in best and best[q] <= k) / n
    out["mrr@100"] = sum(1.0 / best[q] for q in golds.index
                         if q in best and best[q] <= 100) / n
    return out


def main() -> None:
    (J / "results").mkdir(exist_ok=True)

    # ---------- 1. lexical metrics ----------
    rows = []
    for method in ("clone", "kwfuzzy"):
        for rate in RATES:
            for ds in DATASETS:
                f = SRC[method][rate] / f"cde_match_clone_candidates_{ds}__allow{rate}.parquet"
                assert f.exists(), f"missing archived lexical table: {f}"
                rows.append({"method": LABEL[method], "dataset": ds,
                             "allow_rate": float(rate), **metrics(f, gold_sets(ds))})
    t = pd.DataFrame(rows)
    t.to_csv(J / "results/lexical_metrics_recomputed.csv", index=False)

    ref_path = REPO / ".scratch/demap/paper_v13_scientific_audit/allowance_sensitivity_v13.csv"
    ref = pd.read_csv(ref_path)
    sha = hashlib.sha256(ref_path.read_bytes()).hexdigest()
    m = ref.merge(t, on=["method", "dataset", "allow_rate"], suffixes=("_ms", "_re"))
    cols = ["recall@1", "recall@5", "recall@10", "mrr@100"]
    deltas = {c: float((m[f"{c}_ms"] - m[f"{c}_re"]).abs().max()) for c in cols}
    all_match = all(v < 1e-12 for v in deltas.values())
    at070 = t[(t["allow_rate"] == 0.70)].round(6).to_dict(orient="records")
    rep = {"manuscript_csv": str(ref_path), "manuscript_csv_sha256": sha,
           "sha256_matches_figureS6_provenance":
               sha == "33d8fa5805c5634d826279b971365e459fb5be3df0cd5be7f0b27b9a6d7553e5",
           "n_rows_compared": int(len(m)), "max_abs_delta_per_metric": deltas,
           "all_rows_match": bool(all_match), "recomputed_070_operating_point": at070}
    with open(J / "results/lexical_vs_manuscript.json", "w") as fh:
        json.dump(rep, fh, indent=2)
    print(f"[lexical] rows={len(m)} all_match={all_match} deltas={deltas}")

    # ---------- 2. mask nesting ----------
    nest = {"rates": [0.0, 0.5, 0.6, 0.70, 0.8, 1.0], "splits": {}}
    split_files = {ds: REPO / f"data/processed/eval_canonical/{ds}.parquet" for ds in DATASETS}
    split_files["val_train"] = REPO / "data/processed/splits/val_train.parquet"
    split_files["val_dev"] = REPO / "data/processed/splits/val_dev.parquet"
    overall_ok = True
    for name, f in split_files.items():
        qids = (pd.read_parquet(f, columns=["query_id"])["query_id"]
                .astype(str).drop_duplicates().tolist())
        admitted = {r: {q for q in qids if query_allowed(q, r, 42)}
                    for r in nest["rates"]}
        nested = all(admitted[a] <= admitted[b]
                     for a, b in zip(nest["rates"], nest["rates"][1:]))
        strict = all(admitted[a] < admitted[b]
                     for a, b in zip(nest["rates"], nest["rates"][1:]))
        overall_ok &= nested
        nest["splits"][name] = {
            "n_queries": len(qids), "nested": bool(nested),
            "strictly_nested": bool(strict),
            "n_admitted": {str(r): len(admitted[r]) for r in nest["rates"]},
            "realized_rate": {str(r): round(len(admitted[r]) / len(qids), 4)
                              for r in nest["rates"]}}
        print(f"[mask {name}] nested={nested} admitted="
              f"{[len(admitted[r]) for r in nest['rates']]}")
    nest["all_nested"] = bool(overall_ok)
    with open(J / "results/mask_nesting.json", "w") as fh:
        json.dump(nest, fh, indent=2)
    print(f"[mask] all_nested={overall_ok}")


if __name__ == "__main__":
    main()
