#!/usr/bin/env python3
"""Step E-fulltrain, Stage C: build the CE candidate pool over the FULL train
split, PLUS val_dev, with IDENTICAL semantics to the accepted Step E pool
(se20 ∪ kwfuzzy10, NO clone, NO clone_or_fuzzy — benchmark independence).

Differences vs .scratch/demap/final_reranker_stepE/build_ce_pool_se20_kwfuzzy10.py:
  * DATASETS = [train (per_query_one gold injection), val_dev (deployable only)].
    train is the CE-training universe; val_dev is the dev/selection universe.
    The 6 canonical EVAL sets are NOT rebuilt here — CE eval reuses the existing
    Step E eval pool (artifacts/final_reranker/crossencoder/candidate_pool_...).
  * SE source for train  = artifacts/final_reranker/biencoder_deep/train/... (Stage A)
    SE source for val_dev = existing artifacts/final_reranker/biencoder_deep/splits/...
  * KW source for train  = artifacts/final_reranker/keyword_fuzzy_train/... (Stage B)
    KW source for val_dev = existing artifacts/final_reranker/keyword_fuzzy/...
  * NO Step-D grid cross-check for train (train is not in the K-selection grid).
    val_dev IS cross-checked against the existing eval pool's val_dev gold_in_pool
    (must equal 3890) as a regression gate.

Output schema is the SAME contract consumed by build_ce_training_pairs.py.
NEVER reads clone candidates; never touches artifacts_v3_cdisc (asserted).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from demap_repro.crossencoder.eligibility import (
    drop_train_queries_without_obtainable_positive,
)

# Root holding the artifact / data trees. Override with DEMAP_DATA_ROOT.
REPO = Path(os.environ.get("DEMAP_DATA_ROOT", ".")).resolve()

SE_TRAIN = REPO / "artifacts/final_reranker/biencoder_deep/train/biencoder_deep_rankings_top1000.parquet"
SE_VALDEV = REPO / "artifacts/final_reranker/biencoder_deep/splits/biencoder_deep_rankings_top1000.parquet"
KW_TRAIN = REPO / "artifacts/final_reranker/keyword_fuzzy_train_v2_eligible/cde_match_clone_candidates_train__allow0.70.parquet"
KW_VALDEV = REPO / "artifacts/final_reranker/keyword_fuzzy_v2_eligible/cde_match_clone_candidates_val_dev__allow0.70.parquet"
EXISTING_EVAL_POOL_SUMMARY = REPO / "artifacts/final_reranker/crossencoder_v2_eligible/candidate_pool_se20_kwfuzzy10/pool_summary.json"

CATALOG = REPO / ("data/processed/cadsr_xml_2026-06-18/"
                  "cde_master_enriched_eval_production_cde_match.parquet")
CATALOG_FULL = REPO / ("data/processed/cadsr_xml_2026-06-18/"
                       "cde_master_enriched_eval.parquet")
SPLIT_DIR = REPO / "data/processed/splits"

DATASETS = ["train", "val_dev"]
SE_FILE = {"train": SE_TRAIN, "val_dev": SE_VALDEV}
KW_FILE = {"train": KW_TRAIN, "val_dev": KW_VALDEV}
INJECT_SPLITS = {"train"}   # per_query_one injection, CE-training split only
SE_K, KW_K = 20, 10
WINNER_ID = "phase2_allmpnet_rep_6f14e0fbed"

OUT_COLS = ["winner_id", "split", "query_id", "query_text_q3", "cde_id",
            "public_id", "is_label", "is_injected_gold",
            "biencoder_rank", "biencoder_score",
            "cdematch_rank", "cdematch_score", "cand_source",
            "family", "query_source", "in_biencoder_topk", "in_cdematch_topk"]


def pub(c) -> str:
    return str(c).split("::")[0]


def dedup_best(df: pd.DataFrame, rank_col: str) -> pd.DataFrame:
    df = df.dropna(subset=[rank_col]).copy()
    df["query_id"] = df["query_id"].astype(str)
    df["public_id"] = df["cde_id"].map(pub)
    return (df.sort_values(rank_col, kind="stable")
              .drop_duplicates(["query_id", "public_id"]))


def load_split_meta(ds: str):
    import pyarrow.parquet as pq
    f = SPLIT_DIR / f"{ds}.parquet"
    avail = set(pq.ParquetFile(f).schema.names)
    cols = ["query_id", "cde_id", "query_text_q3"] + \
        [c for c in ("family", "query_source") if c in avail]
    s = pd.read_parquet(f, columns=cols)
    s["query_id"] = s["query_id"].astype(str)
    s["pub"] = s["cde_id"].map(pub)
    golds = s.groupby("query_id")["pub"].apply(set).to_dict()
    q3 = s.dropna(subset=["query_text_q3"]).groupby("query_id")["query_text_q3"].first().to_dict()
    meta = {c: (s.groupby("query_id")[c].first().to_dict() if c in s.columns else {})
            for c in ("family", "query_source")}
    return golds, q3, meta


def load_se_pool(ds: str) -> pd.DataFrame:
    se = pd.read_parquet(SE_FILE[ds], columns=["split", "query_id", "cde_id",
                                               "biencoder_rank", "biencoder_score"])
    se = se[se["split"] == ds]
    se = se[se["biencoder_rank"] <= 200]
    d = dedup_best(se, "biencoder_rank")
    return d[d["biencoder_rank"] <= SE_K]


def load_kw_pool(ds: str) -> pd.DataFrame:
    k = pd.read_parquet(KW_FILE[ds], columns=["query_id", "cde_id",
                                              "cdematch_rank", "cdematch_score"])
    d = dedup_best(k, "cdematch_rank")
    return d[d["cdematch_rank"] <= KW_K]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    assert "artifacts_v3_cdisc" not in str(out_dir.resolve()), "protected path"
    for p in list(SE_FILE.values()) + list(KW_FILE.values()) + [CATALOG]:
        assert "cde_match_clone" != Path(p).name, f"clone input forbidden: {p}"
        assert "artifacts_v3_cdisc" not in str(p), f"v3_cdisc input forbidden: {p}"
    out_pq = out_dir / "pool_se20_kwfuzzy10.parquet"
    if out_pq.exists() and not args.overwrite:
        sys.exit(f"ERROR: exists (use --overwrite): {out_pq}")
    missing = [str(p) for p in list(SE_FILE.values()) + list(KW_FILE.values()) if not Path(p).exists()]
    assert not missing, "missing Stage A/B inputs:\n  " + "\n  ".join(missing)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cat = pd.read_parquet(CATALOG, columns=["cde_id"])
    cat["pub"] = cat["cde_id"].map(pub)
    cat_by_pub = (cat.sort_values("cde_id").drop_duplicates("pub")
                     .set_index("pub")["cde_id"].to_dict())

    frames, stats, dist_rows = [], {}, []
    for ds in DATASETS:
        print(f"[{ds}] loading se20 + kwfuzzy10 ...")
        golds, q3, meta = load_split_meta(ds)
        qids = sorted(q for q, g in golds.items() if g)
        sp = load_se_pool(ds); kp = load_kw_pool(ds)
        sp = sp[sp["query_id"].isin(qids)].copy()
        kp = kp[kp["query_id"].isin(qids)].copy()

        s = sp[["query_id", "public_id", "cde_id", "biencoder_rank", "biencoder_score"]]
        k = kp[["query_id", "public_id", "cde_id", "cdematch_rank", "cdematch_score"]]
        u = s.merge(k, on=["query_id", "public_id"], how="outer", suffixes=("_se", "_kw"))
        u["cand_source"] = np.where(
            u["biencoder_rank"].notna() & u["cdematch_rank"].notna(), "both",
            np.where(u["biencoder_rank"].notna(), "biencoder", "keyword"))
        u["cde_id"] = u["cde_id_se"].where(u["cde_id_se"].notna(), u["cde_id_kw"])
        u = u.drop(columns=["cde_id_se", "cde_id_kw"])
        u["split"] = ds
        u["is_label"] = [p in golds[q] for q, p in zip(u["query_id"], u["public_id"])]
        u["is_injected_gold"] = False

        n_injected = 0
        if ds in INJECT_SPLITS:
            covered = set(u.loc[u["is_label"], "query_id"])
            inj = []
            for q in qids:
                if q in covered:
                    continue
                reach = sorted(p for p in golds[q] if p in cat_by_pub)
                if not reach:
                    continue
                p0 = reach[0]
                inj.append({"query_id": q, "public_id": p0, "cde_id": cat_by_pub[p0],
                            "split": ds, "is_label": True, "is_injected_gold": True,
                            "cand_source": "injected_gold"})
            if inj:
                u = pd.concat([u, pd.DataFrame(inj)], ignore_index=True)
                n_injected = len(inj)

        u["query_text_q3"] = u["query_id"].map(q3)
        u["family"] = u["query_id"].map(meta["family"]).fillna("NA")
        u["query_source"] = u["query_id"].map(meta["query_source"]).fillna("NA")
        u["in_biencoder_topk"] = u["biencoder_rank"].notna()
        u["in_cdematch_topk"] = u["cdematch_rank"].notna()
        u["winner_id"] = WINNER_ID
        for c in ("biencoder_rank", "biencoder_score", "cdematch_rank", "cdematch_score"):
            if c not in u.columns:
                u[c] = np.nan
        frames.append(u[OUT_COLS])

        dep = u[~u["is_injected_gold"]]
        sizes = dep.groupby("query_id").size().reindex(qids, fill_value=0)
        gip = dep[dep["is_label"]]["query_id"].nunique()
        stats[ds] = {"n_queries": len(qids), "gold_in_pool": int(gip),
                     "pool_recall": round(gip / len(qids), 4),
                     "n_injected_gold": n_injected,
                     "pool_size_mean": round(float(sizes.mean()), 2),
                     "pool_size_min": int(sizes.min()), "pool_size_max": int(sizes.max()),
                     "n_rows_deployable": int(len(dep))}
        dist_rows.append({"dataset": ds, **stats[ds]})
        print(f"  q={len(qids)} gold_in_pool={gip} recall={stats[ds]['pool_recall']} "
              f"injected={n_injected} rows_dep={len(dep)}")

    # regression gate: val_dev gold_in_pool must equal the existing eval pool's
    if EXISTING_EVAL_POOL_SUMMARY.exists():
        ref = json.loads(EXISTING_EVAL_POOL_SUMMARY.read_text())["per_dataset"]["val_dev"]
        assert stats["val_dev"]["gold_in_pool"] == ref["gold_in_pool"], (
            f"val_dev gold_in_pool {stats['val_dev']['gold_in_pool']} != "
            f"existing eval pool {ref['gold_in_pool']} — pool semantics drifted")
        print(f"OK: val_dev gold_in_pool == existing eval pool ({ref['gold_in_pool']})")

    pool = pd.concat(frames, ignore_index=True)

    # `train` is NOT reachability-filtered (SPLIT_MANIFEST: "train uses all pairs"),
    # so some train golds are off-production. They remain VALID training positives
    # (candidate TEXT resolves in the FULL enriched catalogue). But any train query
    # left with NO positive at all — gold neither in the deployable pool nor
    # injectable in-production — cannot be trained and is dropped from train only.
    # This is the "corrected, production-eligible candidate pool" of S5.4; it removes
    # 172 queries for the canonical inputs. See crossencoder/eligibility.py.
    pool, dropped_train_queries = drop_train_queries_without_obtainable_positive(pool)
    is_train = pool["split"] == "train"

    full_pubs = set(pd.read_parquet(CATALOG_FULL, columns=["cde_id"])["cde_id"].map(pub))
    assert int((~pool["public_id"].isin(full_pubs)).sum()) == 0, "pool rows off FULL catalogue"
    # Gold must be in-production for EVAL/DEV splits (reachability-filtered);
    # off-production train golds are allowed (train-only positives, text from full cat).
    nontrain_gold = pool.loc[pool["is_label"] & ~is_train, "public_id"]
    assert int((~nontrain_gold.isin(cat_by_pub)).sum()) == 0, "non-train gold off-production"
    n_train_gold_offprod = int((~pool.loc[pool["is_label"] & is_train, "public_id"]
                                .isin(cat_by_pub)).sum())
    print(f"  train gold rows off-production (kept as valid training positives): "
          f"{n_train_gold_offprod}")
    assert not pool["query_text_q3"].isna().any(), "missing query_text_q3"
    pool.to_parquet(out_pq, index=False)

    summary = {"se_k": SE_K, "kw_k": KW_K, "branch": "kwfuzzy_only",
               "variant": "fulltrain (train + val_dev; eval reuses existing Step E pool)",
               "winner_id": WINNER_ID,
               "inputs": {"se_train": str(SE_TRAIN), "se_valdev": str(SE_VALDEV),
                          "kw_train": str(KW_TRAIN), "kw_valdev": str(KW_VALDEV)},
               "per_dataset": stats, "n_rows_total": int(len(pool)),
               "train_queries_dropped_no_obtainable_positive": len(dropped_train_queries),
               "train_queries_dropped_ids": dropped_train_queries}
    (out_dir / "pool_summary.json").write_text(json.dumps(summary, indent=2))
    d = pd.DataFrame(dist_rows)
    md = ["# Step E-fulltrain candidate pool — se20 ∪ kwfuzzy10 (train + val_dev)",
          "", "| " + " | ".join(d.columns) + " |",
          "| " + " | ".join("---" for _ in d.columns) + " |"]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    md += ["", f"Total rows: {len(pool)}  ({out_pq})", ""]
    (out_dir / "POOL_SUMMARY.md").write_text("\n".join(md))
    print(f"wrote {out_pq} ({len(pool)} rows) in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
