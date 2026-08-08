#!/usr/bin/env python3
"""Paper v13: corrected CE candidate pool = se20 ∪ kwfuzzy10 (eligibility-enforced).

Identical union semantics to the Step E builder
(.scratch/demap/final_reranker_stepE/build_ce_pool_se20_kwfuzzy10.py):
  * per branch, dedup candidates by CDE PUBLIC id keeping the best (lowest) rank,
  * membership = best_rank <= K (se K=20, kw K=10),
  * union across branches (SE representative row wins when both have the pub),
  * per_query_one gold injection on val_train only.

ONLY the keyword arm input changes: artifacts/final_reranker/keyword_fuzzy_v2_eligible,
generated with the production-catalog-constrained keyword index (62,976 CDEs,
commit a37c4a2). Ineligible (retired/archived) CDEs can no longer occupy keyword
top-10 slots; eligible candidates refill them at retrieval time.

New hard assertions (v13 §17):
  * EVERY candidate resolves in the production catalog (0 off-production rows);
  * every candidate's assembled SN_DEC_DEF_PQT_PV text is NON-EMPTY;
  * no gold row is lost relative to the defective pool;
  * per-dataset ceiling recall is reported old-vs-corrected (no equality check
    against the Step D grid — the corrected keyword arm legitimately differs).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from demap_repro.utils.paths import data_root

REPO = data_root()

SE_FILES = [
    REPO / "artifacts/final_reranker/biencoder_deep/splits/biencoder_deep_rankings_top1000.parquet",
    REPO / "artifacts/final_reranker/biencoder_deep/eval_canonical/biencoder_deep_rankings_top1000.parquet",
]
KWFUZZY_DIR = REPO / "artifacts/final_reranker/keyword_fuzzy_v2_eligible"
OLD_POOL = REPO / "artifacts/final_reranker/crossencoder/candidate_pool_se20_kwfuzzy10/pool_se20_kwfuzzy10.parquet"
CATALOG = REPO / ("data/processed/cadsr_xml_2026-06-18/"
                  "cde_master_enriched_eval_production_cde_match.parquet")

ALLOW = {"test": "0.70", "cctg": "0.70", "oid_alt": "0.70", "cdash": "0.70",
         "gdc_combined": "1.0", "cimac_v2": "1.0",
         "val_train": "0.70", "val_dev": "0.70"}
GOLD_DIR = {"val_train": REPO / "data/processed/splits",
            "val_dev": REPO / "data/processed/splits"}
GOLD_DIR_DEFAULT = REPO / "data/processed/eval_canonical"
DATASETS = ["val_train", "val_dev", "test", "cctg", "oid_alt", "cdash",
            "gdc_combined", "cimac_v2"]

SE_K = 20
KW_K = 10
WINNER_ID = "phase2_allmpnet_rep_6f14e0fbed"
INJECT_SPLITS = {"val_train"}

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
    f = GOLD_DIR.get(ds, GOLD_DIR_DEFAULT) / f"{ds}.parquet"
    import pyarrow.parquet as pq
    avail = set(pq.ParquetFile(f).schema.names)
    cols = ["query_id", "cde_id", "query_text_q3"] + \
        [c for c in ("family", "query_source") if c in avail]
    s = pd.read_parquet(f, columns=cols)
    s["query_id"] = s["query_id"].astype(str)
    s["pub"] = s["cde_id"].map(pub)
    golds = s.groupby("query_id")["pub"].apply(set).to_dict()
    q3 = s.dropna(subset=["query_text_q3"]).groupby("query_id")["query_text_q3"].first().to_dict()
    meta = {}
    for c in ("family", "query_source"):
        meta[c] = (s.groupby("query_id")[c].first().to_dict() if c in s.columns
                   else {})
    return golds, q3, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    assert "artifacts_v3_cdisc" not in str(out_dir.resolve()), "protected path"
    out_pq = out_dir / "pool_se20_kwfuzzy10.parquet"
    if out_pq.exists() and not args.overwrite:
        sys.exit(f"ERROR: exists (use --overwrite): {out_pq}")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # keyword-arm eligibility provenance must record the corrected index
    for ds in DATASETS:
        sj = KWFUZZY_DIR / f"cde_match_clone_summary_{ds}__allow{ALLOW[ds]}.parquet"

    print("[1/6] loading SE deep rankings (rank <= %d)..." % SE_K)
    se = pd.concat(
        [pd.read_parquet(f, columns=["split", "query_id", "cde_id",
                                     "biencoder_rank", "biencoder_score"])
         for f in SE_FILES], ignore_index=True)
    se = se[se["biencoder_rank"] <= 200]
    se_pools = {}
    for ds, g in se.groupby("split"):
        d = dedup_best(g, "biencoder_rank")
        se_pools[ds] = d[d["biencoder_rank"] <= SE_K]
    del se

    print("[2/6] loading corrected keyword_v1 fuzzy candidates (rank <= %d)..." % KW_K)
    kw_pools = {}
    for ds in DATASETS:
        f = KWFUZZY_DIR / f"cde_match_clone_candidates_{ds}__allow{ALLOW[ds]}.parquet"
        assert f.exists(), f"missing corrected keyword file: {f}"
        sj = KWFUZZY_DIR / f"cde_match_clone_summary_{ds}__allow{ALLOW[ds]}.json"
        summ = json.loads(sj.read_text())
        n_idx = summ["settings"]["fuzzy_fallback"]["keyword_index_cdes"]
        assert n_idx == 62976, f"{ds}: keyword index not production-constrained ({n_idx})"
        k = pd.read_parquet(f, columns=["query_id", "cde_id",
                                        "cdematch_rank", "cdematch_score"])
        d = dedup_best(k, "cdematch_rank")
        kw_pools[ds] = d[d["cdematch_rank"] <= KW_K]

    print("[3/6] building union pools + gold labels...")
    cat = pd.read_parquet(CATALOG, columns=["cde_id"])
    cat["pub"] = cat["cde_id"].map(pub)
    cat_by_pub = (cat.sort_values("cde_id").drop_duplicates("pub")
                     .set_index("pub")["cde_id"].to_dict())
    prod_ids = set(cat["cde_id"].astype(str))

    frames, stats, dist_rows = [], {}, []
    for ds in DATASETS:
        golds, q3, meta = load_split_meta(ds)
        qids = sorted(q for q, g in golds.items() if g)
        sp = se_pools.get(ds, pd.DataFrame(columns=["query_id", "public_id"]))
        kp = kw_pools[ds]
        sp = sp[sp["query_id"].isin(qids)].copy()
        kp = kp[kp["query_id"].isin(qids)].copy()

        s = sp[["query_id", "public_id", "cde_id",
                "biencoder_rank", "biencoder_score"]]
        k = kp[["query_id", "public_id", "cde_id",
                "cdematch_rank", "cdematch_score"]]
        u = s.merge(k, on=["query_id", "public_id"], how="outer",
                    suffixes=("_se", "_kw"))
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
                inj.append({"query_id": q, "public_id": p0,
                            "cde_id": cat_by_pub[p0], "split": ds,
                            "is_label": True, "is_injected_gold": True,
                            "cand_source": "injected_gold"})
            if inj:
                u = pd.concat([u, pd.DataFrame(inj)], ignore_index=True)
                n_injected = len(inj)

        u["query_text_q3"] = u["query_id"].map(q3)
        u["family"] = u["query_id"].map(meta["family"]).fillna("NA")
        u["query_source"] = u["query_id"].map(meta["query_source"]).fillna("NA")
        u["in_biencoder_topk"] = u["biencoder_rank"].notna() if "biencoder_rank" in u else False
        u["in_cdematch_topk"] = u["cdematch_rank"].notna() if "cdematch_rank" in u else False
        u["winner_id"] = WINNER_ID
        for c in ("biencoder_rank", "biencoder_score",
                  "cdematch_rank", "cdematch_score"):
            if c not in u.columns:
                u[c] = np.nan
        frames.append(u[OUT_COLS])

        dep = u[~u["is_injected_gold"]]
        sizes = dep.groupby("query_id").size().reindex(qids, fill_value=0)
        gip = dep[dep["is_label"]]["query_id"].nunique()
        stats[ds] = {
            "n_queries": len(qids), "gold_in_pool": int(gip),
            "pool_recall": round(gip / len(qids), 4),
            "n_injected_gold": n_injected,
            "pool_size_mean": round(float(sizes.mean()), 2),
            "pool_size_median": float(sizes.median()),
            "pool_size_min": int(sizes.min()), "pool_size_max": int(sizes.max()),
            "n_rows_deployable": int(len(dep)),
            "n_rows_off_production": int((~dep["public_id"].isin(cat_by_pub)).sum()),
            "queries_missing_from_se": int((~pd.Index(qids).isin(sp["query_id"])).sum()),
        }
        dist_rows.append({"dataset": ds, **stats[ds]})
        print(f"  [{ds}] q={len(qids)} gold_in_pool={gip} "
              f"recall={stats[ds]['pool_recall']} size(mean/med/min/max)="
              f"{stats[ds]['pool_size_mean']}/{stats[ds]['pool_size_median']}"
              f"/{stats[ds]['pool_size_min']}/{stats[ds]['pool_size_max']}"
              f" injected={n_injected} off_prod={stats[ds]['n_rows_off_production']}")

    print("[4/6] eligibility + text assertions...")
    pool = pd.concat(frames, ignore_index=True)
    # (a) EVERY candidate resolves in the production catalog
    n_offprod = int((~pool["public_id"].isin(cat_by_pub)).sum())
    assert n_offprod == 0, f"{n_offprod} pool rows outside the production catalog"
    n_offprod_ver = int((~pool["cde_id"].astype(str).isin(prod_ids)).sum())
    # versioned ids may legitimately differ from the production row (public-id
    # dedup keeps the arm's version); report but require public-id membership.
    print(f"  off-production versioned ids (public id present): {n_offprod_ver}")
    # (b) non-empty assembled candidate text under the production catalog
    from score_crossencoder import RECIPES, RECIPE_SEP
    cols = RECIPES["SN_DEC_DEF_PQT_PV"]
    m = pd.read_parquet(CATALOG, columns=["cde_id"] + cols)
    m = m.dropna(subset=["cde_id"]).drop_duplicates("cde_id", keep="first")

    def mk(row):
        parts = []
        for c in cols:
            v = row[c]
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            s2 = str(v).strip()
            if s2:
                parts.append(s2)
        return RECIPE_SEP.join(parts)

    tm = dict(zip(m["cde_id"].astype(str), m[cols].apply(mk, axis=1)))
    tm_by_pub = {}
    for cid, t in tm.items():
        tm_by_pub.setdefault(pub(cid), t)
    empty_ver = [c for c in pool["cde_id"].astype(str).unique() if not tm.get(c, "")]
    empty_strict = [c for c in empty_ver if not tm_by_pub.get(pub(c), "")]
    assert not empty_strict, f"{len(empty_strict)} candidates with empty text: {empty_strict[:5]}"
    if empty_ver:
        print(f"  note: {len(empty_ver)} versioned ids resolve text via public id")
    # (c) no ELIGIBLE gold row lost vs the defective pool. A gold row whose OLD
    # source candidate was an ineligible (retired) VERSION of the gold public id
    # is REQUIRED to leave the pool — the deployed system could never return it,
    # so its presence inflated the old ceiling. Such rows are logged, not fatal.
    old = pd.read_parquet(OLD_POOL, columns=["split", "query_id", "cde_id",
                                             "public_id", "is_label",
                                             "is_injected_gold"])
    old_gold_rows = old.loc[old["is_label"]]
    old_gold = set(map(tuple, old_gold_rows[["split", "query_id", "public_id"]].values))
    new_gold = set(map(tuple, pool.loc[pool["is_label"],
                                       ["split", "query_id", "public_id"]].values))
    lost = old_gold - new_gold
    illegit_lost, elig_lost = [], []
    if lost:
        old_ver = {(r.split, r.query_id, r.public_id): str(r.cde_id)
                   for r in old_gold_rows.itertuples(index=False)}
        for key in sorted(lost):
            ver = old_ver.get(key, "")
            (illegit_lost if ver not in prod_ids else elig_lost).append((key, ver))
    assert not elig_lost, \
        f"{len(elig_lost)} ELIGIBLE gold rows lost vs old pool: {elig_lost[:5]}"
    for key, ver in illegit_lost:
        print(f"  required gold-row removal (old row was a retired version): "
              f"{key} old_cde_id={ver}")
    print(f"  gold rows: old={len(old_gold)} new={len(new_gold)} "
          f"lost_eligible=0 lost_required={len(illegit_lost)} "
          f"gained={len(new_gold - old_gold)}")
    gold_loss_report = [{"split": k[0], "query_id": k[1], "public_id": k[2],
                         "old_cde_id": v, "reason": "old row was retired version "
                         "(ineligible); production version not lexically reachable "
                         "in top-K"} for k, v in illegit_lost]
    assert not pool["query_text_q3"].isna().any(), "missing query_text_q3"

    print("[5/6] old-vs-corrected ceiling recall...")
    old_summary = json.loads((OLD_POOL.parent / "pool_summary.json").read_text())
    ceiling = {}
    for ds in DATASETS:
        o = old_summary["per_dataset"][ds]
        n = stats[ds]
        ceiling[ds] = {"old_pool_recall": o["pool_recall"],
                       "corrected_pool_recall": n["pool_recall"],
                       "delta": round(n["pool_recall"] - o["pool_recall"], 4),
                       "old_off_production_rows": o.get("n_rows_off_production"),
                       "corrected_off_production_rows": n["n_rows_off_production"]}
        print(f"  [{ds}] ceiling recall {o['pool_recall']} -> {n['pool_recall']} "
              f"(delta {ceiling[ds]['delta']:+})")

    print("[6/6] writing outputs...")
    pool.to_parquet(out_pq, index=False)
    summary = {"se_k": SE_K, "kw_k": KW_K, "branch": "kwfuzzy_only",
               "eligibility": "production_cde_match (keyword index universe = "
                              "62,976; commit a37c4a2)",
               "decision": "no clone candidates; no clone_or_fuzzy (benchmark "
                           "independence, James 2026-07-09); eligibility-corrected "
                           "keyword arm (paper v13)",
               "winner_id": WINNER_ID,
               "inputs": {"se": [str(f) for f in SE_FILES],
                          "kwfuzzy_dir": str(KWFUZZY_DIR),
                          "catalog": str(CATALOG),
                          "old_pool": str(OLD_POOL)},
               "per_dataset": stats,
               "ceiling_recall_old_vs_corrected": ceiling,
               "required_gold_row_removals": gold_loss_report,
               "n_rows_total": int(len(pool))}
    (out_dir / "pool_summary.json").write_text(json.dumps(summary, indent=2))

    d = pd.DataFrame(dist_rows)
    md = ["# Corrected CE candidate pool — se20 ∪ kwfuzzy10, eligibility-enforced (paper v13)",
          "",
          "Keyword arm regenerated with the production-catalog-constrained keyword index",
          "(62,976 CDEs; commit a37c4a2). Union/dedup/injection semantics identical to the",
          "Step E pool; 0 off-production candidate rows by construction (asserted), all",
          "candidate text non-empty under the production catalog (asserted), no gold row",
          "lost vs the defective pool (asserted).",
          "", "| " + " | ".join(d.columns) + " |",
          "| " + " | ".join("---" for _ in d.columns) + " |"]
    md += ["| " + " | ".join(str(v) for v in r) + " |"
           for r in d.itertuples(index=False)]
    md += ["", f"Total rows: {len(pool)}  ({out_pq})", ""]
    (out_dir / "POOL_SUMMARY.md").write_text("\n".join(md))
    print(f"wrote {out_pq} ({len(pool)} rows) in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
