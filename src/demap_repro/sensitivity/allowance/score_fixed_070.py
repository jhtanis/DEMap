#!/usr/bin/env python3
"""Task K -- FIXED-MODEL inference-allowance sensitivity.

Score one allowance rate's regenerated feature table with the UNCHANGED
0.70-trained HGBC. Nothing is retrained, refit or reselected here.

Distinction from Task J (allowance-specific retraining):
  Task J: at each rate, rebuild the pipeline AND retrain the HGBC on that rate.
  Task K: at each rate, rebuild the pipeline but apply the FROZEN 0.70 model.
          This is the deployment scenario -- the shipped classifier meets
          inference-time exact-match evidence that differs from its training
          conditions.

The frozen inference contract (all loaded from the shipped artifact directory,
never re-derived):
  * model            hgbc_model.joblib  (HistGradientBoostingClassifier)
  * feature schema   feature_set.json -> included_features, in that exact order
  * preprocessing    trainer prepare_features(), stable categorical mode
  * categorical vocab categorical_vocab.json (train-fitted; inert -- 0 columns)
  * missing values   consumed natively by the HGBC; no imputer, no scaler
  * random state     42, baked into the fitted model; inference is deterministic
  * ranking          hgbc_score desc, ties by ascending CDE public identifier
                     (final_rank_v1_score_desc_then_public_id_asc)

Preprocessing and metric computation are imported from the pinned manuscript
trainer worktree (2903203) rather than reimplemented, so the inference path is
identical to the one that produced the shipped results by construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from demap_repro.utils.paths import data_root

#: Data and artifact tree. This was an absolute path into the research
#: repository, which made the module unusable anywhere else; see
#: ``demap_repro.utils.paths`` and ``DEMAP_DATA_ROOT``.
REPO = data_root()
WT = REPO / ".scratch/demap/paper_v13_scientific_audit/wt_hgbc_2903203"
SHIPPED = REPO / "artifacts/final_reranker/hgbc_reranker_v2_eligible/with_ce_noprov"

# The four caDSR-derived evaluation datasets are the primary analysis. GDC and
# CIMAC are held at allowance 1.0 by manuscript policy and are carried through
# only as clearly-labelled SECONDARY columns.
PRIMARY_SPLITS = ["test", "cctg", "oid_alt", "cdash"]
SECONDARY_SPLITS = ["gdc_combined", "cimac_v2"]
# val_train / val_dev are reported for completeness only. They are the frozen
# model's own training and selection splits, so their numbers are NOT
# out-of-sample and must never be read as robustness evidence.
DIAGNOSTIC_SPLITS = ["val_train", "val_dev"]

MANUSCRIPT_R5 = {"test": 0.9715, "cctg": 0.9088, "oid_alt": 0.8324, "cdash": 0.9198}

# The pinned manuscript trainer. It used to be imported from a worktree copy of
# ``scripts/train_hgbc_reranker.py`` in the research repository, put on sys.path
# ahead of everything else; ``demap_repro.reranker.train`` is that file migrated,
# with an identical ``prepare_features``.
from demap_repro.reranker import train as T
from demap_repro.reranker.features.categorical_vocab import CategoricalVocab


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def deployment_rank(g: pd.DataFrame) -> pd.DataFrame:
    """Shipped final-ranking convention: score descending, exact ties broken by
    ascending CDE public identifier. Numeric-aware so '100' sorts before '99'
    is impossible -- non-numeric identifiers sort last, then lexically."""
    g = g.copy()
    pub = g["cde_publicid"].astype(str)
    g["_pubnum"] = pd.to_numeric(pub, errors="coerce").fillna(np.inf)
    g = g.sort_values(
        ["query_id", "hgbc_score", "_pubnum", "cde_publicid"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    g["final_rank"] = g.groupby("query_id").cumcount() + 1
    return g


def metrics_for_split(g: pd.DataFrame) -> dict:
    """Deployment-style metrics: every query in the denominator, queries whose
    gold never entered the pool count as misses, injected gold excluded."""
    if "is_injected_gold" in g.columns:
        g = g[~g["is_injected_gold"].astype(bool)]
    g = deployment_rank(g)
    n = g["query_id"].nunique()
    gold = g[g["is_label"].astype(bool)]
    best = gold.groupby("query_id")["final_rank"].min()

    out = {"n_queries": int(n), "n_queries_with_gold_in_pool": int(best.shape[0])}
    for k in (1, 5, 10):
        out[f"recall@{k}"] = float((best <= k).sum() / n)
    out["mrr@100"] = float((1.0 / best[best <= 100]).sum() / n)
    return out


def pool_stats_for_split(g: pd.DataFrame) -> dict:
    """Candidate-pool statistics: what the frozen model was handed at this rate."""
    if "is_injected_gold" in g.columns:
        g = g[~g["is_injected_gold"].astype(bool)]
    sizes = g.groupby("query_id")["cde_id"].nunique()
    n = int(sizes.shape[0])
    gold_present = g[g["is_label"].astype(bool)]["query_id"].nunique()
    # Ceiling Recall@5 is bounded by pool membership, not by pool position: a
    # gold in the pool can always in principle be ranked top-5, one absent
    # never can.
    return {
        "n_queries": n,
        "ceiling_recall_gold_in_pool": float(gold_present / n),
        "n_gold_absent": int(n - gold_present),
        "pct_gold_absent": float(100.0 * (n - gold_present) / n),
        "pool_size_mean": float(sizes.mean()),
        "pool_size_median": float(sizes.median()),
        "pool_size_min": int(sizes.min()),
        "pool_size_max": int(sizes.max()),
        "pct_queries_lt30_candidates": float(100.0 * (sizes < 30).sum() / n),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", required=True, help="inference-time allowance rate")
    ap.add_argument("--feature-table", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-dir", default=str(SHIPPED))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    # ---- load the FROZEN inference contract -------------------------------
    model = joblib.load(model_dir / "hgbc_model.joblib")
    fs = json.load(open(model_dir / "feature_set.json"))
    feature_names = list(fs["included_features"])
    cat_vocab = CategoricalVocab.load(model_dir / "categorical_vocab.json")

    print(f"[frozen] model      {model_dir/'hgbc_model.joblib'}")
    print(f"[frozen] sha256     {sha256(model_dir/'hgbc_model.joblib')}")
    print(f"[frozen] type       {type(model).__name__}")
    print(f"[frozen] n_features {model.n_features_in_}  schema={len(feature_names)}")
    print(f"[frozen] params     {model.get_params()}")
    print(f"[frozen] categoricals {fs['categorical_columns']} "
          f"(mode={fs['categorical_mode']})")

    if model.n_features_in_ != len(feature_names):
        sys.exit(f"FATAL: model expects {model.n_features_in_} features, "
                 f"schema lists {len(feature_names)}")

    # ---- load this rate's regenerated feature table -----------------------
    ft = Path(args.feature_table)
    df = pd.read_parquet(ft)
    print(f"[rate {args.rate}] table {ft}")
    print(f"[rate {args.rate}] rows={len(df)} splits={sorted(df['split'].unique())}")

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        sys.exit(f"FATAL: feature table is missing {len(missing)} frozen "
                 f"features: {missing[:10]}")

    # ---- frozen preprocessing (trainer's own function) --------------------
    X_all, _ = T.prepare_features(df, categorical_vocab=cat_vocab)
    X = X_all[feature_names]          # frozen order; sklearn re-checks names
    if list(X.columns) != feature_names:
        sys.exit("FATAL: feature ordering diverged from the frozen schema")

    # ---- frozen inference -------------------------------------------------
    df = df.copy()
    df["hgbc_score"] = model.predict_proba(X)[:, 1]
    # Trainer-convention rank, retained so the 0.70 gate can be compared
    # column-for-column against the shipped scored-rankings artifact.
    df["hgbc_rank"] = (df.groupby(["split", "query_id"])["hgbc_score"]
                       .rank(ascending=False, method="first").astype("Int64"))

    keep = ["split", "query_id", "cde_id", "cde_publicid", "is_label",
            "hgbc_score", "hgbc_rank", "in_biencoder_topk", "in_cdematch_topk",
            "in_keyword_topk", "is_injected_gold", "is_exact_candidate"]
    keep = [c for c in keep if c in df.columns]
    scored = df[keep]
    scored.to_parquet(out / "hgbc_scored_rankings.parquet", index=False)

    # ---- metrics and pool statistics --------------------------------------
    res = {
        "experiment": "K_fixed070_inference_allowance",
        "design": "FIXED 0.70-trained HGBC; only inference-time allowance varies",
        "inference_allow_rate": args.rate,
        "model_dir": str(model_dir),
        "model_sha256": sha256(model_dir / "hgbc_model.joblib"),
        "model_params": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                             else str(v)) for k, v in model.get_params().items()},
        "n_features_in": int(model.n_features_in_),
        "feature_schema_sha256": hashlib.sha256(
            json.dumps(feature_names).encode()).hexdigest(),
        "feature_table": str(ft),
        "feature_table_rows": int(len(df)),
        "ranking_convention": "final_rank_v1_score_desc_then_public_id_asc",
        "retrained": False,
        "splits": {},
        "pool_stats": {},
    }

    for split in PRIMARY_SPLITS + SECONDARY_SPLITS + DIAGNOSTIC_SPLITS:
        g = df[df["split"] == split]
        if g.empty:
            continue
        m = metrics_for_split(g)
        p = pool_stats_for_split(g)
        tier = ("primary" if split in PRIMARY_SPLITS else
                "secondary_external_fixed_allowance_1.0" if split in SECONDARY_SPLITS
                else "diagnostic_in_sample_not_robustness_evidence")
        m["tier"] = p["tier"] = tier
        res["splits"][split] = m
        res["pool_stats"][split] = p
        print(f"[fixed070 a{args.rate} {split:>14s} {tier[:9]}] "
              f"n={m['n_queries']:>5d} R@1={m['recall@1']:.4f} "
              f"R@5={m['recall@5']:.4f} R@10={m['recall@10']:.4f} "
              f"MRR@100={m['mrr@100']:.4f} | ceiling={p['ceiling_recall_gold_in_pool']:.4f} "
              f"pool_mean={p['pool_size_mean']:.2f}")

    # ---- validation gate: 0.70 must reproduce the shipped pipeline --------
    if args.rate in ("0.70", "0.7"):
        print("\n=== 0.70 VALIDATION GATE vs the shipped pipeline ===")
        ship = pd.read_parquet(SHIPPED / "hgbc_scored_rankings.parquet")
        gate: dict = {"shipped_artifact": str(SHIPPED / "hgbc_scored_rankings.parquet")}

        gate["row_count_equal"] = bool(len(ship) == len(scored))
        key = ["split", "query_id", "cde_id"]
        a = scored.sort_values(key).reset_index(drop=True)
        b = ship.sort_values(key).reset_index(drop=True)
        gate["row_keys_identical"] = bool(a[key].equals(b[key]))

        if gate["row_keys_identical"]:
            d = np.abs(a["hgbc_score"].to_numpy() - b["hgbc_score"].to_numpy())
            gate["hgbc_score_max_abs_diff"] = float(d.max())
            gate["hgbc_score_n_not_bitequal"] = int(
                (a["hgbc_score"].to_numpy() != b["hgbc_score"].to_numpy()).sum())
            gate["hgbc_rank_identical"] = bool(
                a["hgbc_rank"].equals(b["hgbc_rank"]))
            gate["is_label_identical"] = bool(a["is_label"].equals(b["is_label"]))
            # Candidate-pool membership and ordering, per split.
            gate["pool_membership_identical_by_split"] = {
                s: bool(set(map(tuple, a[a.split == s][key].values))
                        == set(map(tuple, b[b.split == s][key].values)))
                for s in sorted(set(a["split"]))
            }
            ra = deployment_rank(a)[key + ["final_rank"]].sort_values(key).reset_index(drop=True)
            rb = deployment_rank(b)[key + ["final_rank"]].sort_values(key).reset_index(drop=True)
            gate["final_ranking_identical"] = bool(ra["final_rank"].equals(rb["final_rank"]))

        gate["table4_recall5"] = {
            s: {"manuscript": MANUSCRIPT_R5[s],
                "fixed070_rerun": res["splits"][s]["recall@5"],
                "delta": res["splits"][s]["recall@5"] - MANUSCRIPT_R5[s],
                "match_4dp": bool(abs(res["splits"][s]["recall@5"]
                                      - MANUSCRIPT_R5[s]) < 5e-5)}
            for s in MANUSCRIPT_R5 if s in res["splits"]
        }
        gate["ALL_GATES_PASS"] = bool(
            gate.get("row_count_equal") and gate.get("row_keys_identical")
            and gate.get("hgbc_score_n_not_bitequal") == 0
            and gate.get("final_ranking_identical")
            and all(v["match_4dp"] for v in gate["table4_recall5"].values())
        )
        res["validation_gate_070"] = gate
        for k, v in gate.items():
            if k != "pool_membership_identical_by_split":
                print(f"  {k}: {v}")
        if not gate["ALL_GATES_PASS"]:
            print("\n*** 0.70 DID NOT REPRODUCE — STOP AND INVESTIGATE ***")

    with open(out / f"metrics_fixed070_a{args.rate}.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {out / f'metrics_fixed070_a{args.rate}.json'}")


if __name__ == "__main__":
    main()
