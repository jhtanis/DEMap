#!/usr/bin/env python3
"""Train and evaluate an HGBC reranker on the candidate-level feature table.

Deployment-style evaluation: injected gold rows (``is_injected_gold=True``)
are excluded from the candidate set when computing Recall@K / MRR. Queries
whose gold CDE is absent from the deployable candidate set count as misses
(rank=inf) in the denominator.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/train_hgbc_reranker.py
    PYTHONPATH=src .venv/bin/python scripts/train_hgbc_reranker.py --feature-table path/to/ft.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from demap_repro.utils.paths import data_root

REPO_ROOT = data_root()

from demap_repro.pool.candidate_union import FORBIDDEN_LEAKAGE_COLUMNS
from demap_repro.reranker.features.categorical_vocab import (
    FALLBACK_TOKEN,
    VOCAB_FILENAME,
    VOCAB_SCHEMA_VERSION,
    CategoricalVocab,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_FT = "artifacts_v3_cdisc/hgbc_reranker/feature_table.parquet"
DEFAULT_OUT = "artifacts_v3_cdisc/hgbc_reranker"

TRAIN_SPLIT = "val_train"
TUNE_SPLIT = "val_dev"
# Default June-18 report splits. main() derives the actual report set from the
# feature table (all splits except train/tune), so this stays correct even if the
# table's split set changes; the constant documents the expected June-18 splits.
REPORT_SPLITS = [
    "test",
    "external_holdout_org",
    "external_holdout_refslice",
    "external_holdout_gdc_altnames",
    "external_holdout_gdc_questiontext",
    "cimac_v2",
    "theradex6_test",
]

METADATA_COLS = {
    "split", "query_id", "cde_id", "cde_publicid", "pair_id", "family",
    "query_source", "is_label", "is_injected_gold", "winner_id", "query_text_q3",
    "PV_BLOCK_SDE", "pv_attached", "hydration_policy",
    # fixed-K build: exact-tier flag is used by the exact-pinned final ranking
    # (scripts/rank_fixed_k_exact_pinned.py), NOT as an HGBC input feature.
    "is_exact_candidate",
}

FORBIDDEN_FEATURE_COLS = FORBIDDEN_LEAKAGE_COLUMNS | METADATA_COLS

CATEGORICAL_COLS = ["text_family", "text_query_source"]

GRID = {
    "max_iter": [200, 500],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "min_samples_leaf": [10, 30],
}

K_VALUES = [1, 5, 10, 100]


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------


def _load_feature_spec(path: str, key: str) -> List[str]:
    """Load a feature-name list from a spec file. Accepts a bare JSON list, a JSON
    object with the given ``key`` (e.g. {"drop": [...]} / {"keep_only": [...]}), or a
    newline-delimited text file (``#`` comments allowed)."""
    txt = Path(path).read_text()
    try:
        obj = json.loads(txt)
        names = (obj.get(key) or obj.get("features") or []) if isinstance(obj, dict) else obj
    except json.JSONDecodeError:
        names = [ln.strip() for ln in txt.splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
    return [str(c) for c in names]


def prepare_features(
    df: pd.DataFrame,
    *,
    categorical_vocab: CategoricalVocab | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build the numeric feature matrix.

    ``categorical_vocab=None`` (legacy): object columns get TABLE-RELATIVE
    ``pd.Categorical`` codes — the historical behavior, kept only so frozen
    models and reference retrains reproduce. NOT batch-composition-safe.

    ``categorical_vocab=CategoricalVocab``: declared categorical columns are
    encoded with the FIXED training-derived vocabulary (missing -> __MISSING__,
    unseen -> __UNKNOWN__), so codes never depend on the table being scored.
    Any other object-dtype column is a hard error (fail closed rather than
    silently table-encode).
    """
    feature_cols = [c for c in df.columns if c not in FORBIDDEN_FEATURE_COLS]
    X = df[feature_cols].copy()

    if categorical_vocab is None:
        for c in CATEGORICAL_COLS:
            if c in X.columns:
                X[c] = X[c].fillna("").astype(str)

        cat_mask = X.dtypes == object
        for c in X.columns[cat_mask]:
            X[c] = pd.Categorical(X[c]).codes.astype(float)
            X.loc[X[c] < 0, c] = np.nan
    else:
        for c in CATEGORICAL_COLS:
            if c in X.columns:
                X[c] = categorical_vocab.encode_series(X[c], c)
        stray = [c for c in X.columns[X.dtypes == object] if c not in CATEGORICAL_COLS]
        if stray:
            raise ValueError(
                f"stable categorical mode: unexpected object-dtype feature columns {stray}; "
                f"declare them in CATEGORICAL_COLS or exclude them"
            )

    for c in X.columns:
        if X[c].dtype == "Int64":
            X[c] = X[c].astype(float)
        elif X[c].dtype == bool:
            X[c] = X[c].astype(float)

    return X, list(X.columns)


# ---------------------------------------------------------------------------
# Metrics — vectorized, deployment style
# ---------------------------------------------------------------------------


def compute_ranking_metrics(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "is_label",
    query_col: str = "query_id",
    k_values: List[int] = K_VALUES,
    total_queries: int = 0,
) -> Dict[str, float]:
    """Compute Recall@K and MRR@100 over a candidate DataFrame (vectorized).

    ``total_queries`` sets the denominator. When >0, queries absent from
    ``df`` (or present but with no positive) count as misses. When 0
    (default), only queries with at least one positive are counted.
    """
    if df.empty:
        results = {f"recall@{k}": 0.0 for k in k_values}
        results["mrr@100"] = 0.0
        results["n_queries"] = total_queries
        results["n_queries_with_gold_in_candidates"] = 0
        return results

    scored = df[[query_col, score_col, label_col]].copy()
    scored = scored.sort_values([query_col, score_col], ascending=[True, False])
    scored["_rank"] = scored.groupby(query_col).cumcount() + 1

    gold = scored[scored[label_col]]
    if gold.empty:
        gold_ranks = pd.Series(dtype=float)
    else:
        gold_ranks = gold.groupby(query_col)["_rank"].min()

    n_hit = len(gold_ranks)
    denom = total_queries if total_queries > 0 else n_hit

    results: Dict[str, float] = {}
    for k in k_values:
        results[f"recall@{k}"] = float((gold_ranks <= k).sum()) / denom if denom else 0.0
    mrr_eligible = gold_ranks[gold_ranks <= 100]
    results["mrr@100"] = float((1.0 / mrr_eligible).sum()) / denom if denom else 0.0

    results["n_queries"] = denom
    results["n_queries_with_gold_in_candidates"] = n_hit
    return results


def _precompute_gold_ranks(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "is_label",
    query_col: str = "query_id",
) -> pd.Series:
    """Return a Series of best gold rank per query (vectorized).

    Queries without a positive candidate are absent from the result.
    """
    if df.empty:
        return pd.Series(dtype=float, name="_gold_rank")
    scored = df[[query_col, score_col, label_col]].copy()
    scored = scored.sort_values([query_col, score_col], ascending=[True, False])
    scored["_rank"] = scored.groupby(query_col).cumcount() + 1
    gold = scored[scored[label_col]]
    if gold.empty:
        return pd.Series(dtype=float, name="_gold_rank")
    return gold.groupby(query_col)["_rank"].min().rename("_gold_rank")


def _metrics_from_gold_ranks(
    gold_ranks: pd.Series,
    total_queries: int,
    k_values: List[int] = K_VALUES,
) -> Dict[str, float]:
    """Compute Recall@K and MRR@100 from precomputed gold ranks."""
    n_hit = len(gold_ranks)
    denom = total_queries if total_queries > 0 else n_hit
    results: Dict[str, float] = {}
    for k in k_values:
        results[f"recall@{k}"] = float((gold_ranks <= k).sum()) / denom if denom else 0.0
    mrr_eligible = gold_ranks[gold_ranks <= 100]
    results["mrr@100"] = float((1.0 / mrr_eligible).sum()) / denom if denom else 0.0
    results["n_queries"] = denom
    results["n_queries_with_gold_in_candidates"] = n_hit
    return results


def _deployment_metrics(
    split_df: pd.DataFrame,
    score_col: str,
) -> Dict[str, float]:
    """Compute deployment-style metrics: exclude injected gold, count all queries."""
    has_flag = "is_injected_gold" in split_df.columns
    if has_flag:
        deployable = split_df[~split_df["is_injected_gold"]]
    else:
        deployable = split_df

    total_q = split_df["query_id"].nunique()
    return compute_ranking_metrics(
        deployable, score_col, total_queries=total_q,
    )


def _deployment_gold_ranks(
    split_df: pd.DataFrame,
    score_col: str,
) -> Tuple[pd.Series, int]:
    """Return (gold_ranks, total_queries) for deployment-style evaluation."""
    has_flag = "is_injected_gold" in split_df.columns
    deployable = split_df[~split_df["is_injected_gold"]] if has_flag else split_df
    total_q = split_df["query_id"].nunique()
    return _precompute_gold_ranks(deployable, score_col), total_q


def compute_stratum_metrics(
    df: pd.DataFrame,
    score_col: str,
    strata_cols: List[str],
) -> pd.DataFrame:
    """Compute per-stratum metrics efficiently from precomputed gold ranks."""
    has_flag = "is_injected_gold" in df.columns
    deployable = df[~df["is_injected_gold"]] if has_flag else df

    gold_ranks = _precompute_gold_ranks(deployable, score_col)

    query_meta = df.drop_duplicates("query_id").set_index("query_id")[strata_cols]

    gold_with_meta = gold_ranks.to_frame().join(query_meta, how="right")
    gold_with_meta["_gold_rank"] = gold_with_meta["_gold_rank"].fillna(np.inf)

    rows = []
    for keys, grp in gold_with_meta.groupby(strata_cols):
        if isinstance(keys, str):
            keys = (keys,)
        total_q = len(grp)
        finite = grp["_gold_rank"][grp["_gold_rank"] < np.inf]
        m = _metrics_from_gold_ranks(finite, total_q)
        row = dict(zip(strata_cols, keys))
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_strata(
    strata_df: pd.DataFrame,
    split: str,
    group_col: str,
) -> pd.DataFrame:
    """Aggregate per-(family, query_source) strata to per-query_source totals."""
    sub = strata_df[strata_df["split"] == split].copy()
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for val, g in sub.groupby(group_col):
        n = int(g["n_queries"].sum())
        if n == 0:
            continue
        row = {"split": split, group_col: val, "n_queries": n}
        for metric in ["recall@1", "recall@5", "recall@10", "mrr@100"]:
            if metric in g.columns:
                row[metric] = (g[metric] * g["n_queries"]).sum() / n
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Baselines — deployment style
# ---------------------------------------------------------------------------


def eval_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate bi-encoder and CDE Match baselines with deployment metrics.

    For each baseline, rank candidates from that source only.  Queries whose
    gold CDE is not in that source's candidate set contribute 0 to all
    recall/MRR metrics but still count in the denominator.
    """
    rows = []
    for split, g in df.groupby("split"):
        total_q = g["query_id"].nunique()

        bienc = g[g["in_biencoder_topk"]].copy()
        if not bienc.empty:
            m = compute_ranking_metrics(bienc, "biencoder_score", total_queries=total_q)
            rows.append({"split": split, "method": "biencoder", **m})

        if "in_cdematch_topk" in g.columns and "cdematch_score" in g.columns:
            cm = g[g["in_cdematch_topk"]].copy()
            if not cm.empty:
                m = compute_ranking_metrics(cm, "cdematch_score", total_queries=total_q)
                rows.append({"split": split, "method": "cdematch", **m})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# HGBC grid search
# ---------------------------------------------------------------------------


def run_grid_search(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_tune: pd.DataFrame,
    tune_df: pd.DataFrame,
    feature_names: List[str],
    categorical_features: List[int] | None = None,
) -> Tuple[Any, Dict, pd.DataFrame]:
    from sklearn.ensemble import HistGradientBoostingClassifier

    configs = list(product(
        GRID["max_iter"],
        GRID["max_depth"],
        GRID["learning_rate"],
        GRID["min_samples_leaf"],
    ))

    # Only pass categorical_features when explicitly requested, so the legacy
    # (table-relative) path constructs the estimator with byte-identical params.
    extra_kwargs: Dict[str, Any] = {}
    if categorical_features:
        extra_kwargs["categorical_features"] = list(categorical_features)

    grid_rows = []
    best_model = None
    best_config = None
    best_score = -1.0

    for max_iter, max_depth, lr, min_leaf in configs:
        model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=lr,
            min_samples_leaf=min_leaf,
            random_state=42,
            **extra_kwargs,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        proba = model.predict_proba(X_tune)[:, 1]
        scored = tune_df.copy()
        scored["_score"] = proba
        metrics = _deployment_metrics(scored, "_score")

        config_dict = {
            "max_iter": max_iter,
            "max_depth": max_depth,
            "learning_rate": lr,
            "min_samples_leaf": min_leaf,
        }
        row = {**config_dict, **{f"tune_{k}": v for k, v in metrics.items()}}
        grid_rows.append(row)

        score = metrics["recall@5"]
        tag = f"iter={max_iter} depth={max_depth} lr={lr} leaf={min_leaf}"
        print(f"    {tag}: R@5={score:.4f} R@10={metrics['recall@10']:.4f} MRR={metrics['mrr@100']:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_config = config_dict

    grid_df = pd.DataFrame(grid_rows)
    return best_model, best_config, grid_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feature-table", default=str(REPO_ROOT / DEFAULT_FT))
    ap.add_argument("--out", default=str(REPO_ROOT / DEFAULT_OUT))
    ap.add_argument("--exclude-features-file", default=None,
                    help="JSON/text spec of feature columns to DROP (blacklist). JSON may be a "
                         "bare list or {\"drop\": [...]}. Only drops; cannot add/leak features.")
    ap.add_argument("--keep-features-file", default=None,
                    help="JSON/text spec of feature columns to KEEP (whitelist); everything else "
                         "is dropped. JSON may be a bare list or {\"keep_only\": [...]}.")
    ap.add_argument("--categorical-mode", choices=["legacy", "stable"], default="legacy",
                    help="'legacy' (default): historical table-relative pd.Categorical codes. "
                         "'stable': fit a fixed vocabulary on the TRAIN split only, persist it "
                         "(categorical_vocab.json), encode absent/empty/unseen to the reserved "
                         "__FALLBACK__ category, and train with sklearn-native "
                         "categorical_features.")
    args = ap.parse_args(argv)

    ft_path = Path(args.feature_table)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ft_path.exists():
        sys.exit(f"ERROR: feature table not found: {ft_path}")

    print(f"Loading feature table: {ft_path}")
    df = pd.read_parquet(ft_path)
    print(f"  {len(df)} rows, {df['split'].nunique()} splits, {df['query_id'].nunique()} queries")

    has_injected = "is_injected_gold" in df.columns
    if has_injected:
        n_inj = df["is_injected_gold"].sum()
        print(f"  Injected gold rows: {n_inj} (excluded from deployment metrics)")
    else:
        print("  WARNING: no is_injected_gold column; all rows treated as deployable")

    bad = sorted(set(df.columns) & FORBIDDEN_LEAKAGE_COLUMNS)
    if bad:
        sys.exit(f"ERROR: leakage columns in feature table: {bad}")

    # Stable mode: fit the categorical vocabulary on the TRAIN split ONLY, then
    # encode the whole table with it (missing -> __MISSING__, unseen -> __UNKNOWN__).
    cat_vocab: CategoricalVocab | None = None
    if args.categorical_mode == "stable":
        fit_rows = df[df["split"] == TRAIN_SPLIT]
        if fit_rows.empty:
            sys.exit(f"ERROR: stable categorical mode: no {TRAIN_SPLIT!r} rows to fit on")
        # Vocabulary is fitted on PRE-mask values so every training category is
        # retained even when some of its queries are later masked.
        cat_vocab = CategoricalVocab.fit(
            fit_rows,
            [c for c in CATEGORICAL_COLS if c in df.columns],
            fitted_on=f"{ft_path} split={TRAIN_SPLIT} n_rows={len(fit_rows)}",
        )
        vocab_path = cat_vocab.save(out_dir / VOCAB_FILENAME)
        for c, v in cat_vocab.columns.items():
            print(f"  stable vocab {c}: {len(v)} categories (incl. 1 reserved)")
        print(f"  persisted vocabulary: {vocab_path}")

    X_all, feature_names = prepare_features(df, categorical_vocab=cat_vocab)
    print(f"  {len(feature_names)} features after encoding")

    leaked = set(feature_names) & FORBIDDEN_FEATURE_COLS
    if leaked:
        sys.exit(f"ERROR: forbidden columns in feature matrix: {leaked}")

    # --- optional feature exclusion (ablation; drops columns only -> no leakage) ---
    all_features = list(feature_names)
    drop_from_exclude = (_load_feature_spec(args.exclude_features_file, "drop")
                         if args.exclude_features_file else [])
    keep_only = (_load_feature_spec(args.keep_features_file, "keep_only")
                 if args.keep_features_file else None)
    drop_set = set(drop_from_exclude)
    if keep_only is not None:
        drop_set |= {c for c in all_features if c not in set(keep_only)}
    feature_names = [c for c in all_features if c not in drop_set]
    X_all = X_all[feature_names]
    excluded_features = [c for c in all_features if c not in feature_names]
    exclude_not_present = sorted(c for c in drop_from_exclude if c not in all_features)
    keep_not_present = sorted(c for c in (keep_only or []) if c not in all_features)
    if args.exclude_features_file or args.keep_features_file:
        print(f"  feature exclusion: {len(all_features)} -> {len(feature_names)} features "
              f"(dropped {len(excluded_features)})")
        if exclude_not_present:
            print(f"  WARNING: exclude names not in table (ignored): {exclude_not_present}")
        if keep_not_present:
            print(f"  WARNING: keep names not in table (ignored): {keep_not_present}")
    if not feature_names:
        sys.exit("ERROR: feature exclusion removed ALL features.")
    # Categorical columns surviving exclusion; indices are into feature_names,
    # i.e. the column order of the fitted X matrix.
    cat_cols_used = [c for c in CATEGORICAL_COLS if c in feature_names]
    cat_indices = [feature_names.index(c) for c in cat_cols_used]
    if args.categorical_mode == "stable" and cat_cols_used:
        print(f"  native categorical features: {cat_cols_used} (indices {cat_indices})")

    with open(out_dir / "feature_set.json", "w") as fh:
        json.dump({
            "feature_table": str(ft_path),
            "n_features_used": len(feature_names),
            "included_features": feature_names,
            "excluded_features": excluded_features,
            "exclude_features_file": args.exclude_features_file,
            "keep_features_file": args.keep_features_file,
            "exclude_names_not_in_table": exclude_not_present,
            "keep_names_not_in_table": keep_not_present,
            "categorical_mode": args.categorical_mode,
            "categorical_columns": cat_cols_used,
            "categorical_feature_indices": cat_indices,
            "categorical_vocab_file": (VOCAB_FILENAME if args.categorical_mode == "stable"
                                       else None),
            "categorical_schema_version": (VOCAB_SCHEMA_VERSION
                                           if args.categorical_mode == "stable" else None),
            # The published model trains on the unmodified provenance columns; the
            # query-provenance dropout experiment is not part of this pipeline.
            "provenance_dropout_rate": 0.0,
            "provenance_dropout_policy_file": None,
            # telemetry: known/missing/unknown input counts per categorical column
            # (post-mask, whole table) — missing vs unknown stay observable here
            # even though both encode to the __FALLBACK__ code.
            "categorical_telemetry": ({
                c: cat_vocab.telemetry(df[c], c) for c in cat_cols_used
            } if args.categorical_mode == "stable" else None),
        }, fh, indent=2)

    # Split data — train includes injected gold rows for positive supervision
    train_mask = df["split"] == TRAIN_SPLIT
    tune_mask = df["split"] == TUNE_SPLIT
    X_train = X_all[train_mask]
    y_train = df.loc[train_mask, "is_label"].astype(int).values

    X_tune = X_all[tune_mask]
    tune_meta_cols = ["split", "query_id", "cde_id", "is_label",
                      "in_biencoder_topk", "in_cdematch_topk",
                      "biencoder_score", "cdematch_score"]
    if has_injected:
        tune_meta_cols.append("is_injected_gold")
    tune_meta_cols = [c for c in tune_meta_cols if c in df.columns]
    tune_df = df[tune_mask][tune_meta_cols].copy()

    print(f"\nTrain: {len(X_train)} rows ({y_train.sum()} pos, {(~y_train.astype(bool)).sum()} neg)")
    print(f"Tune:  {len(X_tune)} rows")

    # Baselines — deployment style
    print("\nEvaluating baselines (deployment-style, all queries in denominator)...")
    baseline_df = eval_baselines(df)
    baseline_df.to_csv(out_dir / "baseline_eval_by_split.csv", index=False)
    for _, row in baseline_df[baseline_df["split"].isin([TUNE_SPLIT, "test"])].iterrows():
        print(
            f"  {row['method']:>12s} | {row['split']:>10s}: "
            f"n={int(row['n_queries'])} "
            f"R@5={row['recall@5']:.4f} R@10={row['recall@10']:.4f} "
            f"MRR={row['mrr@100']:.4f}"
        )

    # Grid search — deployment style on val_dev
    print(f"\nGrid search ({len(list(product(*GRID.values())))} configs)...")
    best_model, best_config, grid_df = run_grid_search(
        X_train, y_train, X_tune, tune_df, feature_names,
        categorical_features=(cat_indices if args.categorical_mode == "stable" else None),
    )
    grid_df.to_csv(out_dir / "hgbc_grid_results.csv", index=False)
    print(f"\nBest config: {best_config}")

    with open(out_dir / "selected_hgbc_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

    # Save model
    import joblib
    joblib.dump(best_model, out_dir / "hgbc_model.joblib")
    print(f"Saved model: {out_dir / 'hgbc_model.joblib'}")

    # Full evaluation — deployment style
    print("\nEvaluating best model on all splits (deployment-style)...")
    proba_all = best_model.predict_proba(X_all)[:, 1]
    df = df.copy()
    df["hgbc_score"] = proba_all
    # Final HGBC rank per (split, query): 1 = highest score.
    df["hgbc_rank"] = (df.groupby(["split", "query_id"])["hgbc_score"]
                       .rank(ascending=False, method="first").astype("Int64"))

    # Report splits: derive from the table (all splits except train/tune) so the
    # eval set tracks whatever the feature table contains; fall back to the
    # documented REPORT_SPLITS order for stable display.
    present = list(df["split"].unique())
    derived = [s for s in present if s not in (TRAIN_SPLIT, TUNE_SPLIT)]
    report_splits = ([s for s in REPORT_SPLITS if s in derived]
                     + [s for s in sorted(derived) if s not in REPORT_SPLITS])
    eval_splits = [TUNE_SPLIT] + report_splits

    # Save per-candidate scored rankings (for downstream CIMAC 131/92/39
    # exact/non-exact stratified evaluation, which is not one of this script's
    # family/query_source strata).
    _save_scored_rankings(df, out_dir)
    eval_rows = []
    for split in eval_splits:
        g = df[df["split"] == split]
        if g.empty:
            continue
        m = _deployment_metrics(g, "hgbc_score")
        eval_rows.append({"split": split, "method": "hgbc", **m})
        print(
            f"  hgbc | {split:>35s}: n={m['n_queries']:>5d} "
            f"(gold_in_cands={m['n_queries_with_gold_in_candidates']}) "
            f"R@1={m['recall@1']:.4f} R@5={m['recall@5']:.4f} "
            f"R@10={m['recall@10']:.4f} MRR={m['mrr@100']:.4f}"
        )

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(out_dir / "hgbc_eval_by_split.csv", index=False)

    # Forced-fallback robustness diagnostics (val_dev only; reported, never used
    # for selection by this script). Natural is recomputed as a sanity anchor.
    if args.categorical_mode == "stable" and cat_cols_used:
        conditions = [("natural", []),
                      ("forced_both", cat_cols_used),
                      ("forced_family", [c for c in ["text_family"] if c in cat_cols_used]),
                      ("forced_source", [c for c in ["text_query_source"] if c in cat_cols_used])]
        rob_rows = []
        vd = df[df["split"] == TUNE_SPLIT]
        for cond_name, force_cols in conditions:
            d = vd.copy()
            for c in force_cols:
                d[c] = FALLBACK_TOKEN
            Xc, _ = prepare_features(d, categorical_vocab=cat_vocab)
            Xc = Xc[feature_names]
            d["_rob_score"] = best_model.predict_proba(Xc)[:, 1]
            m = _deployment_metrics(d, "_rob_score")
            rob_rows.append({"condition": cond_name, "forced_columns": "|".join(force_cols), **m})
            print(f"  robustness | {cond_name:>13s}: R@5={m['recall@5']:.4f} "
                  f"R@10={m['recall@10']:.4f} MRR={m['mrr@100']:.4f}")
        pd.DataFrame(rob_rows).to_csv(out_dir / "robustness_diagnostics.csv", index=False)

    # Stratum evaluation
    print("\nStratum evaluation...")
    strata_rows = []
    for split in eval_splits:
        g = df[df["split"] == split]
        if g.empty:
            continue
        sm = compute_stratum_metrics(g, "hgbc_score", ["split", "family", "query_source"])
        strata_rows.append(sm)

    if strata_rows:
        strata_df = pd.concat(strata_rows, ignore_index=True)
        strata_df.to_csv(out_dir / "hgbc_eval_by_split_stratum.csv", index=False)

        # Aggregate to REF/ALT totals for key splits
        for split in ["test", "external_holdout_org"]:
            agg = _aggregate_strata(strata_df, split, "query_source")
            if agg.empty:
                continue
            for _, r in agg.iterrows():
                label = f"{split[:4]} {r['query_source']}"
                print(
                    f"  {label:>15s}: n={int(r['n_queries']):>5d} "
                    f"R@1={r['recall@1']:.4f} R@5={r['recall@5']:.4f} "
                    f"R@10={r['recall@10']:.4f} MRR={r['mrr@100']:.4f}"
                )

        # Key individual strata
        for label, filt in [
            ("test CDISC", (strata_df["split"] == "test") & (strata_df["family"] == "CDISC")),
            ("ext CCTG/REF", (strata_df["split"] == "external_holdout_org") & (strata_df["family"] == "CCTG")),
            ("ext OID/ALT", (strata_df["split"] == "external_holdout_org") & (strata_df["family"] == "OID")),
        ]:
            sub = strata_df[filt]
            if sub.empty:
                continue
            r = sub.iloc[0]
            print(
                f"  {label:>15s}: n={int(r['n_queries']):>5d} "
                f"R@1={r['recall@1']:.4f} R@5={r['recall@5']:.4f} "
                f"R@10={r['recall@10']:.4f} MRR={r['mrr@100']:.4f}"
            )

    # Feature importance — graceful fallback
    try:
        importances = best_model.feature_importances_
        fi = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi = fi.sort_values("importance", ascending=False)
        fi.to_csv(out_dir / "hgbc_feature_importance.csv", index=False)
        print(f"\nTop 10 features:")
        for _, r in fi.head(10).iterrows():
            print(f"  {r['feature']:>45s}: {r['importance']:.4f}")
    except AttributeError:
        print("\nFeature importance not available for this HGBC version.")

    # Summary
    _write_summary(out_dir, best_config, eval_df, baseline_df, grid_df)
    print(f"\nOutputs written to: {out_dir}/")


def _save_scored_rankings(df: pd.DataFrame, out_dir: Path) -> Path:
    """Write per-candidate HGBC scores + ranks for downstream stratified eval
    (e.g. the CIMAC v2 131/92/39 exact/non-exact split, which this script does
    not compute). Selects the available identity/score/source-flag columns."""
    cols = ["split", "query_id", "cde_id", "cde_publicid", "is_label",
            "hgbc_score", "hgbc_rank", "in_biencoder_topk", "in_cdematch_topk",
            "in_keyword_topk", "is_injected_gold", "is_exact_candidate"]
    cols = [c for c in cols if c in df.columns]
    path = out_dir / "hgbc_scored_rankings.parquet"
    df[cols].to_parquet(path, index=False)
    print(f"Saved scored rankings: {path} ({len(df)} rows, {len(cols)} cols)")
    return path


def _write_summary(
    out_dir: Path,
    config: Dict,
    eval_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    grid_df: pd.DataFrame,
):
    lines = [
        "# HGBC Reranker Summary",
        "",
        "Deployment-style metrics: injected gold rows excluded from candidate ranking.",
        "All queries count in the denominator (gold-absent queries = miss).",
        "",
        "## Best Configuration",
        "",
        "```json",
        json.dumps(config, indent=2),
        "```",
        "",
        "## Evaluation by Split",
        "",
    ]

    header = "| split | method | n | gold_in_cands | R@1 | R@5 | R@10 | MRR@100 |"
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    lines += [header, sep]

    combined = pd.concat([eval_df, baseline_df], ignore_index=True)
    for _, r in combined.sort_values(["split", "method"]).iterrows():
        gic = int(r.get("n_queries_with_gold_in_candidates", r["n_queries"]))
        lines.append(
            f"| {r['split']} | {r['method']} | {int(r['n_queries'])} | {gic} "
            f"| {r['recall@1']:.4f} | {r['recall@5']:.4f} "
            f"| {r['recall@10']:.4f} | {r['mrr@100']:.4f} |"
        )

    lines.append("")
    (out_dir / "hgbc_reranker_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
