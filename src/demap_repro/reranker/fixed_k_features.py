"""Build the fixed-K candidate feature table (K = 30).

The candidate pool for each (split, query) is the union, deduplicated by CDE
public identifier, of

* FT-MPNet bi-encoder top ``--biencoder-k`` (20 in the paper), and
* CDE Match-Fuzzy top ``--keyword-fuzzy-k`` (10 in the paper) by ``cdematch_rank``.

Nominal K is therefore 30; because the two arms overlap, the realized pool
averages about 27.6 candidates per query. No gold is injected — the pool is the
one a deployed system would see.

Allowance routing
-----------------
Which fuzzy candidate table a split reads, and whether its keyword provenance is
exact-controlled, is decided by :mod:`demap_repro.reranker.split_routing`:
caDSR-derived splits (Test, CCTG, OID ALT, CDASH and the internal validation
splits) read the ``a070`` tables and get the 0.70 gold-scoped exact mask;
GDC and CIMAC read the ``a10`` tables and are never masked. Getting this wrong
does not fail loudly — it silently inflates three of the six reported datasets —
so the routing is asserted in ``tests/tier1_invariants/test_split_routing.py``
and its effect is verified against the published feature table in
``tests/tier3_regression/test_fixed_k_feature_table.py``.

Migration note
--------------
Copied from ``scripts/build_fixed_k_feature_table.py`` (research repository,
worktree sha256 ``68de531aa68b4f83d1b7…``; the worktree version, which is the one
that ran, differs from HEAD by adding ``--hydration-policy``). Removed here:

* ``--add-seqkw`` and ``--add-kw-tier3``. Both are development variants that were
  evaluated and rejected, appear nowhere in the manuscript, and were **not**
  passed by the paper build (``chain_D_stepG_v2.sbatch`` runs with
  ``--biencoder-k 20 --keyword-fuzzy-k 10`` and no other feature flags).
* ``--hydration-policy``. It exists to register splits absent from
  ``DEFAULT_HYDRATION_POLICY``; all eight paper splits are registered, and the
  paper build passed no override.

The retained logic is unchanged; ``tests/tier3_regression/test_stage_f_source_parity.py``
pins it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from demap_repro.pool.candidate_union import (
    build_candidate_union_publicid,
    _load_split_meta,
    _infer_winner_id_from_path as _infer_winner_id,
)
from demap_repro.reranker import base_features as B
from demap_repro.reranker.features.lexical_features import kw_feature_columns
from demap_repro.reranker.split_routing import (
    CADSR_ALLOWANCE,
    fuzzy_table_filename,
    is_cadsr_derived,
)

DEFAULT_BIENCODER_K = 20
DEFAULT_KEYWORD_FUZZY_K = 10
DEFAULT_KEYWORD_TOP_K_PER_RULE = 500
DEFAULT_KEYWORD_ALLOW_SEED = 42


def fuzzy_path(split: str, cadsr_dir: Path, ext_dir: Path) -> Path:
    """Route to the correct allow-policy fuzzy table for ``split``."""
    base = cadsr_dir if is_cadsr_derived(split) else ext_dir
    return base / fuzzy_table_filename(split)


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", required=True, help="comma-separated split names")
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--cde-master", required=True,
                    help="production catalog (PV/text features + version map)")
    ap.add_argument("--rankings", required=True,
                    help="bi-encoder LONG rankings parquet (split, query_id, cde_id, "
                         "biencoder_rank, biencoder_score)")
    ap.add_argument("--keyword-index", required=True,
                    help="keyword index joblib for kw_* features. The POOL keyword source "
                         "is the fuzzy tables, NOT this index.")
    ap.add_argument("--fuzzy-dir-cadsr", required=True,
                    help="dir of CDE Match-Fuzzy a0.70 candidate parquets (caDSR splits)")
    ap.add_argument("--fuzzy-dir-external", required=True,
                    help="dir of CDE Match-Fuzzy a1.0 candidate parquets (external splits)")
    ap.add_argument("--biencoder-k", type=int, default=DEFAULT_BIENCODER_K)
    ap.add_argument("--keyword-fuzzy-k", type=int, default=DEFAULT_KEYWORD_FUZZY_K)
    ap.add_argument("--keyword-top-k-per-rule", type=int,
                    default=DEFAULT_KEYWORD_TOP_K_PER_RULE,
                    help="top-K per (rule, field) when regenerating keyword provenance")
    ap.add_argument("--keyword-allow-rate", type=float, default=CADSR_ALLOWANCE,
                    help="kw_* provenance exact-rule allow gate (caDSR-derived splits only). "
                         "Must match the fuzzy a0.70 tables so kw_* features are post-policy.")
    ap.add_argument("--keyword-allow-seed", type=int, default=DEFAULT_KEYWORD_ALLOW_SEED)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble the pool + audit; skip feature computation / write")
    return ap


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    splits_dir = Path(args.splits_dir)
    rankings_path = Path(args.rankings)
    cde_master_path = Path(args.cde_master)
    keyword_index_path = Path(args.keyword_index)
    cadsr_dir = Path(args.fuzzy_dir_cadsr)
    ext_dir = Path(args.fuzzy_dir_external)
    out_path = Path(args.out)

    if out_path.exists() and not args.overwrite and not args.dry_run:
        sys.exit(f"ERROR: {out_path} exists; pass --overwrite to replace.")

    # --- preflight ----------------------------------------------------------
    for f in (rankings_path, cde_master_path, keyword_index_path):
        if not f.exists():
            sys.exit(f"ERROR: missing input: {f}")
    for s in splits:
        fp = fuzzy_path(s, cadsr_dir, ext_dir)
        if not fp.exists():
            sys.exit(f"ERROR: missing fuzzy table for split {s!r}: {fp}")
        if not (splits_dir / f"{s}.parquet").exists():
            sys.exit(f"ERROR: missing split parquet: {splits_dir / f'{s}.parquet'}")

    print("=== build fixed-K feature table ===")
    print(f"  rankings     : {rankings_path}")
    print(f"  cde master   : {cde_master_path}")
    print(f"  keyword index: {keyword_index_path}")
    print(f"  fuzzy caDSR  : {cadsr_dir} (a0.70)")
    print(f"  fuzzy extern : {ext_dir} (a1.0)")
    print(f"  K = {args.biencoder_k} MPNet + {args.keyword_fuzzy_k} CDE Match-Fuzzy "
          f"= {args.biencoder_k + args.keyword_fuzzy_k} (dedup by public id)")
    print(f"  splits       : {splits}")
    print(f"  OUT          : {out_path}")

    winner = _infer_winner_id(rankings_path)
    rankings_df = pd.read_parquet(rankings_path)
    import joblib
    kidx = joblib.load(keyword_index_path)
    print(f"[keyword index] CDEs: {kidx.n_cde}")
    cde_master_small = pd.read_parquet(cde_master_path, columns=["cde_publicid", "cde_version"])
    split_meta = _load_split_meta(splits_dir, splits)

    unions, kw_feats_all, exact_sets = [], [], []
    for split in splits:
        be = B.load_biencoder_long(rankings_df, split, args.biencoder_k, winner)
        if be.empty:
            sys.exit(f"ERROR: split {split!r} produced 0 bi-encoder rows (check rankings).")

        # CDE Match-Fuzzy candidate table for this split (post-policy a0.70 / a1.0).
        fz = pd.read_parquet(fuzzy_path(split, cadsr_dir, ext_dir))
        fz["query_id"] = fz["query_id"].astype(str)
        fz["cde_id"] = fz["cde_id"].astype(str)
        fz["cde_publicid"] = fz["cde_id"].str.split("::").str[0]
        # POOL keyword source = fuzzy top-K by cdematch_rank.
        fz_topk = fz[fz["cdematch_rank"] <= args.keyword_fuzzy_k].copy()
        # decompose into clone tier (-> in_cdematch_topk) and fuzzy-fallback tier
        # (-> in_keyword_topk)
        cm = fz_topk[fz_topk["cand_source"] == "clone"][
            ["query_id", "cde_id", "cdematch_rank", "cdematch_score"]].copy()
        kw = fz_topk[fz_topk["cand_source"] == "fuzzy_fallback"][["query_id", "cde_id"]].copy()

        # POST-POLICY exact tier: exact rows of the full fuzzy table (allow-gate already
        # applied for caDSR a0.70; full set for external a1.0).
        ex = fz[fz["mtch_type"] == "exact"][["query_id", "cde_publicid"]].drop_duplicates()
        ex["split"] = split
        exact_sets.append(ex)

        u = build_candidate_union_publicid(
            be=be, cm=cm, kw=(kw if len(kw) else None),
            split_meta=split_meta[split_meta["split"] == split],
            cde_master=cde_master_small, winner_id=winner,
            hydration_policy=None)
        unions.append(u)

        # kw_* features (schema parity): full provenance, post-policy, attached by publicid.
        queries = pd.read_parquet(splits_dir / f"{split}.parquet")
        prov = B.keyword_provenance(queries, kidx,
                                    top_k_per_rule=args.keyword_top_k_per_rule)
        # gold-by-query (pre-dedup; multi-gold covered) for the shared gold-scoped
        # exact-match mask -- the same gate the clone arm uses.
        gold_by_qid = {}
        if "cde_id" in queries.columns:
            for qid, g in zip(queries["query_id"].astype(str), queries["cde_id"].astype(str)):
                gold_by_qid.setdefault(qid, set()).add(g)
        prov, n_drop = B.apply_keyword_exact_control(
            prov, split, args.keyword_allow_rate, args.keyword_allow_seed, gold_by_qid)
        from demap_repro.reranker.features.lexical_features import build_kw_features
        kwf = (build_kw_features(prov) if len(prov)
               else pd.DataFrame(columns=["query_id", "cde_id"] + kw_feature_columns()))
        if len(kwf):
            kw_feats_all.append(kwf)
        print(f"  {split}: {len(u)} pool rows, {u['query_id'].nunique()} queries "
              f"(be{args.biencoder_k}+fuzzy{args.keyword_fuzzy_k}; clone_cands={len(cm)}, "
              f"fuzzy_cands={len(kw)}, exact_pairs={len(ex)}"
              f"{f', kw exact-gated -{n_drop}' if n_drop else ''})")

    union = pd.concat(unions, ignore_index=True)
    print(f"\nCombined pool: {len(union)} rows, {union['query_id'].nunique()} queries")
    print("Gold injection DISABLED (deployable fixed-K pool).")

    # Multi-gold correctness. build_candidate_union_publicid labels only the FIRST
    # gold public id per query_id, and _load_split_meta itself keeps one gold per
    # query. Re-derive is_label from the RAW split parquets (ALL gold rows) so a
    # candidate counts as positive if its public id matches ANY of the query's
    # golds -- the "any gold" convention used by K-selection and by every reported
    # recall number.
    gold_keys = set()
    for s in splits:
        g = pd.read_parquet(splits_dir / f"{s}.parquet", columns=["query_id", "cde_id"])
        gold_keys.update(zip(
            [s] * len(g),
            g["query_id"].astype(str),
            g["cde_id"].astype(str).str.split("::").str[0]))
    before = int(union["is_label"].sum())
    union["is_label"] = [
        (s, q, p) in gold_keys
        for s, q, p in zip(union["split"].astype(str),
                           union["query_id"].astype(str),
                           union["cde_publicid"].astype(str))
    ]
    after = int(union["is_label"].sum())
    print(f"is_label recomputed over full gold-public-id set (multi-gold): "
          f"{before} -> {after} positive rows (+{after - before})")

    # exact-tier flag (post-policy), keyed by (split, query_id, cde_publicid).
    exact_df = pd.concat(exact_sets, ignore_index=True).drop_duplicates()
    exact_keys = set(map(tuple, exact_df[["split", "query_id", "cde_publicid"]].astype(str).values))

    if args.dry_run:
        issues = B.audit_feature_table(union)
        summ = B.feature_summary(union)
        print("\n=== DRY-RUN ===")
        print(f"  rows={summ['total_rows']} splits={summ['splits']}")
        print(f"  candidates/query={summ['candidates_per_query']}")
        print(f"  positive_rate={summ['positive_rate']:.4f} "
              f"queries_with_positive={summ['queries_with_positive']}/{summ['total_queries']}")
        n_exact = sum(1 for r in union[["split", "query_id", "cde_publicid"]].astype(str).values
                      if tuple(r) in exact_keys)
        print(f"  is_exact_candidate rows (in pool): {n_exact}")
        print(f"  audit: {issues if issues else 'PASS'}")
        return 0

    print("\nComputing features...")
    cde_master_full = pd.read_parquet(cde_master_path)
    union = B.compute_all_features(union, cde_master_full)

    # attach kw_* features by (query_id, cde_publicid)
    print("  Attaching kw_* keyword features...")
    if kw_feats_all:
        kw_feats = pd.concat(kw_feats_all, ignore_index=True)
        kw_feats["cde_publicid"] = kw_feats["cde_id"].astype(str).str.split("::").str[0]
        kw_feats = kw_feats.drop(columns=["cde_id"]).drop_duplicates(["query_id", "cde_publicid"])
        union = union.merge(kw_feats, on=["query_id", "cde_publicid"], how="left")
    for c in kw_feature_columns():
        if c not in union.columns:
            union[c] = np.nan
        union[c] = pd.to_numeric(union[c], errors="coerce").astype("float64")

    # exact-tier metadata column. NOT a model feature -- training excludes it.
    union["is_exact_candidate"] = [
        tuple(r) in exact_keys
        for r in union[["split", "query_id", "cde_publicid"]].astype(str).values
    ]
    n_exact = int(union["is_exact_candidate"].sum())
    print(f"    in_keyword_topk rows={int((union['in_keyword_topk'] == True).sum())}; "
          f"is_exact_candidate rows={n_exact}")

    # The merged table carries is_injected_gold=False so downstream CE-coverage
    # checks have the column; the fixed-K pool never injects gold.
    if "is_injected_gold" not in union.columns:
        union["is_injected_gold"] = False

    issues = B.audit_feature_table(union)
    if issues:
        print(f"  audit warnings: {issues}")

    summary = B.feature_summary(union)
    print(f"\nFeature table: {summary['total_rows']} rows, "
          f"{summary['total_feature_cols']} features")
    print("  candidates/query: "
          + ", ".join(f"{s}={v['mean']:.1f}" for s, v in summary["candidates_per_query"].items()))
    print(f"  positive_rate={summary['positive_rate']:.4f} "
          f"queries_with_positive={summary['queries_with_positive']}/{summary['total_queries']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    union.to_parquet(out_path, index=False)
    print(f"\nWrote feature table: {out_path}")

    summary["provenance"] = {
        "build": "fixed_k",
        "cde_master": str(cde_master_path),
        "splits_dir": str(splits_dir),
        "rankings": str(rankings_path),
        "keyword_index": str(keyword_index_path),
        "fuzzy_dir_cadsr": str(cadsr_dir),
        "fuzzy_dir_external": str(ext_dir),
        "biencoder_k": args.biencoder_k,
        "keyword_fuzzy_k": args.keyword_fuzzy_k,
        "k_total": args.biencoder_k + args.keyword_fuzzy_k,
        "keyword_allow_rate_cadsr": args.keyword_allow_rate,
        "keyword_allow_seed": args.keyword_allow_seed,
        "allow_policy": "caDSR-derived -> fuzzy a0.70; external/non-caDSR -> fuzzy a1.0",
        "split_routing": {s: ("cadsr_a070" if is_cadsr_derived(s) else "external_a10")
                          for s in splits},
        "inject_gold": False,
        "publicid_union": True,
        "is_exact_candidate_rows": n_exact,
        "exact_tier": "post-policy clone/fuzzy mtch_type=='exact'",
        "kw_feature_cols": len(kw_feature_columns()),
        "source_flags": ["in_biencoder_topk", "in_cdematch_topk", "in_keyword_topk"],
        "metadata_nonfeature": ["is_exact_candidate"],
    }
    summary_path = out_path.parent / "feature_table_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
