"""Shared feature-assembly helpers for the fixed-K candidate table.

Migrated from ``scripts/build_hgbc_feature_table.py`` (research repository,
tracked and clean at HEAD ``cf8743f``, worktree sha256
``1bb38ad3a96fe5dd8aef…``). That script carried two paths: a legacy "wide"
top-1000 builder with gold injection, driven by its own ``main()``, and a set of
helpers that :mod:`demap_repro.reranker.fixed_k_features` imports. **Only the
helpers reached the paper.** The legacy ``main()``, its v3 constants, the gold
injection routine and the CIMAC/wide CDE Match loaders are not migrated: the
final pool is deployable and injects no gold.

Function bodies below are unchanged from the source. The differences are:

* ``_is_cadsr_derived`` and the ``_CADSR_DERIVED_SPLITS`` frozenset moved to
  :mod:`demap_repro.reranker.split_routing`, where the routing decision is
  documented and independently testable;
* the private ``_``-prefixed names are public here, because they are the module's
  actual interface;
* ``from demap.features.cde_match_clone import ExactMatchControl`` is dropped —
  it was imported but never referenced;
* the keyword-provenance generator, which remains gated, is resolved through
  :mod:`demap_repro.lexical.cde_match_interface` instead of imported directly.
  ``compute_cdematch_features`` was reclassified as independent on 2026-08-08 (it
  is generic rank/score arithmetic over an already-produced candidate list) and is
  imported normally.

``tests/tier3_regression/test_stage_f_source_parity.py`` pins the abstract syntax
tree of every function here against the hash it had in the source, so a later
edit cannot change behaviour unnoticed.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from demap_repro.lexical import cde_match_interface
from demap_repro.lexical import mask as emm
from demap_repro.pool.candidate_union import FORBIDDEN_LEAKAGE_COLUMNS
from demap_repro.reranker.features.biencoder import compute_biencoder_features
from demap_repro.reranker.features.cdematch import (
    CDEMATCH_FEATURE_COLUMNS,
    compute_cdematch_features,
)
from demap_repro.reranker.features.lexical_features import (
    build_kw_features,
    kw_feature_columns,
)
from demap_repro.reranker.features.pv_overlap import (
    compute_pv_overlap_for_union,
    feature_columns as pv_feature_columns,
)
from demap_repro.reranker.features.text_features import (
    compute_text_features,
    text_feature_columns,
)
from demap_repro.reranker.split_routing import is_cadsr_derived

__all__ = [
    "METADATA_COLS",
    "FORBIDDEN_FEATURE_COLS",
    "load_biencoder_long",
    "keyword_provenance",
    "apply_keyword_exact_control",
    "compute_all_features",
    "audit_feature_table",
    "feature_summary",
]


#: Columns carried through the table that are identity/context, not model inputs.
METADATA_COLS = [
    "split", "query_id", "cde_id", "pair_id", "family", "query_source",
    "is_label", "in_biencoder_topk", "in_cdematch_topk",
]

#: Columns the reranker must never train on. Anything derived from the gold, the
#: split identity, or the raw query/PV text would leak the answer.
FORBIDDEN_FEATURE_COLS = FORBIDDEN_LEAKAGE_COLUMNS | {
    "is_label", "is_injected_gold", "cde_id", "query_id", "pair_id",
    "split", "winner_id", "query_text_q3", "PV_BLOCK_SDE", "pv_attached",
    "hydration_policy",
}


def load_biencoder_long(rankings_df: pd.DataFrame, split: str, K: int,
                        winner_id: str) -> pd.DataFrame:
    """Bi-encoder candidates for one split from the LONG rankings parquet
    (columns: split, query_id, cde_id, biencoder_rank, biencoder_score). Capped at
    rank<=K.

    The long rankings carry LITERAL split names (``retrieve_biencoder_deep``
    writes ``split=split``), so filter on the literal name -- NOT through any
    legacy alias map, whose ``val_dev``/``val_train`` -> ``natural_internal_eval``
    rewrite (for the old array-format rankings) would silently drop the
    train/tune splits here.
    """
    r = rankings_df[(rankings_df["split"] == split) &
                    (rankings_df["biencoder_rank"] <= K)].copy()
    if r.empty:
        return pd.DataFrame(columns=["winner_id", "split", "query_id", "cde_id",
                                     "biencoder_rank", "biencoder_score", "in_biencoder_topk"])
    r["split"] = split
    r["winner_id"] = winner_id
    r["query_id"] = r["query_id"].astype(str)
    r["cde_id"] = r["cde_id"].astype(str)
    r["biencoder_rank"] = r["biencoder_rank"].astype("Int64")
    r["biencoder_score"] = r["biencoder_score"].astype(float)
    r["in_biencoder_topk"] = True
    return r[["winner_id", "split", "query_id", "cde_id",
              "biencoder_rank", "biencoder_score", "in_biencoder_topk"]]


def keyword_provenance(queries: pd.DataFrame, keyword_index, *, top_k_per_rule: int,
                       query_col: str = "query_text_q3") -> pd.DataFrame:
    """Regenerate the LONG keyword provenance (query_id, cde_id, rule, field,
    rule_rank, rule_score) over the production keyword index. Uses the full rule
    set (not the collapsed keyword_rankings.parquet). Deployment-safe: the
    retriever fits only on the public CDE index; no gold is read.

    Requires the gated CDE Match-Fuzzy retriever; see
    :mod:`demap_repro.lexical.cde_match_interface`.
    """
    qcol = query_col if query_col in queries.columns else "query_text_q3"
    return cde_match_interface.generate_candidates(
        queries, keyword_index,
        top_k_per_rule=top_k_per_rule,
        query_text_col=qcol,
        strip_query_pv_block=True,
        use_word_ngram=True,
        include_containment=False,
        verbose=False,
    )


def apply_keyword_exact_control(prov: pd.DataFrame, split: str, allow_rate: float,
                                seed: int, gold_by_qid=None):
    """Keyword exact-rule leakage control — the SHARED gold-scoped mask.

    On caDSR-derived splits, when ``allow_rate < 1.0``, drop the keyword ``exact``
    rule rows of the deterministically-BLOCKED fraction of queries, using the same
    shared nested mask as the clone (``mask.query_allowed``). The suppression is
    **gold-scoped** (mask semantics: the mask controls whether the GOLD CDE's
    exact-matching rows are eligible): only rows whose candidate matches the
    query's gold CDE — exact ``cde_id`` for versioned golds, public-id for
    unversioned golds — are dropped. Non-gold exact rows, all fuzzy rows
    (charngram/token/wordngram/pv), and non-caDSR splits are never suppressed.

    ``gold_by_qid`` maps ``query_id -> set of gold cde_ids``; without it the
    control cannot scope to gold and raises (silent all-exact suppression was the
    pre-v13 divergence between the two lexical methods).

    Returns (prov, n_exact_rows_dropped).
    """
    if (prov is None or len(prov) == 0 or allow_rate >= 1.0
            or not is_cadsr_derived(split)):
        return prov, 0
    is_exact = prov["rule"].to_numpy() == "exact"
    if not is_exact.any():
        return prov, 0
    if gold_by_qid is None:
        raise ValueError("apply_keyword_exact_control requires gold_by_qid for "
                         "gold-scoped suppression (shared exact_match_mask_v1)")
    qids = prov["query_id"].astype(str)
    blocked = qids.map(lambda q: not emm.query_allowed(q, allow_rate, seed)).to_numpy()

    cand = prov["cde_id"].astype(str)
    cand_pub = cand.str.split("::").str[0]

    def _gold_sets(q):
        golds = {str(g) for g in gold_by_qid.get(q, set())
                 if g is not None and str(g) not in ("", "nan", "None")}
        ver = frozenset(g for g in golds if "::" in g)
        pub = frozenset(g.split("::")[0] for g in golds if "::" not in g)
        return ver, pub

    gold_cache = {q: _gold_sets(q) for q in set(qids[blocked & is_exact])}
    is_gold_cand = np.fromiter(
        ((cand.iat[i] in gold_cache[qids.iat[i]][0]
          or cand_pub.iat[i] in gold_cache[qids.iat[i]][1])
         if (blocked[i] and is_exact[i]) else False
         for i in range(len(prov))), dtype=bool, count=len(prov))
    drop = is_exact & blocked & is_gold_cand
    n = int(drop.sum())
    return prov.loc[~drop].reset_index(drop=True), n


def compute_all_features(union: pd.DataFrame, cde_master: pd.DataFrame) -> pd.DataFrame:
    """Bi-encoder, CDE Match, PV-overlap and text features, in that order."""
    print("  Computing bi-encoder features...")
    union = compute_biencoder_features(union)

    print("  Computing CDE Match features...")
    union = compute_cdematch_features(union)

    print("  Computing PV-overlap features...")
    pv_feats = compute_pv_overlap_for_union(union, cde_master=cde_master)
    union = pd.concat([union, pv_feats], axis=1)

    print("  Computing text features...")
    union = compute_text_features(union, cde_master=cde_master)

    return union


def audit_feature_table(df: pd.DataFrame) -> Dict:
    """Fail closed on leakage columns; report queries without a positive and
    duplicate candidate keys."""
    bad_cols = sorted(set(df.columns) & FORBIDDEN_LEAKAGE_COLUMNS)
    if bad_cols:
        raise RuntimeError(f"Leakage columns in feature table: {bad_cols}")

    issues = {}

    per_query = df.groupby(["split", "query_id"])
    no_positive = per_query["is_label"].sum() == 0
    if no_positive.any():
        count = int(no_positive.sum())
        issues["queries_without_positive"] = count

    dups = df.duplicated(subset=["split", "query_id", "cde_id"], keep=False)
    if dups.any():
        issues["duplicate_rows"] = int(dups.sum())

    return issues


def feature_summary(df: pd.DataFrame) -> Dict:
    """Row/feature/missingness summary written alongside the feature table."""
    from demap_repro.reranker.features.biencoder import biencoder_feature_columns

    bienc_cols = list(biencoder_feature_columns())
    cm_cols = list(CDEMATCH_FEATURE_COLUMNS)
    pv_cols = pv_feature_columns()
    text_cols = list(text_feature_columns(include_tier2=True, include_tier3=True))

    all_feature_cols = bienc_cols + cm_cols + pv_cols + text_cols + [
        "in_biencoder_topk", "in_cdematch_topk",
    ]
    # The public-id union path adds keyword evidence as model features.
    if "in_keyword_topk" in df.columns:
        all_feature_cols = all_feature_cols + ["in_keyword_topk"] + list(kw_feature_columns())
    present = [c for c in all_feature_cols if c in df.columns]
    missing_from_df = [c for c in all_feature_cols if c not in df.columns]

    missingness = {}
    for c in present:
        if df[c].dtype == object:
            na_count = (df[c].isna() | (df[c] == "")).sum()
        else:
            na_count = df[c].isna().sum()
        if na_count > 0:
            missingness[c] = int(na_count)

    return {
        "total_rows": len(df),
        "total_feature_cols": len(present),
        "missing_feature_cols": missing_from_df,
        "feature_missingness": missingness,
        "splits": df["split"].value_counts().to_dict(),
        "candidates_per_query": {
            split: {
                "mean": float(g.groupby("query_id").size().mean()),
                "min": int(g.groupby("query_id").size().min()),
                "max": int(g.groupby("query_id").size().max()),
            }
            for split, g in df.groupby("split")
        },
        "positive_rate": float(df["is_label"].mean()),
        "queries_with_positive": int(
            df.groupby(["split", "query_id"])["is_label"].any().sum()
        ),
        "total_queries": int(df.groupby(["split", "query_id"]).ngroups),
    }
