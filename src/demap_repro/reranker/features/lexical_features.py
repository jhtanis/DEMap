"""Convert Stage-A keyword provenance into numeric kw_* reranker features.

Input: long provenance from ``keyword_retriever.generate_candidates``
(``query_id, cde_id, rule, field, rule_rank, rule_score``).

Output: a wide, all-numeric feature frame keyed by ``(query_id, cde_id)`` with
RRF fusion, rank/hit aggregates, per-(rule,field) scores, and cross-field
aggregates. No string columns; no collisions with HGBC FORBIDDEN_FEATURE_COLS.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

# Rule weights for RRF-independent "best rule" selection + a single keyword_score.
RULE_WEIGHT: Dict[str, float] = {
    "exact": 1.0, "rev_containment": 0.85, "containment": 0.80,
    "pv": 0.75, "charngram": 0.70, "wordngram": 0.68, "token": 0.60,
}
RULE_CODES: Dict[str, int] = {
    "exact": 1, "rev_containment": 2, "containment": 3, "pv": 4,
    "charngram": 5, "wordngram": 6, "token": 7,
}
FIELD_CODES: Dict[str, int] = {
    "SHORT_NAME": 1, "LONG_NAME": 2, "DEC_LONG_NAME": 3,
    "PREFERRED_QUESTION_TEXT": 4, "DEFINITION": 5, "PV_SUMMARY": 6,
    "SHORT_NAME@query_field": 7,
}
RRF_K0 = 60

# Per-(rule, field) score columns we expose (stable schema across splits).
_PER_FIELD = {
    "charngram": ["SHORT_NAME", "LONG_NAME", "DEC_LONG_NAME", "PREFERRED_QUESTION_TEXT"],
    "wordngram": ["SHORT_NAME", "LONG_NAME", "DEC_LONG_NAME", "PREFERRED_QUESTION_TEXT"],
    "token": ["SHORT_NAME", "LONG_NAME", "DEC_LONG_NAME", "PREFERRED_QUESTION_TEXT", "DEFINITION"],
    "exact": ["SHORT_NAME", "LONG_NAME", "DEC_LONG_NAME", "PREFERRED_QUESTION_TEXT", "SHORT_NAME@query_field"],
    "containment": ["SHORT_NAME", "LONG_NAME", "DEC_LONG_NAME", "PREFERRED_QUESTION_TEXT", "DEFINITION"],
    "rev_containment": ["SHORT_NAME", "LONG_NAME", "DEC_LONG_NAME", "PREFERRED_QUESTION_TEXT", "DEFINITION"],
    "pv": ["PV_SUMMARY"],
}


def _col(rule: str, field: str) -> str:
    return f"kw_{rule}_{field.replace('@', '_at_')}"


def kw_feature_columns() -> List[str]:
    """Stable, ordered list of all kw_* feature columns (for schema alignment)."""
    cols = ["rrf_score", "keyword_rank", "n_rule_hits", "n_distinct_rules",
            "best_rule_code", "best_field_code",
            "kw_charngram_max", "kw_charngram_mean", "kw_wordngram_max",
            "kw_token_max", "kw_any_exact", "kw_pv_score"]
    for rule, fields in _PER_FIELD.items():
        for f in fields:
            cols.append(_col(rule, f))
    return cols


def per_source_ranks(prov: pd.DataFrame) -> pd.DataFrame:
    """Per (query_id, cde_id, rule): best score + within-(query,rule) rank.

    Deterministic ordering contract (see
    ``keyword_retriever.KEYWORD_TIE_BREAK_POLICY_VERSION``): ranks are assigned
    by score DESCENDING then ``cde_id`` ASCENDING, so equal scores never fall
    back to incidental row order.
    """
    g = (prov.groupby(["query_id", "cde_id", "rule"], sort=False)["rule_score"]
              .max().reset_index())
    g = g.sort_values(["query_id", "rule", "rule_score", "cde_id"],
                      ascending=[True, True, False, True],
                      kind="stable").reset_index(drop=True)
    g["src_rank"] = g.groupby(["query_id", "rule"], sort=False).cumcount() + 1
    return g


def rrf_fuse(src: pd.DataFrame, k0: int = RRF_K0) -> pd.DataFrame:
    """Reciprocal-rank-fuse per-source ranks -> (query_id, cde_id, rrf_score, keyword_rank)."""
    s = src.copy()
    s["rr"] = 1.0 / (k0 + s["src_rank"])
    fused = s.groupby(["query_id", "cde_id"], sort=False)["rr"].sum().reset_index(name="rrf_score")
    # Deterministic: fused score DESCENDING then cde_id ASCENDING.
    fused = fused.sort_values(["query_id", "rrf_score", "cde_id"],
                              ascending=[True, False, True],
                              kind="stable").reset_index(drop=True)
    fused["keyword_rank"] = fused.groupby("query_id", sort=False).cumcount() + 1
    return fused


def build_kw_features(prov: pd.DataFrame, k0: int = RRF_K0) -> pd.DataFrame:
    """Wide all-numeric kw_* feature frame keyed by (query_id, cde_id)."""
    if prov.empty:
        return pd.DataFrame(columns=["query_id", "cde_id"] + kw_feature_columns())
    prov = prov.copy()
    prov["w"] = (prov["rule"].map(RULE_WEIGHT).fillna(0.5).to_numpy()
                 * prov["rule_score"].clip(0, 1).to_numpy())

    # per-(rule,field) max score -> wide pivot
    gf = (prov.groupby(["query_id", "cde_id", "rule", "field"], sort=False)["rule_score"]
              .max().reset_index())
    gf["col"] = [_col(r, f) for r, f in zip(gf["rule"], gf["field"])]
    wide = gf.pivot_table(index=["query_id", "cde_id"], columns="col",
                          values="rule_score", aggfunc="max")
    wide = wide.reindex(columns=[_col(r, f) for r, fs in _PER_FIELD.items() for f in fs])

    # aggregates / fusion / provenance summary
    pair = prov.groupby(["query_id", "cde_id"], sort=False)
    agg = pd.DataFrame({
        "n_rule_hits": pair.size(),
        "n_distinct_rules": pair["rule"].nunique(),
    })
    # best rule/field by weighted contribution. Deterministic: weight DESCENDING,
    # then rule code and field code ASCENDING (both explicit, documented
    # priorities), so a tie never resolves by incidental row order.
    _best_src = prov.assign(
        _rule_code=prov["rule"].map(RULE_CODES).fillna(99).astype(int),
        _field_code=prov["field"].map(FIELD_CODES).fillna(99).astype(int),
    ).sort_values(["query_id", "cde_id", "w", "_rule_code", "_field_code"],
                  ascending=[True, True, False, True, True], kind="stable")
    best = (_best_src.drop_duplicates(["query_id", "cde_id"])
                     .set_index(["query_id", "cde_id"])[["rule", "field"]])
    agg["best_rule_code"] = best["rule"].map(RULE_CODES).astype("Int64")
    agg["best_field_code"] = best["field"].map(FIELD_CODES).astype("Int64")

    out = wide.join(agg, how="outer")
    # cross-field aggregates
    cg = [c for c in (_col("charngram", f) for f in _PER_FIELD["charngram"]) if c in out]
    wg = [c for c in (_col("wordngram", f) for f in _PER_FIELD["wordngram"]) if c in out]
    tk = [c for c in (_col("token", f) for f in _PER_FIELD["token"]) if c in out]
    ex = [c for c in (_col("exact", f) for f in _PER_FIELD["exact"]) if c in out]
    out["kw_charngram_max"] = out[cg].max(axis=1) if cg else np.nan
    out["kw_charngram_mean"] = out[cg].mean(axis=1) if cg else np.nan
    out["kw_wordngram_max"] = out[wg].max(axis=1) if wg else np.nan
    out["kw_token_max"] = out[tk].max(axis=1) if tk else np.nan
    out["kw_any_exact"] = (out[ex].max(axis=1) > 0).astype(float) if ex else 0.0
    out["kw_pv_score"] = out[_col("pv", "PV_SUMMARY")] if _col("pv", "PV_SUMMARY") in out else np.nan

    fused = rrf_fuse(per_source_ranks(prov), k0).set_index(["query_id", "cde_id"])
    out = out.join(fused, how="left")

    out = out.reset_index()
    # enforce stable, all-numeric schema
    for c in kw_feature_columns():
        if c not in out.columns:
            out[c] = np.nan
    keep = ["query_id", "cde_id"] + kw_feature_columns()
    out = out[keep]
    for c in kw_feature_columns():
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    # Canonical emitted row order: (query_id, cde_id) ASCENDING. Callers collapse
    # versions of a public id with drop_duplicates(..., keep="first"), so pinning
    # this order here makes that collapse deterministic for every caller without
    # each one repeating the rule. (pivot_table already produced this order; the
    # explicit sort turns an emergent property into a contract.)
    out = out.sort_values(["query_id", "cde_id"], kind="stable").reset_index(drop=True)
    return out
