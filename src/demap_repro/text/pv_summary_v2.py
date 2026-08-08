from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from demap_repro.text.normalize import normalize_query_text


GENERIC_MEANINGS: Dict[str, Sequence[str]] = {
    "UNKNOWN": ("unknown", "unk", "not known", "don't know", "do not know"),
    "NA": ("not applicable", "n/a", "na", "notapplicable"),
    "NOT_REPORTED": ("not reported", "notreported"),
    "NOT_TESTED": ("not tested", "nottested"),
    "YES": ("yes", "true"),
    "NO": ("no", "false"),
    "NORMAL": ("normal",),
    "ABNORMAL": ("abnormal",),
    "POSITIVE": ("positive", "pos"),
    "NEGATIVE": ("negative", "neg"),
    "PRESENT": ("present",),
    "ABSENT": ("absent",),
}
GENERIC_LOOKUP: Dict[str, str] = {}
for concept, vals in GENERIC_MEANINGS.items():
    for v in vals:
        GENERIC_LOOKUP[normalize_query_text(v).casefold()] = concept

NUMERIC_RE = __import__("re").compile(r"^[+-]?(?:\d+)(?:\.\d+)?$")


@dataclass(frozen=True)
class CanonicalPVRecord:
    valid_value: str
    preferred_meaning_label: str
    generic_concept: Optional[str]
    generic_bucket_key: Optional[str]
    order_rank: float
    input_order: int

    @property
    def stable_key(self) -> str:
        return f"{self.generic_bucket_key or ''}::{self.valid_value}::{self.preferred_meaning_label}::{self.order_rank}::{self.input_order}"


@dataclass
class CanonicalPVPool:
    cde_publicid: str
    cde_version: str
    records: List[CanonicalPVRecord]
    generic_buckets: Dict[str, List[CanonicalPVRecord]]
    non_generic_records: List[CanonicalPVRecord]

    @property
    def n(self) -> int:
        return len(self.records)

    @property
    def m(self) -> int:
        return len(self.generic_buckets)

    @property
    def generic_dominant(self) -> bool:
        return self.n > 0 and (float(self.m) / float(self.n)) >= 0.5


def _norm_text(x: Any) -> str:
    return normalize_query_text(x)


def _cf(x: Any) -> str:
    return _norm_text(x).casefold()


def _dedupe_preserve_order(xs: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _sha1_int(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)


def _hash_key(*parts: object) -> int:
    return _sha1_int("::".join(str(x) for x in parts))


def generic_concept(value_meaning: str) -> Optional[str]:
    return GENERIC_LOOKUP.get(_cf(value_meaning))


def preferred_meaning_label_from_row(row: Mapping[str, Any]) -> str:
    for col in ["value_meaning_long_name", "value_meaning", "meaning_description"]:
        if col in row:
            text = _norm_text(row.get(col, ""))
            if text:
                return text
    return ""


def pv_type_signature(value_meanings: Sequence[str]) -> str:
    vals = _dedupe_preserve_order([_cf(v) for v in value_meanings if _norm_text(v)])
    s = set(vals)
    if not s:
        return "EMPTY"

    yes_set = {"yes", "true"}
    no_set = {"no", "false"}
    unknown_set = {"unknown", "unk", "not known", "don't know", "do not know"}
    na_set = {"not applicable", "n/a", "na", "notapplicable"}
    not_rep_set = {"not reported", "notreported"}
    not_test_set = {"not tested", "nottested"}
    allowed_binary = yes_set | no_set | unknown_set | na_set | not_rep_set | not_test_set

    has_yes = any(v in yes_set for v in s)
    has_no = any(v in no_set for v in s)
    if has_yes and has_no and s.issubset(allowed_binary):
        extras: List[str] = []
        if any(v in unknown_set for v in s):
            extras.append("UNKNOWN")
        if any(v in na_set for v in s):
            extras.append("NA")
        if any(v in not_rep_set for v in s):
            extras.append("NOT_REPORTED")
        if any(v in not_test_set for v in s):
            extras.append("NOT_TESTED")
        return "BINARY" if not extras else "BINARY_WITH_" + "_".join(extras)

    likert_freq = {"never", "rarely", "sometimes", "often", "always"}
    if s == likert_freq:
        return "LIKERT_5_FREQUENCY"

    likert_agree = {"strongly disagree", "disagree", "neutral", "agree", "strongly agree"}
    if s == likert_agree:
        return "LIKERT_5_AGREEMENT"

    if all(NUMERIC_RE.fullmatch(v) for v in s):
        try:
            ints = sorted({int(float(v)) for v in s})
            if len(ints) == len(s) and ints and all(ints[i] + 1 == ints[i + 1] for i in range(len(ints) - 1)):
                return f"ORDINAL_INT_{ints[0]}_{ints[-1]}"
        except Exception:
            pass

    return "ENUM"


def canonicalize_pv_pool(df_group: pd.DataFrame, *, profile: str, config: Mapping[str, Any]) -> CanonicalPVPool:
    pid = str(df_group["cde_publicid"].iloc[0])
    ver = str(df_group["cde_version"].iloc[0])
    rows: List[CanonicalPVRecord] = []
    seen = set()
    for idx, (_, row) in enumerate(df_group.reset_index(drop=True).iterrows()):
        preferred = preferred_meaning_label_from_row(row)
        valid_value = _norm_text(row.get("valid_value", "")) if "valid_value" in row.index else ""
        if not preferred and not valid_value:
            continue
        generic_bucket_key = None
        generic = None
        explicit_generic_code = _norm_text(row.get("generic_concept_code", "")) if "generic_concept_code" in row.index else ""
        if explicit_generic_code:
            generic = explicit_generic_code
            generic_bucket_key = explicit_generic_code
        else:
            generic = generic_concept(preferred)
            if generic:
                generic_bucket_key = generic
        if "is_generic" in row.index:
            try:
                is_generic_value = bool(row.get("is_generic"))
                if not is_generic_value:
                    generic = None
                    generic_bucket_key = None
            except Exception:
                pass
        order_rank = pd.to_numeric(row.get("meaning_concept_display_order", pd.NA), errors="coerce")
        order_value = float(order_rank) if not pd.isna(order_rank) else float("inf")
        semantic_key = (_norm_text(valid_value), _norm_text(preferred))
        if semantic_key in seen:
            continue
        seen.add(semantic_key)
        rows.append(
            CanonicalPVRecord(
                valid_value=valid_value,
                preferred_meaning_label=preferred,
                generic_concept=generic,
                generic_bucket_key=generic_bucket_key,
                order_rank=order_value,
                input_order=int(idx),
            )
        )

    rows.sort(key=lambda r: (r.order_rank, r.input_order, r.preferred_meaning_label, r.valid_value))
    generic_buckets: Dict[str, List[CanonicalPVRecord]] = {}
    non_generic: List[CanonicalPVRecord] = []
    for rec in rows:
        if rec.generic_bucket_key:
            generic_buckets.setdefault(rec.generic_bucket_key, []).append(rec)
        else:
            non_generic.append(rec)
    return CanonicalPVPool(
        cde_publicid=pid,
        cde_version=ver,
        records=rows,
        generic_buckets=generic_buckets,
        non_generic_records=non_generic,
    )


def _normalize_probability(value: float | int) -> float:
    x = float(value)
    return x / 100.0 if x > 1.0 else x


def _seed_base(pool: CanonicalPVPool, *, side: str, salt: str) -> str:
    return f"{salt}::{pool.cde_publicid}::{pool.cde_version}::{side}"


def compute_side_sample_size(pool: CanonicalPVPool, *, side: str, config: Mapping[str, Any], seed: str) -> int:
    n = int(pool.n)
    if n <= 0:
        return 0
    min_n = 1 if n < 2 else int(config.get("pv_min_n", 2))
    max_cap = int(config["pv_max_n_query"] if side == "query" else config["pv_max_n_cde"])
    center_fraction = float(config.get("pv_center_fraction", 0.5))
    center = int(round(center_fraction * n))
    center = max(min_n, min(max_cap, center))
    if n >= 4:
        opts = []
        for cand in [center - 1, center, center + 1]:
            cand2 = max(min_n, min(max_cap, cand))
            if cand2 not in opts:
                opts.append(cand2)
        return int(opts[_hash_key(seed, "k") % len(opts)])
    return int(center)


def allocate_strata_counts(pool: CanonicalPVPool, *, side: str, k: int, config: Mapping[str, Any]) -> tuple[int, int]:
    if k <= 0 or pool.n <= 0:
        return 0, 0
    p_generic = float(pool.m) / float(pool.n) if pool.n > 0 else 0.0
    p_target_generic = min(p_generic + float(config.get("pv_generic_boost", 0.10)), float(config.get("pv_generic_cap", 0.70)))
    target_generic = int(round(k * p_target_generic))
    target_generic = max(0, min(target_generic, pool.m, k))
    target_non_generic = k - target_generic
    available_non_generic = len(pool.non_generic_records)
    if target_non_generic > available_non_generic:
        spill = target_non_generic - available_non_generic
        target_non_generic = available_non_generic
        target_generic = min(k - target_non_generic, pool.m)
        if spill > 0:
            target_generic = min(k - target_non_generic, pool.m)
    if target_generic > pool.m:
        target_generic = pool.m
        target_non_generic = min(k - target_generic, available_non_generic)
    if target_generic + target_non_generic < k:
        remaining = k - (target_generic + target_non_generic)
        extra_non_generic = min(remaining, available_non_generic - target_non_generic)
        target_non_generic += extra_non_generic
        remaining -= extra_non_generic
        if remaining > 0:
            target_generic = min(target_generic + remaining, pool.m)
    return int(target_generic), int(target_non_generic)


def _deterministic_select(items: Sequence[Any], *, k: int, seed: str, key_fn) -> List[Any]:
    if k <= 0:
        return []
    if len(items) <= k:
        return list(items)
    ranked = sorted(
        list(items),
        key=lambda item: (_hash_key(seed, key_fn(item)), key_fn(item)),
    )
    chosen = ranked[:k]
    return sorted(chosen, key=lambda item: key_fn(item))


def sample_side_subset(pool: CanonicalPVPool, *, side: str, k: int, config: Mapping[str, Any], seed: str) -> List[CanonicalPVRecord]:
    target_generic, target_non_generic = allocate_strata_counts(pool, side=side, k=k, config=config)
    generic_keys = _deterministic_select(
        list(pool.generic_buckets.keys()),
        k=target_generic,
        seed=seed + "::generic",
        key_fn=lambda x: str(x),
    )
    selected: List[CanonicalPVRecord] = []
    for gk in generic_keys:
        bucket = pool.generic_buckets[str(gk)]
        chosen = sorted(bucket, key=lambda r: (r.order_rank, r.input_order, r.valid_value, r.preferred_meaning_label))[0]
        selected.append(chosen)
    non_generic = _deterministic_select(
        list(pool.non_generic_records),
        k=target_non_generic,
        seed=seed + "::nongeneric",
        key_fn=lambda r: r.stable_key,
    )
    selected.extend(non_generic)
    if len(selected) < k:
        remaining_records = [r for r in pool.records if r not in selected]
        filler = _deterministic_select(
            remaining_records,
            k=k - len(selected),
            seed=seed + "::fill",
            key_fn=lambda r: r.stable_key,
        )
        selected.extend(filler)
    selected.sort(key=lambda r: (r.order_rank, r.input_order, r.preferred_meaning_label, r.valid_value))
    return selected[:k]


def render_cde_side(sampled_subset: Sequence[CanonicalPVRecord], *, config: Mapping[str, Any]) -> List[str]:
    return _dedupe_preserve_order([rec.preferred_meaning_label or rec.valid_value for rec in sampled_subset if (rec.preferred_meaning_label or rec.valid_value)])


def _query_render_generic_as_label(*, pool: CanonicalPVPool, rec: CanonicalPVRecord, config: Mapping[str, Any], seed: str) -> bool:
    p = _normalize_probability(float(config.get("sde_generic_label_p", 0.30)))
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    key = f"{seed}::{rec.generic_bucket_key or rec.generic_concept or rec.stable_key}"
    return (_hash_key(key) % 10000) < int(round(p * 10000))


def render_query_side(sampled_subset: Sequence[CanonicalPVRecord], *, pool: CanonicalPVPool, config: Mapping[str, Any], seed: str) -> List[str]:
    out: List[str] = []
    for rec in sampled_subset:
        if rec.generic_bucket_key:
            use_label = _query_render_generic_as_label(pool=pool, rec=rec, config=config, seed=seed)
            value = rec.preferred_meaning_label if use_label and rec.preferred_meaning_label else (rec.valid_value or rec.preferred_meaning_label)
        else:
            value = rec.valid_value or rec.preferred_meaning_label
        value = _norm_text(value)
        if value:
            out.append(value)
    return _dedupe_preserve_order(out)


def _format_block(sig: str, pv_n: int, examples: Sequence[str], *, sep: str) -> str:
    head = f"PV_TYPE: {sig}(n={int(pv_n)})"
    ex = _dedupe_preserve_order([_norm_text(x) for x in examples if _norm_text(x)])
    if not ex:
        return head
    return head + "; PV: " + str(sep).join(ex)


def _items_strict_subset(xs: Sequence[str], ys: Sequence[str]) -> bool:
    sx = list(xs)
    sy = list(ys)
    if not sx or len(sx) >= len(sy):
        return False
    return set(sx).issubset(set(sy)) and set(sx) != set(sy)


def _token_overlap_score(xs: Sequence[str], ys: Sequence[str]) -> float:
    tx = set(" ".join(xs).split())
    ty = set(" ".join(ys).split())
    if not tx or not ty:
        return 0.0
    return float(len(tx & ty)) / float(len(tx | ty))


def _query_omission_text(*, config: Mapping[str, Any]) -> str:
    if bool(config.get("use_placeholder_for_query_omission", False)):
        return str(config.get("pv_placeholder_token", "<MISSING_PV_SUMMARY>"))
    return ""


def apply_small_n_policy(
    pool: CanonicalPVPool,
    query_items: Sequence[str],
    cde_items: Sequence[str],
    *,
    profile: str,
    config: Mapping[str, Any],
) -> tuple[List[str], Dict[str, Any]]:
    query = list(query_items)
    policy: Dict[str, Any] = {"omitted": False, "subset_trigger": False, "small_n_rule": None}
    n = pool.n
    any_non_generic = bool(pool.non_generic_records)
    generic_ratio = (float(pool.m) / float(pool.n)) if pool.n > 0 else 0.0
    generic_dominant = generic_ratio >= float(config.get("pv_small_n_generic_threshold", 0.5))

    if profile == "gdc_strict" and n <= 3 and any_non_generic:
        policy.update({"omitted": True, "small_n_rule": "gdc_strict_small_n_non_generic"})
        return [], policy

    if n == 1:
        policy.update({"omitted": True, "small_n_rule": "n1"})
        return [], policy
    if n == 2 and any_non_generic:
        policy.update({"omitted": True, "small_n_rule": "n2_non_generic"})
        return [], policy
    if n == 2 and not any_non_generic:
        policy["small_n_rule"] = "n2_all_generic"
        return query, policy
    if n == 3 and generic_dominant:
        policy["small_n_rule"] = "n3_generic_dominant"
        return query, policy
    if n == 3 and not generic_dominant:
        if len(query) < 2 or len(cde_items) < 2:
            policy.update({"omitted": True, "small_n_rule": "n3_non_generic_dominant_insufficient_items"})
            return [], policy
        policy["small_n_rule"] = "n3_non_generic_dominant"
        return query, policy
    return query, policy


def apply_anti_leakage(
    pool: CanonicalPVPool,
    query_items: Sequence[str],
    cde_items: Sequence[str],
    *,
    profile: str,
    config: Mapping[str, Any],
    seed_base: str,
) -> tuple[List[str], Dict[str, Any]]:
    query = list(query_items)
    diag: Dict[str, Any] = {"identity_retry_count": 0, "subset_trigger": False, "exact_identity_before": False, "exact_identity_after": False}
    max_attempts = int(config.get("pv_max_resample_attempts", 3))
    if query == list(cde_items):
        diag["exact_identity_before"] = True
        k = len(query)
        for attempt in range(1, max_attempts + 1):
            alt = render_query_side(
                sample_side_subset(pool, side="query", k=k, config=config, seed=f"{seed_base}::resample::{attempt}"),
                pool=pool,
                config=config,
                seed=f"{seed_base}::render::{attempt}",
            )
            diag["identity_retry_count"] = int(attempt)
            if alt != list(cde_items):
                query = alt
                break
        if query == list(cde_items):
            diag["exact_identity_after"] = True
            return [], diag

    non_generic_dominant = not pool.generic_dominant
    if pool.n <= 3 and non_generic_dominant and len(query) in {1, 2} and _items_strict_subset(query, cde_items):
        diag["subset_trigger"] = True
        return [], diag
    if profile == "gdc_strict" and _items_strict_subset(query, cde_items):
        diag["subset_trigger"] = True
        return [], diag
    return query, diag


def build_dual_summary_for_group(
    df_group: pd.DataFrame,
    *,
    profile: str,
    config: Mapping[str, Any],
    salt: str,
    sep: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    pool = canonicalize_pv_pool(df_group, profile=profile, config=config)
    pv_n = int(pool.n)
    sig = pv_type_signature([rec.preferred_meaning_label for rec in pool.records])
    seed_query = _seed_base(pool, side="query", salt=salt)
    seed_cde = _seed_base(pool, side="cde", salt=salt)
    k_query = compute_side_sample_size(pool, side="query", config=config, seed=seed_query)
    k_cde = compute_side_sample_size(pool, side="cde", config=config, seed=seed_cde)

    query_subset = sample_side_subset(pool, side="query", k=k_query, config=config, seed=seed_query)
    cde_subset = sample_side_subset(pool, side="cde", k=k_cde, config=config, seed=seed_cde)
    query_items = render_query_side(query_subset, pool=pool, config=config, seed=seed_query)
    cde_items = render_cde_side(cde_subset, config=config)

    query_items, small_diag = apply_small_n_policy(pool, query_items, cde_items, profile=profile, config=config)
    anti_diag = {"identity_retry_count": 0, "subset_trigger": False, "exact_identity_before": False, "exact_identity_after": False}
    if not small_diag.get("omitted", False):
        query_items, anti_diag = apply_anti_leakage(
            pool,
            query_items,
            cde_items,
            profile=profile,
            config=config,
            seed_base=seed_query,
        )
    query_omitted = small_diag.get("omitted", False) or (not query_items and pv_n > 0)
    query_block = _query_omission_text(config=config) if query_omitted else _format_block(sig, pv_n, query_items, sep=sep)
    cde_block = _format_block(sig, pv_n, cde_items, sep=sep)

    summary = {
        "cde_publicid": pool.cde_publicid,
        "cde_version": pool.cde_version,
        "PV_N": int(pv_n),
        "PV_TYPE": sig,
        "PV_BLOCK_CDE": cde_block,
        "PV_BLOCK_SDE": query_block,
    }
    diag = {
        "cde_publicid": pool.cde_publicid,
        "cde_version": pool.cde_version,
        "profile": profile,
        "N": int(pool.n),
        "M": int(pool.m),
        "U": int(len(pool.generic_buckets)),
        "k_query": int(k_query),
        "k_cde": int(k_cde),
        "query_len": int(len(query_items)),
        "cde_len": int(len(cde_items)),
        "exact_overlap_count": int(len(set(query_items) & set(cde_items))),
        "token_overlap_score": float(_token_overlap_score(query_items, cde_items)),
        "placeholder_flag": bool(bool(config.get("use_placeholder_for_query_omission", False)) and query_omitted),
        "omission_flag": bool(query_omitted),
        "identity_retry_count": int(anti_diag.get("identity_retry_count", 0)),
        "subset_trigger_flag": bool(small_diag.get("subset_trigger", False) or anti_diag.get("subset_trigger", False)),
        "small_n_rule": small_diag.get("small_n_rule"),
        "exact_identity_before": bool(anti_diag.get("exact_identity_before", False)),
        "exact_identity_after": bool(anti_diag.get("exact_identity_after", False)),
        "generic_dominant": bool(pool.generic_dominant),
    }
    return summary, diag


def default_config_from_kwargs(**kwargs: Any) -> Dict[str, Any]:
    pv_max_n = int(kwargs.get("pv_max_n", 10))
    pv_max_n_query = kwargs.get("pv_max_n_query")
    pv_max_n_cde = kwargs.get("pv_max_n_cde")
    return {
        "pv_center_fraction": float(kwargs.get("pv_center_fraction", 0.5)),
        "pv_min_n": int(kwargs.get("pv_min_n", 2)),
        "pv_max_n_query": int(min(pv_max_n, 8) if pv_max_n_query is None else pv_max_n_query),
        "pv_max_n_cde": int(pv_max_n if pv_max_n_cde is None else pv_max_n_cde),
        "pv_size_jitter": int(kwargs.get("pv_size_jitter", 1)),
        "pv_generic_boost": float(kwargs.get("pv_generic_boost", 0.10)),
        "pv_generic_cap": float(kwargs.get("pv_generic_cap", 0.70)),
        "sde_generic_label_p": kwargs.get("sde_generic_label_p", 0.30),
        "pv_small_n_generic_threshold": float(kwargs.get("pv_small_n_generic_threshold", 0.5)),
        "pv_max_resample_attempts": int(kwargs.get("pv_max_resample_attempts", 3)),
        "pv_placeholder_token": str(kwargs.get("pv_placeholder_token", "<MISSING_PV_SUMMARY>")),
        "use_placeholder_for_query_omission": bool(kwargs.get("use_placeholder_for_query_omission", False)),
    }


def build_pv_summary_tables(
    pv: pd.DataFrame,
    *,
    profile: str = "cadsr",
    salt: str = "demap",
    sep: str = " | ",
    diagnostics_summary_path: str | Path | None = None,
    diagnostics_records_path: str | Path | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["cde_publicid", "cde_version", "PV_N", "PV_TYPE", "PV_BLOCK_CDE", "PV_BLOCK_SDE"]
    diag_cols = [
        "cde_publicid",
        "cde_version",
        "profile",
        "N",
        "M",
        "U",
        "k_query",
        "k_cde",
        "query_len",
        "cde_len",
        "exact_overlap_count",
        "token_overlap_score",
        "placeholder_flag",
        "omission_flag",
        "identity_retry_count",
        "subset_trigger_flag",
        "small_n_rule",
        "exact_identity_before",
        "exact_identity_after",
        "generic_dominant",
    ]
    if pv is None or pv.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=diag_cols)
    required = ["cde_publicid", "cde_version"]
    missing = [c for c in required if c not in pv.columns]
    if missing:
        raise ValueError(f"PV table missing required columns: {missing}")
    if not any(c in pv.columns for c in ["value_meaning_long_name", "value_meaning", "meaning_description"]):
        raise ValueError("PV table missing a meaning column. Tried: value_meaning_long_name, value_meaning, meaning_description")

    cfg = default_config_from_kwargs(**kwargs)
    summary_rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []
    for _, group in pv.groupby(["cde_publicid", "cde_version"], dropna=False, sort=False):
        summary, diag = build_dual_summary_for_group(group, profile=profile, config=cfg, salt=salt, sep=sep)
        summary_rows.append(summary)
        diag_rows.append(diag)
    summary_df = pd.DataFrame(summary_rows, columns=columns)
    diag_df = pd.DataFrame(diag_rows, columns=diag_cols)

    if diagnostics_summary_path is not None:
        p = Path(diagnostics_summary_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "profile": profile,
            "n_cdes": int(len(diag_df)),
            "omission_rate": float(diag_df["omission_flag"].mean()) if not diag_df.empty else 0.0,
            "placeholder_rate": float(diag_df["placeholder_flag"].mean()) if not diag_df.empty else 0.0,
            "exact_identity_pre_rate": float(diag_df["exact_identity_before"].mean()) if not diag_df.empty else 0.0,
            "exact_identity_post_rate": float(diag_df["exact_identity_after"].mean()) if not diag_df.empty else 0.0,
            "subset_trigger_rate": float(diag_df["subset_trigger_flag"].mean()) if not diag_df.empty else 0.0,
            "mean_query_len": float(diag_df["query_len"].mean()) if not diag_df.empty else 0.0,
            "mean_cde_len": float(diag_df["cde_len"].mean()) if not diag_df.empty else 0.0,
            "config": cfg,
        }
        p.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    if diagnostics_records_path is not None:
        p = Path(diagnostics_records_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        diag_df.to_csv(p, index=False)
    return summary_df, diag_df
