#!/usr/bin/env python3
"""Build CDE Match (V1 clone) candidate tables for evaluation splits.

Runs the Python CDE Match approximation (``clone``) over one or more splits and
writes, per split:
  - a long candidate parquet (schema compatible with the HGBC CDE Match loader):
      cde_match_clone_candidates_<split>[_<tag>].parquet
  - a sidecar summary JSON: per-rule candidate counts, query coverage, recall@K
      (when gold labels are present), and the inferred-constant settings.

Unsupervised: NO fitting on any split. test / external / CIMAC are report-only;
val_dev is the inspection split. The script never overwrites an existing output
unless ``--overwrite`` (OVERWRITE=1) is given.

Biowulf scratch hygiene: TMPDIR/TEMP/TMP are pinned to /lscratch/$SLURM_JOB_ID
when present, else an artifact-local tmp dir (never shared /tmp).

Local dry-run (tiny subset, no Slurm):
    PYTHONPATH=src .venv/bin/python scripts/build_cde_match_candidates.py \
        --splits val_dev --limit-queries 50 --out-dir .scratch/cde_match_clone_dryrun
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

from demap_repro.data import eligibility as elig
from demap_repro.lexical.cde_match import clone as cmc
from demap_repro.utils.paths import data_root

#: Data tree. Was the research repository's own root; see DEMAP_DATA_ROOT.
REPO_ROOT = data_root()

DEFAULT_SPLIT_DIR = REPO_ROOT / "data/processed/splits_v3_cdisc"
DEFAULT_CDE_MASTER = REPO_ROOT / "data/processed/cde_master_enriched.parquet"
DEFAULT_ALT_NAMES = REPO_ROOT / "data/interim/cadsr_merged/cde_alternate_names.parquet"
DEFAULT_PVS = REPO_ROOT / "data/interim/cadsr_merged/cde_permissible_values.parquet"
DEFAULT_REF_DOCS = REPO_ROOT / "data/interim/cadsr_merged/cde_reference_documents.parquet"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts_v3_cdisc/cde_match_clone"


# Splits derived from caDSR metadata (where exact-query-match leakage control is
# meaningful). Non-caDSR calibration sets (CIMAC / GDC / Theradex / clinical_synonym)
# should be left at allow_rate=1.0.
_CADSR_DERIVED_SPLITS = {
    "test", "val_dev", "val_train", "train", "train_noleak", "train_natural",
    "train_natural_noleak", "natural_internal_eval", "external_holdout_org",
    "external_holdout_standard", "external_holdout_refslice",
    # Canonical final-reranker externals generated at allow0.70 (exact-gate applied);
    # kept in sync with scripts/build_hgbc_feature_table.py:_CADSR_DERIVED_SPLITS.
    "cctg", "oid_alt", "cdash",
}


def _is_cadsr_derived(split: str) -> bool:
    return split in _CADSR_DERIVED_SPLITS


def _set_safe_tempdir(out_dir: Path) -> Path:
    """Biowulf-safe temp: /lscratch/$SLURM_JOB_ID if available, else artifact-local
    tmp (gitignored under artifacts_v3_cdisc). Never shared /tmp."""
    job = os.environ.get("SLURM_JOB_ID")
    cand = Path(f"/lscratch/{job}") if job and Path(f"/lscratch/{job}").is_dir() else None
    if cand is None:
        cand = (DEFAULT_OUT_DIR / "tmp")
        cand.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(cand)
    for k in ("TMPDIR", "TEMP", "TMP"):
        os.environ[k] = str(cand)
    return cand


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default="val_dev",
                    help="comma-separated split names (default: val_dev)")
    ap.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    ap.add_argument("--cde-master", default=str(DEFAULT_CDE_MASTER))
    ap.add_argument("--alt-names", default=str(DEFAULT_ALT_NAMES))
    ap.add_argument("--pvs", default=str(DEFAULT_PVS))
    ap.add_argument("--ref-docs", default=str(DEFAULT_REF_DOCS),
                    help="reference-documents table for multi-row question text (Fidelity Pass 1)")
    ap.add_argument("--question-source", choices=["ref_docs", "master_pqt"], default="ref_docs",
                    help="ref_docs = Fidelity Pass 1 multi-row question text (default); "
                         "master_pqt = V1 single PREFERRED_QUESTION_TEXT")
    ap.add_argument("--leakage-holdout", choices=["none", "gold", "any"], default="none",
                    help="LEGACY leakage-aware eval (Mode A; Q-field only, binary, global): drop "
                         "reference/question-text rows that exactly reuse a source query text. "
                         "none=default FP1 (no holdout); gold=exclude reuse only for that query's "
                         "gold CDE; any=exclude exact source-text reuse for ANY CDE. Kept for "
                         "backward compatibility; prefer --exact-query-match-allow-rate (the "
                         "general query-level mechanism over P/Q/A).")
    # ---- exact-query-match leakage control (query-level; preferred mechanism) ----
    ap.add_argument("--exact-query-match-allow-rate", type=float, default=1.0,
                    help="fraction of queries that KEEP their gold CDE's exact-matching metadata "
                         "row(s). 1.0 = DEFAULT (no change, baseline); 0.0 = block all exact "
                         "query-text gold-metadata rows; e.g. 0.65/0.73 calibrates caDSR-derived "
                         "splits to the empirical non-caDSR (CIMAC v2) exact-match rate. Decision "
                         "is per query (deterministic hash of seed+query_id); only the exact-"
                         "matching row is hidden, never the CDE or its other rows.")
    ap.add_argument("--exact-match-seed", type=int, default=42,
                    help="seed for the deterministic per-query allow/block hash (default 42)")
    ap.add_argument("--exact-match-fields", default="P,Q,A",
                    help="comma list of candidate-side fields to control: P=long name, "
                         "Q=question text, A=alternate name (default P,Q,A). PVs are never controlled.")
    ap.add_argument("--exact-match-scope", choices=["gold", "any"], default="gold",
                    help="gold = control exact matches for the query's GOLD CDE only (DEFAULT); "
                         "any = diagnostic, control exact matches for ANY CDE.")
    ap.add_argument("--exact-match-level", choices=["auto", "versioned", "publicid"], default="auto",
                    help="gold match level: auto (DEFAULT; versioned when the gold carries a "
                         "version, else public-id), versioned, or publicid.")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--tag", default="", help="optional output filename suffix")
    # ---- inferred constants, surfaced as explicit knobs ----
    ap.add_argument("--match-limit", type=int, default=cmc.DEFAULT_MATCH_LIMIT,
                    help=f"v_mtch_lmt; max CDEs per query (default {cmc.DEFAULT_MATCH_LIMIT})")
    ap.add_argument("--min-like-len", type=int, default=cmc.DEFAULT_MIN_LIKE_LEN,
                    help=f"v_mtch_min_len; min CDE-term length for like/reverse rules "
                         f"(INFERRED default {cmc.DEFAULT_MIN_LIKE_LEN})")
    ap.add_argument("--pv-match-pct", type=float, default=cmc.DEFAULT_PV_MATCH_PCT,
                    help=f"v_mtch_pcnt; fractional PV-overlap threshold "
                         f"(default {cmc.DEFAULT_PV_MATCH_PCT})")
    ap.add_argument("--normalization", choices=["v1", "exact_sql"], default="exact_sql",
                    help="text normalization: exact_sql = exact per-field SQL regexes "
                         "(v_reg_str_ds / v_reg_str / v_reg_str_adv) -- DEFAULT, matches the "
                         "SQL constants and improves official-overlap; v1 = original inferred "
                         "single normalizer (kept for A/B reproducibility)")
    ap.add_argument("--eligibility", choices=list(elig.ELIGIBILITY_MODES), default="none",
                    help="production-CDE-Match eligibility filter on the CDE catalog "
                         "(getDSFilterString universe). none = DEFAULT, no filter (full master, "
                         "preserves existing artifacts); production_cde_match = drop RETIRED-family "
                         "workflow_status + TEST/Training context CDEs. Use a distinct --tag when "
                         "enabling so eligible outputs do not clobber the unfiltered ones.")
    ap.add_argument("--query-name-col", default="query_text_raw",
                    help="source-name column (falls back to query_text_q3 if absent)")
    ap.add_argument("--limit-queries", type=int, default=None,
                    help="cap queries per split for a dry-run")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing outputs (else refuse); OVERWRITE=1 also works")
    # ---- optional keyword fuzzy fallback (opt-in; baseline output unchanged) ----
    ap.add_argument("--fuzzy-fallback", default=None,
                    choices=sorted(cmc.FUZZY_FALLBACK_SIGNALS),
                    help="append tested keyword fuzzy fallback candidates after the "
                         "clone's exact-rule hits (3-tier rank). Off by default.")
    ap.add_argument("--keyword-index", default=None,
                    help="path to a cached KeywordCatalogIndex joblib; if omitted, the "
                         "index is built/cached from --cde-master next to --out-dir")
    ap.add_argument("--fuzzy-top-k-per-rule", type=int, default=500,
                    help="keyword top-K per (rule, field) when generating fallback candidates")
    ap.add_argument("--fuzzy-top-n", type=int, default=1000,
                    help="cap on fuzzy fallback candidates kept per query")
    ap.add_argument("--fuzzy-word-ngram", action=argparse.BooleanOptionalAction, default=True,
                    help="use the word 1-2gram index for fuzzy fallback (matches the "
                         "June-18 keyword baseline build)")
    args = ap.parse_args(argv)

    overwrite = args.overwrite or os.environ.get("OVERWRITE") == "1"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_safe_tempdir(out_dir)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    suffix = f"_{args.tag}" if args.tag else ""

    # ---- overwrite guard (check ALL targets before doing any work) ----
    planned = {s: out_dir / f"cde_match_clone_candidates_{s}{suffix}.parquet" for s in splits}
    clobber = [str(p) for p in planned.values() if p.exists()]
    if clobber and not overwrite:
        print("ERROR: output(s) already exist; refusing to overwrite "
              "(use --overwrite or OVERWRITE=1):")
        for p in clobber:
            print(f"  {p}")
        return 3

    # ---- leakage-aware holdout sets (Mode A), computed from the eval splits ----
    holdout_terms = holdout_pairs = None
    if args.leakage_holdout != "none":
        all_terms, all_pairs = set(), set()
        for split in splits:
            sp = Path(args.split_dir) / f"{split}.parquet"
            if not sp.exists():
                continue
            qdf = pd.read_parquet(sp)
            if args.limit_queries:
                qdf = qdf.head(args.limit_queries)
            t, p = cmc.compute_refdoc_holdout(qdf, query_name_col=args.query_name_col,
                                              normalization=args.normalization)
            all_terms |= t
            all_pairs |= p
        if args.leakage_holdout == "any":
            holdout_terms = all_terms
        else:  # gold
            holdout_pairs = all_pairs
        print(f"leakage-holdout={args.leakage_holdout}: "
              f"{len(all_terms)} source texts, {len(all_pairs)} (gold,text) pairs")

    print(f"building CDE catalog from:\n  {args.cde_master}\n  {args.alt_names}\n  {args.pvs}\n"
          f"  {args.ref_docs} (question_source={args.question_source}, "
          f"normalization={args.normalization}, leakage_holdout={args.leakage_holdout}, "
          f"eligibility={args.eligibility})")
    catalog = cmc.CdeCatalog.build(
        Path(args.cde_master), Path(args.alt_names), Path(args.pvs),
        ref_docs_path=Path(args.ref_docs), question_source=args.question_source,
        holdout_q_terms=holdout_terms, holdout_q_pairs=holdout_pairs,
        normalization=args.normalization, eligibility=args.eligibility)
    if catalog.eligibility_report is not None:
        print(elig.format_report(catalog.eligibility_report))
    print(f"catalog: {catalog.n_cdes} CDEs | P={len(catalog.fields['P'])} "
          f"Q={len(catalog.fields['Q'])} A={len(catalog.fields['A'])} rows | "
          f"{len(catalog.cde_pv)} CDEs with PVs | question_source={catalog.question_source} "
          f"| normalization={catalog.normalization} | eligibility={catalog.eligibility}")

    clone = cmc.CdeMatchClone(catalog, match_limit=args.match_limit,
                              min_like_len=args.min_like_len, pv_match_pct=args.pv_match_pct)

    # ---- optional keyword fuzzy fallback: build spec + load/build the index ----
    fuzzy_spec = keyword_index = None
    if args.fuzzy_fallback:
        from demap_repro.lexical.cde_match import keyword_retriever as kr  # heavy import
        fuzzy_spec = cmc.FuzzyFallback.from_name(
            args.fuzzy_fallback,
            top_k_per_rule=args.fuzzy_top_k_per_rule,
            query_text_col=args.query_name_col,
            use_word_ngram=args.fuzzy_word_ngram,
            fuzzy_top_n=args.fuzzy_top_n,
        )
        if args.keyword_index:
            import joblib
            print(f"loading keyword index: {args.keyword_index}")
            keyword_index = joblib.load(args.keyword_index)
            kr._assert_index_eligibility(keyword_index, args.eligibility,
                                         Path(args.keyword_index))
        else:
            cache_dir = out_dir / "keyword_index"
            # The keyword candidate universe honors the SAME --eligibility flag
            # as the clone catalog. Historically the flag reached only the clone
            # catalog while the index was built from the unfiltered master, which
            # let retired/archived CDEs enter candidate pools via the keyword arm
            # (the paper-v13 eligibility defect). The cache name encodes the mode
            # so a filtered and an unfiltered index can never be confused.
            cache_name = ("keyword_index_v2.joblib" if args.eligibility == "none"
                          else f"keyword_index_v3_{args.eligibility}.joblib")
            keyword_index = kr.load_or_build_index(
                Path(args.cde_master), cache_dir,
                word_ngram=args.fuzzy_word_ngram,
                eligibility=args.eligibility,
                cache_name=cache_name)
        print(f"fuzzy-fallback={fuzzy_spec.name} signals={list(fuzzy_spec.signals)} "
              f"index_CDEs={keyword_index.n_cde} top_k_per_rule={fuzzy_spec.top_k_per_rule} "
              f"top_n={fuzzy_spec.fuzzy_top_n} word_ngram={fuzzy_spec.use_word_ngram}")

    # ---- exact-query-match leakage control (query-level; default no-op at 1.0) ----
    exact_ctrl = cmc.ExactMatchControl(
        allow_rate=args.exact_query_match_allow_rate,
        seed=args.exact_match_seed,
        fields=tuple(f.strip().upper() for f in args.exact_match_fields.split(",") if f.strip()),
        scope=args.exact_match_scope,
        level=args.exact_match_level,
    )
    if exact_ctrl.is_active():
        print(f"exact-query-match control ACTIVE: allow_rate={exact_ctrl.allow_rate} "
              f"seed={exact_ctrl.seed} fields={','.join(exact_ctrl.fields)} "
              f"scope={exact_ctrl.scope} level={exact_ctrl.level}")
        if not args.tag:
            print("  WARNING: control is active but no --tag set; outputs may clobber baseline. "
                  "Use a distinct --tag (e.g. exactallow0.65_seed42).")
        non_cadsr = [s for s in splits if not _is_cadsr_derived(s)]
        if non_cadsr:
            print(f"  WARNING: exact-match control is intended for caDSR-DERIVED splits; these "
                  f"look non-caDSR (calibration targets, leave at 1.0): {non_cadsr}")

    settings = {"match_limit": args.match_limit, "min_like_len": args.min_like_len,
                "pv_match_pct": args.pv_match_pct, "query_name_col": args.query_name_col,
                "question_source": catalog.question_source, "fidelity_pass": 2,
                "normalization": catalog.normalization,
                "leakage_holdout": args.leakage_holdout,
                "exact_query_match_control": {
                    "active": exact_ctrl.is_active(),
                    "allow_rate": exact_ctrl.allow_rate,
                    "seed": exact_ctrl.seed,
                    "fields": list(exact_ctrl.fields),
                    "scope": exact_ctrl.scope,
                    "level": exact_ctrl.level,
                },
                "fuzzy_fallback": ({
                    "variant": fuzzy_spec.name,
                    "signals": [list(s) for s in fuzzy_spec.signals],
                    "top_k_per_rule": fuzzy_spec.top_k_per_rule,
                    "fuzzy_top_n": fuzzy_spec.fuzzy_top_n,
                    "use_word_ngram": fuzzy_spec.use_word_ngram,
                    "query_text_col": fuzzy_spec.query_text_col,
                    "keyword_index_cdes": int(keyword_index.n_cde),
                } if fuzzy_spec is not None else None),
                "eligibility": catalog.eligibility,
                "eligibility_report": catalog.eligibility_report,
                "n_cdes": int(catalog.n_cdes),
                "q_field_rows": int(len(catalog.fields["Q"])),
                "a_field_rows": int(len(catalog.fields["A"])),
                "inferred_constants_note": "min_like_len & pv_match_pct are INFERRED; "
                "see CDE_MATCH_CLONE_ASSUMPTIONS.md"}

    rc = 0
    for split in splits:
        sp = Path(args.split_dir) / f"{split}.parquet"
        if not sp.exists():
            print(f"WARNING: split not found, skipping: {sp}")
            rc = max(rc, 1)
            continue
        queries = pd.read_parquet(sp)
        if args.limit_queries:
            queries = queries.head(args.limit_queries).copy()
        print(f"\n[{split}] queries={len(queries)}  (name_col={args.query_name_col})")

        candidates = clone.run_over_queries(
            queries, split, query_name_col=args.query_name_col,
            exact_match_control=exact_ctrl,
            fuzzy_fallback=fuzzy_spec, keyword_index=keyword_index)

        out_path = planned[split]
        candidates.to_parquet(out_path, index=False)

        summary = {
            "split": split,
            "n_candidates": int(len(candidates)),
            "settings": settings,
            "per_rule_counts": cmc.per_rule_counts(candidates).to_dict(orient="records"),
            "query_coverage": cmc.query_coverage(candidates, queries),
            "recall": cmc.recall_at_k(candidates, queries),
        }
        summ_path = out_dir / f"cde_match_clone_summary_{split}{suffix}.json"
        summ_path.write_text(json.dumps(summary, indent=2, default=str))

        cov = summary["query_coverage"]
        rec = summary["recall"]
        print(f"  candidates={len(candidates)}  coverage="
              f"{cov['n_queries_with_candidates']}/{cov['n_queries']} "
              f"({cov['coverage_frac']:.3f})")
        if "recall@5" in rec:
            print(f"  recall@5={rec['recall@5']:.4f} recall@10={rec['recall@10']:.4f} "
                  f"(publicid@10={rec['recall_publicid@10']:.4f}, "
                  f"n_gold={rec['n_queries_with_gold']})")
        print(f"  wrote: {out_path}\n         {summ_path}")

    print("\nDONE.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
