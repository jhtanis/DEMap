#!/usr/bin/env python3
"""catalog_filter.py

First-class, reproducible **CDE-side catalog construction + query reachability filtering**.

Motivation
----------
The query-side datasets/splits and the CDE-side catalog may be built from the SAME caDSR XML
snapshot, or from DIFFERENT snapshots (e.g. queries from a research export, catalog from the
production CDE-Match snapshot). Every emitted query sample must have its gold ``cde_id`` present in
the selected CDE-side catalog; otherwise the query is un-retrievable *by construction* and must be
removed. This module makes that construction + filtering deterministic, recorded, and permanent —
replacing the earlier ad-hoc ``scripts/build_phase0_production_reachable_splits.py`` fix.

What it does
------------
1. Resolve the CDE-side catalog from EXACTLY ONE source:
     * ``--catalog-enriched PATH``  an enriched CDE master parquet (apply eligibility) — this is the
       "same as query source" path (pass the query-side enriched master).
     * ``--catalog-xml PATH``       a *different* caDSR XML folder → extract→merge→enrich→eligibility.
     * ``--catalog-parquet PATH``   an already-final catalog parquet, used verbatim (no eligibility).
   The resolved/eligibility-filtered catalog is written to ``--catalog-out``.
2. Build the catalog ``cde_id`` set.
3. For each requested query split (``--emit-splits``, e.g. ``train`` only for production), drop rows
   whose gold ``cde_id`` is not in the catalog set. Deterministic set-membership; identical logic to
   the prior script (``cde_id``; fall back to ``cde_publicid::cde_version``).
4. Write filtered splits to ``--out-dir``, a dropped-query report (parquet + CSV) to ``--reports-dir``,
   and a reproducibility manifest (config, both XML sources, catalog path, eligibility report, splits,
   before/after counts, dropped counts, dropped-report path, git commit).

This module is used both standalone (``demap catalog-filter``) and as the opt-in final stage of
``demap make-dataset``.

Idempotence note
----------------
When building a catalog from ``--catalog-xml``, an existing ``--catalog-out`` parquet is reused
unless ``--force`` is passed. Likewise, when changing the query XML source in ``make-dataset``
(``--query-xml``), the query stages are skipped if their interim outputs already exist — pass
``--force`` (or use clean/distinct interim dirs) when switching either source so stale outputs are
not silently reused.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from demap_repro.data import eligibility as elig


# ---------------------------------------------------------------------------
# small helpers (mirroring make_dataset conventions)
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _git_info(repo_root: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        info["commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root)).decode().strip()
        )
    except Exception:
        pass
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo_root)).decode().strip()
        info["dirty"] = bool(status)
        if status:
            info["dirty_paths"] = len(status.splitlines())
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# pure core (unit-testable)
# ---------------------------------------------------------------------------
def split_gold_ids(df: pd.DataFrame) -> pd.Series:
    """Return the gold CDE identifier per query row as a string Series.

    Uses ``cde_id`` when present; otherwise ``cde_publicid::cde_version``. This mirrors
    ``scripts/build_phase0_production_reachable_splits.py`` exactly so the two agree.
    """
    if "cde_id" in df.columns:
        return df["cde_id"].astype(str)
    if {"cde_publicid", "cde_version"}.issubset(df.columns):
        return df["cde_publicid"].astype(str) + "::" + df["cde_version"].astype(str)
    raise KeyError("split missing gold id column (need 'cde_id', or 'cde_publicid'+'cde_version')")


def catalog_gold_ids(catalog: pd.DataFrame) -> set:
    """Return the set of catalog CDE identifiers (same key convention as ``split_gold_ids``)."""
    if "cde_id" in catalog.columns:
        return {str(x) for x in catalog["cde_id"].tolist()}
    if {"cde_publicid", "cde_version"}.issubset(catalog.columns):
        return {
            f"{pid}::{ver}"
            for pid, ver in zip(catalog["cde_publicid"].astype(str), catalog["cde_version"].astype(str))
        }
    raise KeyError("catalog missing id column (need 'cde_id', or 'cde_publicid'+'cde_version')")


def filter_split_by_catalog(
    df: pd.DataFrame, catalog_ids: set
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (kept, dropped) by gold ``cde_id`` membership in ``catalog_ids``.

    Deterministic: pure set-membership on the gold id, row order preserved.
    """
    gold = split_gold_ids(df)
    keep = gold.isin(catalog_ids)
    kept = df[keep.values].reset_index(drop=True)
    dropped = df[(~keep).values].reset_index(drop=True)
    return kept, dropped


def _dropped_report_rows(dropped: pd.DataFrame, split: str) -> pd.DataFrame:
    """Build a compact, human-readable dropped-query report frame for one split."""
    if len(dropped) == 0:
        cols = ["split", "reason", "gold_cde_id", "pair_id", "query_id", "query_text"]
        return pd.DataFrame(columns=cols)
    gold = split_gold_ids(dropped)
    out = pd.DataFrame(
        {
            "split": split,
            "reason": "gold_cde_id_not_in_catalog",
            "gold_cde_id": gold.values,
        }
    )
    for c in ("pair_id", "query_id", "query_text", "query_source", "query_field"):
        if c in dropped.columns:
            out[c] = dropped[c].values
    return out


# ---------------------------------------------------------------------------
# catalog resolution
# ---------------------------------------------------------------------------
def _enrich_argv_from_config(cfg: Dict[str, Any], merged_dir: Path, out_parquet: Path) -> List[str]:
    """Build the ``enrich-cde-master`` argv from a pipeline config's enrich/pv_summary sections.

    Mirrors ``make_dataset``'s enrich stage so a separately-built catalog is enriched identically
    to the query side (only the ``cde_id`` set matters for reachability, but the catalog parquet is
    a real drop-in CDE-side catalog).
    """
    enrich_cfg = cfg.get("enrich", {})
    pv_cfg = cfg.get("pv_summary", {})
    language = str(enrich_cfg.get("language", "English"))
    pv_max_n = int(enrich_cfg.get("pv_max_n", enrich_cfg.get("pv_max", 10)))
    argv = [
        "--merged-dir", str(merged_dir),
        "--out-parquet", str(out_parquet),
        "--pv-max-n", str(pv_max_n),
        "--pv-summary-profile", str(pv_cfg.get("pv_summary_profile", "cadsr")),
        "--pv-center-fraction", str(float(pv_cfg.get("pv_center_fraction", 0.5))),
        "--pv-min-n", str(int(pv_cfg.get("pv_min_n", 2))),
        "--pv-size-jitter", str(int(pv_cfg.get("pv_size_jitter", 1))),
        "--pv-generic-boost", str(float(pv_cfg.get("pv_generic_boost", 0.10))),
        "--pv-generic-cap", str(float(pv_cfg.get("pv_generic_cap", 0.70))),
        "--sde-generic-label-p", str(float(pv_cfg.get("sde_generic_label_p", 0.30))),
        "--pv-small-n-generic-threshold", str(float(pv_cfg.get("pv_small_n_generic_threshold", 0.5))),
        "--pv-max-resample-attempts", str(int(pv_cfg.get("pv_max_resample_attempts", 3))),
        "--pv-placeholder-token", str(pv_cfg.get("pv_placeholder_token", "<MISSING_PV_SUMMARY>")),
        "--language", language,
    ]
    if pv_cfg.get("pv_max_n_query", None) is not None:
        argv += ["--pv-max-n-query", str(int(pv_cfg["pv_max_n_query"]))]
    if pv_cfg.get("pv_max_n_cde", None) is not None:
        argv += ["--pv-max-n-cde", str(int(pv_cfg["pv_max_n_cde"]))]
    return argv


def build_catalog_from_xml(
    xml_source: Path,
    interim_root: Path,
    out_enriched: Path,
    cfg: Dict[str, Any],
    *,
    force: bool = False,
) -> Path:
    """Build an enriched CDE master from a caDSR XML folder via extract→merge→enrich.

    Reuses the existing stage ``main`` functions with catalog-scoped output dirs so it never
    collides with the query-side interim outputs. Idempotent unless ``force``.
    """
    # Imported here so tests can monkeypatch the module attributes.
    from demap_repro.data import extract_cadsr_xml, merge_cadsr_parquet
    from demap_repro.data import enrich_master

    extracted = interim_root / "cadsr_extracted"
    merged = interim_root / "cadsr_merged"
    summaries = interim_root / "summaries"
    _ensure_dir(interim_root)

    if force or not out_enriched.exists():
        extract_argv = ["--input", str(xml_source), "--out-root", str(extracted)]
        if force:
            extract_argv += ["--overwrite"]
        extract_cadsr_xml.main(extract_argv)

        merge_cadsr_parquet.main(
            ["--out-root", str(extracted), "--merged-dir", str(merged), "--summaries-dir", str(summaries)]
        )

        _ensure_dir(out_enriched.parent)
        enrich_master.main(_enrich_argv_from_config(cfg, merged, out_enriched))

    return out_enriched


def resolve_catalog(
    *,
    catalog_enriched: Optional[str],
    catalog_xml: Optional[str],
    catalog_parquet: Optional[str],
    eligibility_mode: str,
    catalog_out: Path,
    interim_root: Path,
    cfg: Dict[str, Any],
    force: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Resolve the final CDE-side catalog dataframe + a provenance record.

    Exactly one of ``catalog_enriched`` / ``catalog_xml`` / ``catalog_parquet`` must be given.
    """
    sources = [("catalog_enriched", catalog_enriched), ("catalog_xml", catalog_xml),
               ("catalog_parquet", catalog_parquet)]
    given = [(k, v) for k, v in sources if v]
    if len(given) != 1:
        raise SystemExit(
            "catalog-filter: provide EXACTLY ONE of --catalog-enriched / --catalog-xml / "
            f"--catalog-parquet (got: {[k for k, _ in given] or 'none'})"
        )
    source_kind, source_val = given[0]

    prov: Dict[str, Any] = {
        "source_kind": source_kind,
        "source_value": str(source_val),
        "eligibility_mode": eligibility_mode,
        "catalog_out": str(catalog_out),
    }

    if source_kind == "catalog_parquet":
        # Use verbatim; assume already final. No eligibility applied.
        catalog = pd.read_parquet(source_val)
        prov["eligibility_mode"] = "none (used --catalog-parquet as-is)"
        prov["eligibility_report"] = None
        prov["catalog_rows"] = int(len(catalog))
        prov["catalog_xml_source"] = None
        return catalog, prov

    if source_kind == "catalog_xml":
        enriched_path = build_catalog_from_xml(
            Path(source_val), interim_root, catalog_out.with_suffix(".enriched.parquet"), cfg, force=force
        )
        prov["catalog_xml_source"] = str(source_val)
        enriched = pd.read_parquet(enriched_path)
    else:  # catalog_enriched
        prov["catalog_xml_source"] = None
        enriched = pd.read_parquet(source_val)

    catalog, report = elig.filter_eligible(enriched, mode=eligibility_mode, return_report=True)
    catalog = catalog.reset_index(drop=True)
    prov["eligibility_report"] = report
    prov["n_before_eligibility"] = int(len(enriched))
    prov["catalog_rows"] = int(len(catalog))

    _ensure_dir(catalog_out.parent)
    catalog.to_parquet(catalog_out, index=False)
    return catalog, prov


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def emit_filtered_splits(
    *,
    splits_dir: Path,
    emit_splits: List[str],
    catalog_ids: Optional[set],
    out_dir: Path,
    reports_dir: Path,
    do_filter: bool,
) -> Dict[str, Any]:
    """Emit selected query splits, optionally filtered against the catalog id-set.

    Returns a dict with per-split counts and the dropped-report path. When ``do_filter`` is False,
    splits are copied through unchanged (still recorded, with 0 dropped).
    """
    # Output safety: never write filtered splits back over the (canonical) source dir, which would
    # silently mutate/clobber the inputs we are reading from. The filtered set is a derived artifact.
    if out_dir.resolve() == splits_dir.resolve():
        raise SystemExit(
            f"catalog-filter: --out-dir ({out_dir}) must differ from --splits-dir ({splits_dir}); "
            "refusing to overwrite the source splits in place."
        )
    _ensure_dir(out_dir)
    _ensure_dir(reports_dir)

    per_split: Dict[str, Any] = {}
    all_dropped: List[pd.DataFrame] = []

    for split in emit_splits:
        src = splits_dir / f"{split}.parquet"
        if not src.exists():
            per_split[split] = {"status": "missing_source", "src": str(src)}
            continue
        df = pd.read_parquet(src)
        n_before = int(len(df))

        if do_filter:
            if catalog_ids is None:
                raise ValueError("do_filter=True requires a catalog_ids set")
            kept, dropped = filter_split_by_catalog(df, catalog_ids)
        else:
            kept, dropped = df.reset_index(drop=True), df.iloc[0:0]

        kept.to_parquet(out_dir / f"{split}.parquet", index=False)
        n_after = int(len(kept))
        n_dropped = int(len(dropped))
        per_split[split] = {
            "status": "ok",
            "n_before_filter": n_before,
            "n_after_filter": n_after,
            "n_dropped_orphan": n_dropped,
            "out": str(out_dir / f"{split}.parquet"),
        }
        if n_dropped:
            all_dropped.append(_dropped_report_rows(dropped, split))

    # dropped-query report (parquet + CSV), deterministic ordering
    dropped_all = (
        pd.concat(all_dropped, ignore_index=True) if all_dropped else _dropped_report_rows(pd.DataFrame(), "")
    )
    if len(dropped_all):
        sort_cols = [c for c in ("split", "gold_cde_id", "pair_id", "query_id") if c in dropped_all.columns]
        dropped_all = dropped_all.sort_values(sort_cols).reset_index(drop=True)
    dropped_parquet = reports_dir / "dropped_queries.parquet"
    dropped_csv = reports_dir / "dropped_queries.csv"
    dropped_all.to_parquet(dropped_parquet, index=False)
    dropped_all.to_csv(dropped_csv, index=False)

    return {
        "per_split": per_split,
        "dropped_report_parquet": str(dropped_parquet),
        "dropped_report_csv": str(dropped_csv),
        "n_dropped_total": int(len(dropped_all)),
    }


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Build a CDE-side catalog and filter query splits so every gold CDE is reachable."
    )
    ap.add_argument("--config", default=os.path.join("configs", "pipeline.yaml"),
                    help="Pipeline YAML (sourced only for enrich/pv params when building from --catalog-xml)")
    ap.add_argument("--splits-dir", default=os.path.join("data", "processed", "splits"),
                    help="Directory holding the query-side <split>.parquet files")
    ap.add_argument("--emit-splits", default="train,val_dev,test,val_train",
                    help="Comma-separated splits to emit (e.g. 'train' for train-only production)")
    ap.add_argument("--out-dir", default=os.path.join("data", "processed", "splits_catalog_filtered"),
                    help="Output directory for the filtered query splits")
    ap.add_argument("--reports-dir", default=os.path.join("artifacts", "manifests", "catalog_filter"),
                    help="Directory for the dropped-query report + manifest")

    # catalog source (exactly one)
    ap.add_argument("--catalog-enriched", default=None,
                    help="An enriched CDE master parquet to use as catalog (apply eligibility). "
                         "Use the query-side enriched master here for 'same as query source'.")
    ap.add_argument("--catalog-xml", default=None,
                    help="A DIFFERENT caDSR XML folder for the CDE-side catalog (extract→merge→enrich).")
    ap.add_argument("--catalog-parquet", default=None,
                    help="An already-final catalog parquet, used verbatim (no eligibility applied).")

    ap.add_argument("--catalog-eligibility", default="production_cde_match",
                    choices=list(elig.ELIGIBILITY_MODES),
                    help="Eligibility filter applied to --catalog-enriched / --catalog-xml catalogs.")
    ap.add_argument("--catalog-out", default=os.path.join("data", "processed", "cde_catalog_enriched.parquet"),
                    help="Where to write the resolved (eligibility-filtered) CDE-side catalog.")
    ap.add_argument("--catalog-interim-root", default=os.path.join("data", "interim", "cde_catalog"),
                    help="Interim root for building a catalog from --catalog-xml.")
    ap.add_argument("--no-filter", action="store_true",
                    help="Emit selected splits WITHOUT catalog filtering (still recorded).")
    ap.add_argument("--manifest", default=None,
                    help="Manifest path (default: <reports-dir>/catalog_filter_manifest.json)")
    ap.add_argument("--force", action="store_true", help="Rebuild catalog from XML even if it exists.")
    args = ap.parse_args(argv)

    from demap_repro.utils.config import load_config

    cfg_path = Path(args.config)
    cfg_all = load_config(cfg_path) if cfg_path.exists() else {}
    cfg = cfg_all.get("pipeline", cfg_all) if isinstance(cfg_all, dict) else {}

    repo_root = Path.cwd()
    emit_splits = [s.strip() for s in str(args.emit_splits).split(",") if s.strip()]
    reports_dir = Path(args.reports_dir)
    do_filter = not args.no_filter

    catalog_ids: Optional[set] = None
    catalog_prov: Dict[str, Any] = {"filtering_enabled": do_filter}
    if do_filter:
        catalog, catalog_prov = resolve_catalog(
            catalog_enriched=args.catalog_enriched,
            catalog_xml=args.catalog_xml,
            catalog_parquet=args.catalog_parquet,
            eligibility_mode=args.catalog_eligibility,
            catalog_out=Path(args.catalog_out),
            interim_root=Path(args.catalog_interim_root),
            cfg=cfg,
            force=args.force,
        )
        catalog_prov["filtering_enabled"] = True
        catalog_ids = catalog_gold_ids(catalog)
        catalog_prov["n_catalog_ids"] = len(catalog_ids)

    emit_result = emit_filtered_splits(
        splits_dir=Path(args.splits_dir),
        emit_splits=emit_splits,
        catalog_ids=catalog_ids,
        out_dir=Path(args.out_dir),
        reports_dir=reports_dir,
        do_filter=do_filter,
    )

    manifest = {
        "created_at": _now_iso(),
        "command": "catalog-filter",
        "argv": list(argv) if argv is not None else sys.argv[1:],
        "config": str(cfg_path),
        "repo_root": str(repo_root.resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "git": _git_info(repo_root),
        "query_source": {"splits_dir": str(args.splits_dir), "emit_splits": emit_splits},
        "catalog_source": catalog_prov,
        "filter_by_catalog": do_filter,
        "outputs": {"filtered_splits_dir": str(args.out_dir)},
        "per_split_counts": emit_result["per_split"],
        "dropped": {
            "n_dropped_total": emit_result["n_dropped_total"],
            "report_parquet": emit_result["dropped_report_parquet"],
            "report_csv": emit_result["dropped_report_csv"],
        },
    }

    manifest_path = Path(args.manifest) if args.manifest else reports_dir / "catalog_filter_manifest.json"
    _ensure_dir(manifest_path.parent)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[catalog-filter] catalog: {catalog_prov.get('source_kind', '(none/no-filter)')} "
          f"-> {catalog_prov.get('n_catalog_ids', 0)} ids")
    for s, rec in emit_result["per_split"].items():
        if rec.get("status") == "ok":
            print(f"  {s:<12} before={rec['n_before_filter']:>7} after={rec['n_after_filter']:>7} "
                  f"dropped_orphan={rec['n_dropped_orphan']:>5}")
        else:
            print(f"  {s:<12} {rec.get('status')}")
    print(f"[catalog-filter] dropped report: {emit_result['dropped_report_csv']}")
    print(f"[catalog-filter] manifest: {manifest_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
