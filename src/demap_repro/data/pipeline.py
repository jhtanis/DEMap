#!/usr/bin/env python3
"""make_dataset.py

Run dataset preparation steps 1–6 in one command.

This is an orchestration layer that chains together existing CLI modules:

  1) extract-cadsr-xml
  2) merge-cadsr-parquet
  3) enrich-cde-master
  4) build-queries
  5) build-pairs
  6) split

Key behaviors
-------------
- Idempotent by default: skip a step if its expected outputs already exist.
- Use --force to re-run steps even if outputs exist.
- Use --start-at / --stop-after to resume or short-circuit.
- Writes a dataset build manifest JSON to BOTH:
    - data/processed/dataset_build_manifest.json
    - artifacts/manifests/dataset_build_manifest.json

Configuration
-------------
This command is driven by a YAML config (default: configs/pipeline.yaml).

"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from demap_repro.data import pairs as build_pairs, queries as inwild_queries, splits as split_policy
from demap_repro.data import extract_cadsr_xml, merge_cadsr_parquet
from demap_repro.data import catalog_filter
from demap_repro.utils.paths import artifact_root, data_root, repo_root as source_repo_root
from demap_repro.data import enrich_master
from demap_repro.utils.config import load_config


STAGES: List[str] = [
    "extract",
    "merge",
    "enrich",
    "build-queries",
    "build-pairs",
    "split",
    "catalog-filter",
]

STAGE_ALIASES: Dict[str, str] = {
    "extract": "extract",
    "extract-cadsr-xml": "extract",
    "merge": "merge",
    "merge-cadsr-parquet": "merge",
    "enrich": "enrich",
    "enrich-cde-master": "enrich",
    "build-queries": "build-queries",
    "queries": "build-queries",
    "build-pairs": "build-pairs",
    "pairs": "build-pairs",
    "split": "split",
    "catalog-filter": "catalog-filter",
    "catalog": "catalog-filter",
    "filter": "catalog-filter",
}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _file_exists(p: str | Path) -> bool:
    pp = Path(p)
    return pp.exists() and pp.is_file()


def _dir_has_parquet(root: str | Path) -> bool:
    r = Path(root)
    if not r.exists() or not r.is_dir():
        return False
    return any(p.is_file() for p in r.rglob("*.parquet"))


def _safe_parquet_rows(path: str | Path) -> Optional[int]:
    """Best-effort parquet row count.

    Returns None if the file is missing or unreadable.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(p)
        return int(len(df))
    except Exception:
        return None



def _is_up_to_date(output: Path, inputs: List[Path]) -> bool:
    """Return True if `output` exists and is newer than all existing `inputs`.

    This is a lightweight dependency check so `demap make-dataset` can avoid
    skipping steps when their declared inputs have changed.
    """

    if not output.exists() or not output.is_file():
        return False
    try:
        out_mtime = output.stat().st_mtime
    except Exception:
        return False

    # Some environments (notably when working from a zip export) can contain
    # files whose stored mtime is slightly in the future relative to the local
    # clock. A strict mtime dependency check would then treat outputs as stale
    # forever, breaking idempotence. We ignore such inputs for the purposes of
    # "is this output up to date?"; users can always force a rebuild.
    now = time.time()
    for inp in inputs:
        try:
            if not (inp.exists() and inp.is_file()):
                continue
            inp_mtime = inp.stat().st_mtime
            # Ignore obviously-future timestamps (clock skew / zip metadata).
            if inp_mtime > now + 1.0:
                continue
            if inp_mtime > out_mtime:
                return False
        except Exception:
            # If we can't stat an input, assume output is not safely up to date.
            return False

    return True


def _git_info(repo_root: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root)).decode().strip()
        info["commit"] = commit
    except Exception:
        pass
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo_root)).decode().strip()
        info["dirty"] = bool(status)
        if status:
            # Don't include full status (can be huge); just include a count.
            info["dirty_paths"] = len(status.splitlines())
    except Exception:
        pass
    return info


def _choose_zip_in_dir(dir_path: Path) -> Optional[Path]:
    preferred = dir_path / "releasedCDEsXML-OD.zip"
    if preferred.exists() and preferred.is_file():
        return preferred
    zips = sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".zip"])
    if len(zips) == 1:
        return zips[0]
    return None


def _list_xml_inputs(input_path: Path) -> Tuple[str, List[str]]:
    """Return (input_type, xml_paths_or_names).

    This is side-effect-free (does not unzip). It is used only for the manifest.
    """
    if input_path.is_file():
        if input_path.suffix.lower() == ".zip":
            try:
                import zipfile

                with zipfile.ZipFile(input_path, "r") as z:
                    xmls = [n for n in z.namelist() if n.lower().endswith(".xml")]
                return "zip", sorted(xmls)
            except Exception:
                return "zip", []
        if input_path.suffix.lower() == ".xml":
            return "xml_file", [str(input_path)]
        return "file", [str(input_path)]

    if input_path.is_dir():
        xmls = sorted([str(p) for p in input_path.glob("*.xml") if p.is_file()])
        if xmls:
            return "xml_dir", xmls
        z = _choose_zip_in_dir(input_path)
        if z is None:
            return "dir_no_xml_or_zip", []
        try:
            import zipfile

            with zipfile.ZipFile(z, "r") as zz:
                xmls2 = [n for n in zz.namelist() if n.lower().endswith(".xml")]
            return "zip_in_dir", sorted(xmls2)
        except Exception:
            return "zip_in_dir", []

    return "missing", []


@dataclass
class Stage:
    name: str
    fn: Callable[[Optional[List[str]]], None]
    argv: List[str]
    done_check: Callable[[], bool]
    outputs: List[str]


def _write_manifest(manifest: Dict[str, Any], processed_path: Path, artifacts_path: Path) -> None:
    _ensure_dir(processed_path.parent)
    _ensure_dir(artifacts_path.parent)
    s = json.dumps(manifest, indent=2, ensure_ascii=False)
    processed_path.write_text(s, encoding="utf-8")
    artifacts_path.write_text(s, encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> None:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description="Run dataset preparation steps 1–6 in one command.")
    ap.add_argument("--config", default=os.path.join("configs", "pipeline.yaml"), help="Pipeline YAML config")
    ap.add_argument("--force", action="store_true", help="Overwrite step outputs (re-run even if outputs exist)")
    ap.add_argument(
        "--start-at",
        default="extract",
        help=f"Stage to start at (default: extract). One of: {', '.join(STAGES)}",
    )
    ap.add_argument(
        "--stop-after",
        default="split",
        help=f"Stage to stop after (default: split). One of: {', '.join(STAGES)}. "
             "Use 'catalog-filter' to also build the CDE-side catalog and drop orphan queries.",
    )
    # Query-side / CDE-side catalog source overrides (config values used when omitted).
    ap.add_argument("--query-xml", default=None,
                    help="Override the query-side caDSR XML source folder (extract.input). "
                         "NOTE: the query stages are idempotent by output existence, so when you "
                         "point this at a NEW source you must also pass --force (or use clean/"
                         "distinct interim dirs), otherwise stale interim outputs are reused.")
    ap.add_argument("--catalog-xml", default=None,
                    help="CDE-side catalog caDSR XML folder (different from the query source). "
                         "Sets the catalog source_mode to 'separate'. NOTE: an existing catalog "
                         "parquet is reused unless --force is given, so pass --force (or use a "
                         "distinct catalog.out_parquet/interim_root) when changing this source.")
    ap.add_argument("--catalog-same-as-query", action="store_true",
                    help="Force the CDE-side catalog to be built from the query-side enriched master.")
    ap.add_argument("--catalog-eligibility", default=None,
                    help="Eligibility filter for the CDE-side catalog (none | production_cde_match).")
    ap.add_argument("--emit-splits", default=None,
                    help="Comma-separated query splits to emit in catalog-filter (e.g. 'train' "
                         "for train-only production). Defaults to the config's emit.splits.")
    ap.add_argument("--paper-config", default=None,
                    help="canonical final-system manifest; roots data/artifact outputs explicitly")

    args = ap.parse_args(argv_list)
    paper_mode = bool(args.paper_config)
    if paper_mode:
        from demap_repro.config.paper import (
            PaperConfigError, reject_conflicting_flags, validate_paper_config,
        )
        try:
            reject_conflicting_flags(
                argv_list, ["--config"], context="make-dataset --paper-config")
            paper = validate_paper_config(args.paper_config)
        except PaperConfigError as exc:
            ap.error(str(exc))
        args.config = paper["references"]["public_build"]

    source_root = source_repo_root()

    def _rooted(value, *, kind: str = "data") -> Path:
        path = Path(value)
        if path.is_absolute() or not paper_mode:
            return path
        if kind == "artifact" or (path.parts and path.parts[0] == "artifacts"):
            return artifact_root() / path
        if kind == "repo" or (path.parts and path.parts[0] == "configs"):
            return source_root / path
        return data_root() / path

    cfg_path = _rooted(args.config, kind="repo")
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    cfg_all = load_config(cfg_path)
    cfg = cfg_all.get("pipeline", cfg_all)  # allow either top-level or nested

    # Resolve stage bounds
    start = STAGE_ALIASES.get(str(args.start_at).strip(), None)
    stop = STAGE_ALIASES.get(str(args.stop_after).strip(), None)
    if start is None:
        raise SystemExit(f"Unknown --start-at: {args.start_at}. Expected one of: {', '.join(STAGES)}")
    if stop is None:
        raise SystemExit(f"Unknown --stop-after: {args.stop_after}. Expected one of: {', '.join(STAGES)}")

    start_i = STAGES.index(start)
    stop_i = STAGES.index(stop)
    if stop_i < start_i:
        raise SystemExit("--stop-after must be at or after --start-at")

    repo_root = source_root if paper_mode else Path.cwd()

    # Manifest locations (write both)
    processed_manifest = _rooted(cfg.get("manifests", {}).get("processed", os.path.join("data", "processed", "dataset_build_manifest.json")))
    artifacts_manifest = _rooted(cfg.get("manifests", {}).get("artifacts", os.path.join("artifacts", "manifests", "dataset_build_manifest.json")), kind="artifact")

    # Base paths
    extract_cfg = cfg.get("extract", {})
    merge_cfg = cfg.get("merge", {})
    enrich_cfg = cfg.get("enrich", {})
    q_cfg = cfg.get("build_queries", cfg.get("build-queries", {}))
    pv_summary_cfg = cfg.get("pv_summary", {})
    pairs_cfg = cfg.get("build_pairs", cfg.get("build-pairs", {}))
    split_cfg = cfg.get("split", {})

    # Query-side XML source: CLI --query-xml overrides extract.input.
    input_xml = _rooted(args.query_xml) if args.query_xml else _rooted(
        extract_cfg.get("input", os.path.join("data", "raw", "cadsr_xml"))
    )
    extracted_root = _rooted(extract_cfg.get("out_root", os.path.join("data", "interim", "cadsr_extracted")))
    merged_dir = _rooted(merge_cfg.get("merged_dir", os.path.join("data", "interim", "cadsr_merged")))
    merge_summaries_dir = _rooted(merge_cfg.get("summaries_dir", os.path.join("artifacts", "summaries", "cadsr_merge")), kind="artifact")

    enriched_parquet = _rooted(enrich_cfg.get("out_parquet", os.path.join("data", "processed", "cde_master_enriched.parquet")))

    alt_parquet = _rooted(q_cfg.get("alt_parquet", merged_dir / "cde_alternate_names.parquet"))
    ref_parquet = _rooted(q_cfg.get("ref_parquet", merged_dir / "cde_reference_documents.parquet"))
    queries_out_dir = _rooted(q_cfg.get("out_dir", os.path.join("artifacts", "summaries", "inwild_strict")), kind="artifact")
    queries_parquet = _rooted(q_cfg.get("out_parquet", os.path.join("data", "processed", "queries.parquet")))

    pairs_parquet = _rooted(pairs_cfg.get("out_parquet", os.path.join("data", "processed", "pairs.parquet")))

    splits_dir = _rooted(split_cfg.get("out_dir", os.path.join("data", "processed", "splits")))

    # Shared params
    language = str(enrich_cfg.get("language", q_cfg.get("language", "English")))
    pv_max_n = int(enrich_cfg.get("pv_max_n", enrich_cfg.get("pv_max", 10)))
    pv_summary_profile = str(pv_summary_cfg.get("pv_summary_profile", "cadsr"))
    pv_max_n_query = pv_summary_cfg.get("pv_max_n_query", None)
    pv_max_n_cde = pv_summary_cfg.get("pv_max_n_cde", None)
    pv_center_fraction = float(pv_summary_cfg.get("pv_center_fraction", 0.5))
    pv_min_n = int(pv_summary_cfg.get("pv_min_n", 2))
    pv_size_jitter = int(pv_summary_cfg.get("pv_size_jitter", 1))
    pv_generic_boost = float(pv_summary_cfg.get("pv_generic_boost", 0.10))
    pv_generic_cap = float(pv_summary_cfg.get("pv_generic_cap", 0.70))
    sde_generic_label_p = float(pv_summary_cfg.get("sde_generic_label_p", 0.30))
    pv_small_n_generic_threshold = float(pv_summary_cfg.get("pv_small_n_generic_threshold", 0.5))
    pv_max_resample_attempts = int(pv_summary_cfg.get("pv_max_resample_attempts", 3))
    pv_placeholder_token = str(pv_summary_cfg.get("pv_placeholder_token", "<MISSING_PV_SUMMARY>"))
    pv_diagnostics_summary = pv_summary_cfg.get("diagnostics_summary", None)
    pv_diagnostics_records = pv_summary_cfg.get("diagnostics_records", None)
    if pv_diagnostics_summary:
        pv_diagnostics_summary = _rooted(pv_diagnostics_summary, kind="artifact")
    if pv_diagnostics_records:
        pv_diagnostics_records = _rooted(pv_diagnostics_records, kind="artifact")

    # Build stages
    stages: List[Stage] = []

    # 1) extract
    extract_argv = ["--input", str(input_xml), "--out-root", str(extracted_root)]
    if "chunk_size" in extract_cfg:
        extract_argv += ["--chunk-size", str(int(extract_cfg.get("chunk_size")))]
    # Force maps to overwrite for extraction
    if args.force or bool(extract_cfg.get("overwrite", False)):
        extract_argv += ["--overwrite"]

    stages.append(
        Stage(
            name="extract",
            fn=extract_cadsr_xml.main,
            argv=extract_argv,
            done_check=lambda: _dir_has_parquet(extracted_root),
            outputs=[str(extracted_root)],
        )
    )

    # 2) merge
    merge_argv = ["--out-root", str(extracted_root), "--merged-dir", str(merged_dir), "--summaries-dir", str(merge_summaries_dir)]
    if "sample_size" in merge_cfg:
        merge_argv += ["--sample-size", str(int(merge_cfg.get("sample_size")))]
    if "seed" in merge_cfg:
        merge_argv += ["--seed", str(int(merge_cfg.get("seed")))]
    if "tables" in merge_cfg and merge_cfg.get("tables"):
        tables = merge_cfg.get("tables")
        if isinstance(tables, str):
            tables = [t.strip() for t in tables.split(",") if t.strip()]
        merge_argv += ["--tables"] + [str(t) for t in tables]

    stages.append(
        Stage(
            name="merge",
            fn=merge_cadsr_parquet.main,
            argv=merge_argv,
            done_check=lambda: _file_exists(merged_dir / "cde_master.parquet"),
            outputs=[str(merged_dir), str(merge_summaries_dir)],
        )
    )

    # 3) enrich
    enrich_argv = [
        "--merged-dir",
        str(merged_dir),
        "--out-parquet",
        str(enriched_parquet),
        "--pv-max-n",
        str(pv_max_n),
        "--pv-summary-profile",
        str(pv_summary_profile),
        "--pv-center-fraction",
        str(pv_center_fraction),
        "--pv-min-n",
        str(pv_min_n),
        "--pv-size-jitter",
        str(pv_size_jitter),
        "--pv-generic-boost",
        str(pv_generic_boost),
        "--pv-generic-cap",
        str(pv_generic_cap),
        "--sde-generic-label-p",
        str(sde_generic_label_p),
        "--pv-small-n-generic-threshold",
        str(pv_small_n_generic_threshold),
        "--pv-max-resample-attempts",
        str(pv_max_resample_attempts),
        "--pv-placeholder-token",
        str(pv_placeholder_token),
        "--language",
        str(language),
    ]
    if pv_max_n_query is not None:
        enrich_argv += ["--pv-max-n-query", str(int(pv_max_n_query))]
    if pv_max_n_cde is not None:
        enrich_argv += ["--pv-max-n-cde", str(int(pv_max_n_cde))]
    if pv_diagnostics_summary:
        enrich_argv += ["--pv-diagnostics-summary", str(pv_diagnostics_summary)]
    if pv_diagnostics_records:
        enrich_argv += ["--pv-diagnostics-records", str(pv_diagnostics_records)]

    stages.append(
        Stage(
            name="enrich",
            fn=enrich_master.main,
            argv=enrich_argv,
            done_check=lambda: _file_exists(enriched_parquet),
            outputs=[str(enriched_parquet)],
        )
    )

    # 4) build-queries
    q_argv = [
        "--alt-parquet",
        str(alt_parquet),
        "--ref-parquet",
        str(ref_parquet),
        "--out-dir",
        str(queries_out_dir),
        "--out-parquet",
        str(queries_parquet),
        "--language",
        str(language),
        "--pv-max-n",
        str(pv_max_n),
        "--pv-summary-profile",
        str(pv_summary_profile),
        "--pv-center-fraction",
        str(pv_center_fraction),
        "--pv-min-n",
        str(pv_min_n),
        "--pv-size-jitter",
        str(pv_size_jitter),
        "--pv-generic-boost",
        str(pv_generic_boost),
        "--pv-generic-cap",
        str(pv_generic_cap),
        "--sde-generic-label-p",
        str(sde_generic_label_p),
        "--pv-small-n-generic-threshold",
        str(pv_small_n_generic_threshold),
        "--pv-max-resample-attempts",
        str(pv_max_resample_attempts),
        "--pv-placeholder-token",
        str(pv_placeholder_token),
    ]
    if pv_max_n_query is not None:
        q_argv += ["--pv-max-n-query", str(int(pv_max_n_query))]
    if pv_max_n_cde is not None:
        q_argv += ["--pv-max-n-cde", str(int(pv_max_n_cde))]

    # Optional knobs for build-queries (strict allowlist)
    alt_allowlist = q_cfg.get("alt_allowlist", q_cfg.get("alt-allowlist", None))
    if alt_allowlist:
        alt_allowlist = _rooted(alt_allowlist, kind="repo")
        q_argv += ["--alt-allowlist", str(alt_allowlist)]

    refdoc_allowlist = q_cfg.get("refdoc_allowlist", q_cfg.get("refdoc-allowlist", None))
    if refdoc_allowlist:
        refdoc_allowlist = _rooted(refdoc_allowlist, kind="repo")
        q_argv += ["--refdoc-allowlist", str(refdoc_allowlist)]

    excluded_top_n = q_cfg.get("excluded_top_n", q_cfg.get("excluded-top-n", None))
    if excluded_top_n is not None:
        q_argv += ["--excluded-top-n", str(int(excluded_top_n))]

    if bool(q_cfg.get("skip_alt_candidate_inventory", False)):
        q_argv += ["--skip-alt-candidate-inventory"]
    # Treat build-queries outputs as stale if any of its primary inputs are newer.
    # This avoids confusing partial rebuilds where merge/enrich ran but query outputs
    # from an earlier run were silently reused.
    q_deps: List[Path] = [alt_parquet, ref_parquet, cfg_path]
    if alt_allowlist:
        q_deps.append(Path(alt_allowlist))
    if refdoc_allowlist:
        q_deps.append(Path(refdoc_allowlist))
    # ALT concat recipes are enabled by default via a repo-local allowlist; include
    # it as a dependency if present so edits trigger rebuilds.
    default_recipes = _rooted(
        Path("configs") / "allowlists" / "alt_query_recipes_current.csv", kind="repo")
    if default_recipes.exists() and default_recipes.is_file():
        q_deps.append(default_recipes)

    stages.append(
        Stage(
            name="build-queries",
            fn=inwild_queries.main,
            argv=q_argv,
            done_check=lambda: _is_up_to_date(queries_parquet, q_deps),
            outputs=[str(queries_parquet), str(queries_out_dir)],
        )
    )


    # 5) build-pairs
    bp_argv = ["--queries-parquet", str(queries_parquet), "--out-parquet", str(pairs_parquet)]

    stages.append(
        Stage(
            name="build-pairs",
            fn=build_pairs.main,
            argv=bp_argv,
            done_check=lambda: _is_up_to_date(pairs_parquet, [queries_parquet]),
            outputs=[str(pairs_parquet)],
        )
    )

    # 6) split
    split_seed = int(split_cfg.get("seed", 1234))
    val_frac = float(split_cfg.get("val_frac", 0.1))
    test_frac = float(split_cfg.get("test_frac", 0.1))
    train_frac = split_cfg.get("train_frac", None)

    sp_argv = ["--pairs-parquet", str(pairs_parquet), "--out-dir", str(splits_dir), "--seed", str(split_seed), "--val-frac", str(val_frac), "--test-frac", str(test_frac)]
    if train_frac is not None:
        sp_argv += ["--train-frac", str(float(train_frac))]

    stages.append(
        Stage(
            name="split",
            fn=split_policy.main,
            argv=sp_argv,
            done_check=lambda: _is_up_to_date(splits_dir / "train.parquet", [pairs_parquet]),
            outputs=[str(splits_dir)],
        )
    )

    # 7) catalog-filter (opt-in; runs only when reached via --stop-after/--start-at)
    catalog_cfg = cfg.get("catalog", {})
    emit_cfg = cfg.get("emit", {})

    # Catalog source mode: CLI overrides config. --catalog-xml implies 'separate';
    # --catalog-same-as-query forces 'same_as_query'.
    if args.catalog_same_as_query:
        catalog_source_mode = "same_as_query"
    elif args.catalog_xml:
        catalog_source_mode = "separate"
    else:
        catalog_source_mode = str(catalog_cfg.get("source_mode", "same_as_query"))

    catalog_xml_source = args.catalog_xml or catalog_cfg.get("xml_source", None)
    if catalog_xml_source:
        catalog_xml_source = _rooted(catalog_xml_source)
    catalog_eligibility = args.catalog_eligibility or str(catalog_cfg.get("eligibility", "production_cde_match"))
    catalog_out = _rooted(catalog_cfg.get("out_parquet", os.path.join("data", "processed", "cde_catalog_enriched.parquet")))
    catalog_interim_root = _rooted(catalog_cfg.get("interim_root", os.path.join("data", "interim", "cde_catalog")))

    emit_splits_list = (
        [s.strip() for s in str(args.emit_splits).split(",") if s.strip()]
        if args.emit_splits
        else list(emit_cfg.get("splits", ["train", "val_dev", "test", "val_train"]))
    )
    emit_filter_by_catalog = bool(emit_cfg.get("filter_by_catalog", True))
    emit_out_dir = _rooted(emit_cfg.get("out_dir", os.path.join("data", "processed", "splits_catalog_filtered")))
    emit_reports_dir = _rooted(emit_cfg.get("reports_dir", os.path.join("artifacts", "manifests", "catalog_filter")), kind="artifact")

    cf_argv = [
        "--config", str(cfg_path),
        "--splits-dir", str(splits_dir),
        "--emit-splits", ",".join(emit_splits_list),
        "--out-dir", str(emit_out_dir),
        "--reports-dir", str(emit_reports_dir),
        "--catalog-eligibility", str(catalog_eligibility),
        "--catalog-out", str(catalog_out),
        "--catalog-interim-root", str(catalog_interim_root),
    ]
    if catalog_source_mode == "separate":
        if not catalog_xml_source:
            raise SystemExit(
                "catalog source_mode='separate' requires --catalog-xml or catalog.xml_source in the config"
            )
        cf_argv += ["--catalog-xml", str(catalog_xml_source)]
    else:  # same_as_query
        cf_argv += ["--catalog-enriched", str(enriched_parquet)]
    if not emit_filter_by_catalog:
        cf_argv += ["--no-filter"]
    if args.force:
        cf_argv += ["--force"]

    stages.append(
        Stage(
            name="catalog-filter",
            fn=catalog_filter.main,
            done_check=lambda: _is_up_to_date(
                emit_out_dir / f"{emit_splits_list[0]}.parquet", [splits_dir / f"{emit_splits_list[0]}.parquet", catalog_out]
            ) if emit_splits_list else False,
            argv=cf_argv,
            outputs=[str(emit_out_dir), str(emit_reports_dir), str(catalog_out)],
        )
    )

    # Initialize manifest
    input_type, xml_list = _list_xml_inputs(input_xml)

    manifest: Dict[str, Any] = {
        "created_at": _now_iso(),
        "config": str(cfg_path),
        "repo_root": str(repo_root.resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "git": _git_info(repo_root),
        "inputs": {
            "input_xml": str(input_xml),
            "input_type": input_type,
            "n_xml_listed": int(len(xml_list)),
            "xml_list": xml_list[:1000],  # safety cap
        },
        "params": {
            "force": bool(args.force),
            "start_at": start,
            "stop_after": stop,
            "language": language,
            "pv_max_n": pv_max_n,
            "pv_summary_profile": pv_summary_profile,
            "pv_max_n_query": pv_max_n_query,
            "pv_max_n_cde": pv_max_n_cde,
            "pv_center_fraction": pv_center_fraction,
            "pv_min_n": pv_min_n,
            "pv_size_jitter": pv_size_jitter,
            "pv_generic_boost": pv_generic_boost,
            "pv_generic_cap": pv_generic_cap,
            "sde_generic_label_p": sde_generic_label_p,
            "pv_small_n_generic_threshold": pv_small_n_generic_threshold,
            "pv_max_resample_attempts": pv_max_resample_attempts,
            "pv_placeholder_token": pv_placeholder_token,
            "pv_diagnostics_summary": (str(pv_diagnostics_summary)
                                       if pv_diagnostics_summary else None),
            "pv_diagnostics_records": (str(pv_diagnostics_records)
                                       if pv_diagnostics_records else None),
            "split_seed": split_seed,
            "val_frac": val_frac,
            "test_frac": test_frac,
            "train_frac": train_frac,
        },
        "catalog": {
            "source_mode": catalog_source_mode,
            "query_xml_source": str(input_xml),
            "catalog_xml_source": (str(catalog_xml_source) if catalog_source_mode == "separate" else str(input_xml)),
            "eligibility": catalog_eligibility,
            "catalog_out": str(catalog_out),
        },
        "emit": {
            "splits": emit_splits_list,
            "filter_by_catalog": emit_filter_by_catalog,
            "out_dir": str(emit_out_dir),
            "reports_dir": str(emit_reports_dir),
        },
        "stages": {},
    }

    # Write initial manifest
    _write_manifest(manifest, processed_manifest, artifacts_manifest)

    # Execute stages within bounds
    for i, st in enumerate(stages):
        if i < start_i or i > stop_i:
            continue

        rec: Dict[str, Any] = {
            "name": st.name,
            "argv": st.argv,
            "outputs": st.outputs,
            "started_at": _now_iso(),
        }

        should_skip = (not args.force) and st.done_check()
        if should_skip:
            rec["status"] = "skipped"
            rec["ended_at"] = _now_iso()
            manifest["stages"][st.name] = rec
            _write_manifest(manifest, processed_manifest, artifacts_manifest)
            print(f"[skip] {st.name} (outputs already exist)")
            continue

        print(f"[run] {st.name} :: {' '.join(st.argv)}")
        t0 = time.time()
        try:
            st.fn(list(st.argv))
            rec["status"] = "ran"
        except SystemExit as e:
            # Propagate cleanly but write manifest first
            rec["status"] = "error"
            rec["error"] = f"SystemExit({e.code})"
            rec["ended_at"] = _now_iso()
            rec["elapsed_sec"] = float(time.time() - t0)
            manifest["stages"][st.name] = rec
            _write_manifest(manifest, processed_manifest, artifacts_manifest)
            raise
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = repr(e)
            rec["ended_at"] = _now_iso()
            rec["elapsed_sec"] = float(time.time() - t0)
            manifest["stages"][st.name] = rec
            _write_manifest(manifest, processed_manifest, artifacts_manifest)
            raise

        rec["ended_at"] = _now_iso()
        rec["elapsed_sec"] = float(time.time() - t0)

        # Best-effort row counts for key outputs
        counts: Dict[str, Any] = {}
        if st.name == "enrich":
            counts["cde_master_enriched_rows"] = _safe_parquet_rows(enriched_parquet)
        elif st.name == "build-queries":
            counts["queries_rows"] = _safe_parquet_rows(queries_parquet)
        elif st.name == "build-pairs":
            counts["pairs_rows"] = _safe_parquet_rows(pairs_parquet)
        elif st.name == "split":
            counts["train_rows"] = _safe_parquet_rows(splits_dir / "train.parquet")
            counts["val_rows"] = _safe_parquet_rows(splits_dir / "val.parquet")
            counts["test_rows"] = _safe_parquet_rows(splits_dir / "test.parquet")
            counts["holdout_org_rows"] = _safe_parquet_rows(splits_dir / "external_holdout_org.parquet")
            counts["holdout_standard_rows"] = _safe_parquet_rows(splits_dir / "external_holdout_standard.parquet")
            counts["holdout_refslice_rows"] = _safe_parquet_rows(splits_dir / "external_holdout_refslice.parquet")
        elif st.name == "merge":
            report = merge_summaries_dir / "merge_report.json"
            if report.exists():
                try:
                    counts["merge_report"] = json.loads(report.read_text(encoding="utf-8"))
                except Exception:
                    pass
        elif st.name == "catalog-filter":
            cf_manifest = emit_reports_dir / "catalog_filter_manifest.json"
            if cf_manifest.exists():
                try:
                    cm = json.loads(cf_manifest.read_text(encoding="utf-8"))
                    counts["catalog_source"] = cm.get("catalog_source")
                    counts["per_split_counts"] = cm.get("per_split_counts")
                    counts["dropped"] = cm.get("dropped")
                    counts["catalog_filter_manifest"] = str(cf_manifest)
                except Exception:
                    pass

        if counts:
            rec["counts"] = counts

        manifest["stages"][st.name] = rec
        _write_manifest(manifest, processed_manifest, artifacts_manifest)

    print(f"[done] Wrote manifests:\n  - {processed_manifest}\n  - {artifacts_manifest}")


if __name__ == "__main__":  # pragma: no cover
    main()
