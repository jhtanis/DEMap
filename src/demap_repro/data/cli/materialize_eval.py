#!/usr/bin/env python3
"""Materialize the six canonical evaluation datasets into a stable location.

Writes ``$DEMAP_DATA_ROOT/data/processed/eval_canonical/<name>.parquet`` — the
path ``configs/paper/eval_datasets_v1.yaml`` declares and every evaluation stage
reads. Dropped-query reports go to ``artifacts/manifests/eval_canonical/``.

The six datasets arrive by two routes, because two of them cannot be rebuilt
from public inputs in the form the manuscript used:

**Four are derived here.** ``test``, ``cctg``, ``oid_alt`` and ``cdash`` are
built from the caDSR-derived splits and reachability-filtered: for every
dataset, drop any query row whose gold CDE is absent from the production
catalog. Unreachable golds are removed, never counted as misses.

**Two are shipped frozen.** ``cimac_v2`` and ``gdc_combined`` are copied from
``<repo>/data/frozen/`` after their pinned SHA-256 is verified. They are the
exact evaluation inputs the study used. CIMAC's permissible-value workbook
changed upstream and GDC's gold mappings are expert curation with no public
source, so rebuilding either would silently produce different numbers. See
``data/frozen/README.md``.

This replaces what used to be a manual ``cp``. Nothing is written back into
``data/frozen/``; it is a read-only distribution location.

Usage::

    demap materialize-eval                 # all six
    demap materialize-eval --frozen-only   # just the two shipped inputs
    demap materialize-eval --dry-run       # report, write nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from demap_repro.utils.paths import artifact_root, data_root, repo_root

#: Shipped frozen evaluation inputs: canonical name -> (filename, pinned sha256).
#: The digests are the study's own; a mismatch means the artifact was replaced.
FROZEN_INPUTS = {
    "cimac_v2": ("cimac_v2.parquet",
                 "1d0797330864cb8a73d277c1885fa5bba41a63d3a011cfaaa148e59c6dce4fe0"),
    "gdc_combined": ("gdc_combined.parquet",
                     "289c4f5fc966748c54fc3d4c8c0edda28937b55910796a65db7834034bf86f21"),
}

#: Datasets built here from the caDSR-derived splits.
DERIVED = ("test", "cctg", "oid_alt", "cdash")


def frozen_dir() -> Path:
    """The shipped artifacts, resolved against the source tree, not the CWD."""
    return repo_root() / "data" / "frozen"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def materialize_frozen(out_dir: Path, *, dry_run: bool = False,
                       src_dir: Path | None = None) -> list[dict]:
    """Copy the shipped frozen evaluation inputs into ``out_dir``.

    Verifies each artifact's pinned digest before copying, creates the
    destination directory, and is idempotent: a destination that already
    matches the pinned digest is left alone. A destination that *differs* is an
    error rather than an overwrite — it means a different evaluation population
    is in place, and silently replacing it would change reported numbers
    without saying so.
    """
    src_dir = src_dir or frozen_dir()
    rows = []
    for name, (filename, want) in FROZEN_INPUTS.items():
        src = src_dir / filename
        if not src.is_file():
            sys.exit(f"ERROR: shipped frozen input missing: {src}\n"
                     f"It should be committed at data/frozen/{filename}.")
        got = _sha256(src)
        if got != want:
            sys.exit(f"ERROR: {src} does not match its pinned digest.\n"
                     f"  expected {want}\n  got      {got}\n"
                     f"This is the exact evaluation input the manuscript used; "
                     f"do not substitute it.")

        dst = out_dir / f"{name}.parquet"
        if dst.exists():
            dst_digest = _sha256(dst)
            if dst_digest == want:
                print(f"{name:14s} already materialized, digest matches — skipping")
                rows.append(dict(dataset=name, action="skipped", source=f"data/frozen/{filename}",
                                 sha256=want))
                continue
            sys.exit(
                f"ERROR: {dst} exists and differs from the shipped frozen input.\n"
                f"  shipped  {want}\n  on disk  {dst_digest}\n"
                f"Refusing to overwrite. That file evaluates a different population "
                f"than the manuscript did. Move it aside, then re-run.")

        print(f"{name:14s} frozen        <- data/frozen/{filename}"
              f"{'  (dry-run)' if dry_run else ''}")
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if _sha256(dst) != want:
                sys.exit(f"ERROR: copy of {dst} does not match the pinned digest")
        rows.append(dict(dataset=name, action="copied", source=f"data/frozen/{filename}",
                         sha256=want))
    return rows


def _source_frames(splits: Path, *, logical_prefix: str = "data/processed/splits"):
    """The four datasets derived from the caDSR splits."""
    org = pd.read_parquet(splits / "external_holdout_org.parquet")
    return {
        "test": (pd.read_parquet(splits / "test.parquet"),
                 [f"{logical_prefix}/test.parquet"], "none"),
        "cctg": (org[org["family"] == "CCTG"].copy(),
                 [f"{logical_prefix}/external_holdout_org.parquet"], "family == 'CCTG'"),
        "oid_alt": (org[org["family"] == "OID"].copy(),
                    [f"{logical_prefix}/external_holdout_org.parquet"], "family == 'OID'"),
        "cdash": (pd.read_parquet(splits / "external_holdout_refslice.parquet"),
                  [f"{logical_prefix}/external_holdout_refslice.parquet"], "none"),
    }


def main(argv=None):
    argv_list = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written; write nothing")
    ap.add_argument("--frozen-only", action="store_true",
                    help="materialize only the two shipped frozen inputs (needs no splits tree)")
    ap.add_argument("--out-dir", default=None,
                    help="destination (default: $DEMAP_DATA_ROOT/data/processed/eval_canonical)")
    ap.add_argument("--splits-dir", default=None,
                    help="reachable caDSR split source (default: data/processed/splits)")
    ap.add_argument("--paper-config", default=None,
                    help="use the canonical paper build/materialization paths")
    args = ap.parse_args(argv_list)

    if args.paper_config:
        from demap_repro.config.paper import (
            PaperConfigError, reject_conflicting_flags, resolve_paper_data_path,
            validate_paper_config,
        )
        try:
            reject_conflicting_flags(
                argv_list, ["--out-dir", "--splits-dir"],
                context="materialize-eval --paper-config")
            paper = validate_paper_config(args.paper_config)
        except PaperConfigError as exc:
            ap.error(str(exc))
        args.out_dir = str(resolve_paper_data_path(
            paper["data"]["build"]["canonical_eval_dir"]))
        args.splits_dir = str(resolve_paper_data_path(
            paper["data"]["build"]["reachable_splits_dir"]))

    root = data_root()
    out_data = Path(args.out_dir) if args.out_dir else root / "data/processed/eval_canonical"
    if not out_data.is_absolute():
        out_data = root / out_data
    out_manifest = artifact_root() / "artifacts/manifests/eval_canonical"

    frozen_rows = materialize_frozen(out_data, dry_run=args.dry_run)

    if args.frozen_only:
        print(f"\n{'DRY-RUN (nothing written).' if args.dry_run else f'Frozen inputs -> {out_data}/'}")
        return 0

    splits = Path(args.splits_dir) if args.splits_dir else root / "data/processed/splits"
    if not splits.is_absolute():
        splits = root / splits
    prod = root / ("data/processed/cadsr_xml_2026-06-18/"
                   "cde_master_enriched_eval_production_cde_match.parquet")
    for p in (splits, prod):
        if not p.exists():
            sys.exit(f"ERROR: {p} not found.\n"
                     f"Set DEMAP_DATA_ROOT to a prepared data tree, or use --frozen-only "
                     f"to materialize just the two shipped evaluation inputs.")

    prod_ids = set(pd.read_parquet(prod)["cde_id"].astype(str))
    summary = list(frozen_rows)
    logical_prefix = ("data/processed/splits_catalog_filtered"
                      if args.paper_config else "data/processed/splits")
    for name, (df, source, rule) in _source_frames(
            splits, logical_prefix=logical_prefix).items():
        assert "query_text_q3" in df.columns, f"{name}: missing query_text_q3"
        assert "cde_id" in df.columns, f"{name}: missing cde_id"
        orig = len(df)
        reach_mask = df["cde_id"].astype(str).isin(prod_ids)
        kept = df[reach_mask].reset_index(drop=True)
        dropped = df[~reach_mask].reset_index(drop=True)
        summary.append(dict(dataset=name, original_rows=orig, reachable_rows=len(kept),
                            dropped_rows=len(dropped), query_text_q3=True,
                            source=source, rule=rule))
        print(f"{name:14s} original={orig:5d}  reachable={len(kept):5d}  dropped={len(dropped):4d}")
        if args.dry_run:
            continue
        out_data.mkdir(parents=True, exist_ok=True)
        out_manifest.mkdir(parents=True, exist_ok=True)
        dst = out_data / f"{name}.parquet"
        if dst.exists():
            sys.exit(f"ERROR: refusing to overwrite {dst}. Move it aside, then re-run.")
        kept.to_parquet(dst, index=False)
        if len(dropped):
            qid = ("query_id" if "query_id" in dropped.columns
                   else ("pair_id" if "pair_id" in dropped.columns else dropped.columns[0]))
            rep = dropped[[c for c in [qid, "query_source", "family", "cde_id", "cde_publicid"]
                           if c in dropped.columns]].copy()
            rep.insert(0, "dataset", name)
            rep["reason"] = "gold_not_in_production_catalog"
            rep.to_csv(out_manifest / f"{name}_dropped_queries.csv", index=False)

    if not args.dry_run:
        out_manifest.mkdir(parents=True, exist_ok=True)
        (out_manifest / "reachability_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        pd.DataFrame([{k: v for k, v in r.items() if k != "source"} for r in summary]).to_csv(
            out_manifest / "reachability_summary.csv", index=False)
    print("\nDRY-RUN (nothing written)." if args.dry_run
          else f"\nWrote 6 parquets -> {out_data}/  + reports -> {out_manifest}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
