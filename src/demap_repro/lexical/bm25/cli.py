"""Canonical BM25 paper CLI: ``demap paper bm25``.

    demap paper bm25 --validate-only     # protocol + existing artifact root, no scoring
    demap paper bm25 --dry-run           # resolve and print the plan, no scoring
    demap paper bm25 --run               # execute the canonical baseline

This is the documented canonical route. ``scripts/paper_bm25_canonical.py`` remains only as a
thin wrapper around it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from demap_repro.lexical.bm25 import run as R
from demap_repro.lexical.bm25.protocol import DEFAULT_PROTOCOL, load_protocol, repo_root


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        prog="demap paper bm25",
        description="Canonical BM25 lexical baseline (June production catalog, val_dev-only "
                    "representation selection, query-level public-id evaluation).")
    ap.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    ap.add_argument("--out-root", default=None,
                    help="output root; defaults to the protocol's output.root")
    ap.add_argument("--validate-only", action="store_true",
                    help="validate the protocol and an existing artifact root; no scoring")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and print the execution plan; no scoring")
    ap.add_argument("--run", action="store_true", help="execute the canonical baseline")
    ap.add_argument("--no-recompute", action="store_true",
                    help="--validate-only: skip recomputing metrics from the saved rankings")
    ap.add_argument("--no-check-rows", action="store_true",
                    help="skip parquet row-count checks (fast structural validation)")
    args = ap.parse_args(argv)

    n_modes = sum(bool(x) for x in (args.validate_only, args.dry_run, args.run))
    if n_modes != 1:
        raise SystemExit("choose exactly one of --validate-only, --dry-run, --run")

    proto = load_protocol(args.protocol)
    out_root = Path(args.out_root) if args.out_root else Path(proto["output"]["root"])
    if not out_root.is_absolute():
        out_root = repo_root() / out_root
    check_rows = not args.no_check_rows

    if args.validate_only:
        cat = R.verify_catalog(proto, check_rows=check_rows)
        splits = R.verify_splits(proto, check_rows=check_rows)
        report = R.validate_outputs(proto, out_root, recompute=not args.no_recompute)
        print(json.dumps({"mode": "validate-only", "protocol_id": proto["protocol_id"],
                          "catalog": cat, "splits": splits, "outputs": report,
                          "result": "PASS"}, indent=1))
        return

    if args.dry_run:
        print(json.dumps({"mode": "dry-run",
                          **R.execute(proto, out_root, dry_run=True, check_rows=check_rows)},
                         indent=1))
        return

    print(json.dumps({"mode": "run",
                      **R.execute(proto, out_root, dry_run=False, check_rows=check_rows)}, indent=1))


if __name__ == "__main__":
    main()
