"""Paper-mode adapter for the migration-locked CDE Match candidate builder.

The underlying builder remains byte/AST comparable to the research source.
This adapter resolves the canonical manuscript job, rejects competing scientific
flags, and delegates the resulting explicit argument list to that builder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from demap_repro.lexical.cde_match import build_candidates


def main(argv=None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if "--paper-config" not in argv_list:
        return build_candidates.main(argv_list)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper-config", required=True)
    ap.add_argument("--paper-dataset", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args, unknown = ap.parse_known_args(argv_list)
    if unknown:
        ap.error(
            "paper mode owns the candidate-builder configuration; remove conflicting "
            f"or unknown argument(s): {unknown}")

    from demap_repro.config.paper import (
        PaperConfigError,
        resolve_paper_artifact_path,
        resolve_paper_data_path,
        stage_job,
    )
    try:
        job = stage_job("cde_match_fuzzy", args.paper_config)
    except PaperConfigError as exc:
        ap.error(str(exc))
    if args.paper_dataset is None:
        if args.dry_run:
            print(json.dumps(job, indent=2, sort_keys=True))
            return 0
        ap.error("--paper-dataset is required to execute paper mode")
    if args.paper_dataset not in job["allowance"]:
        ap.error(f"unknown paper dataset {args.paper_dataset!r}")

    split = args.paper_dataset
    rate = float(job["allowance"][split])
    split_dir_key = (
        "training_split_dir" if split in {"train", "val_train", "val_dev"}
        else "split_dir"
    )
    resolved = {
        "dataset": split,
        "split_dir": str(resolve_paper_data_path(job[split_dir_key])),
        "cde_master": str(resolve_paper_data_path(job["cde_master"])),
        "alt_names": str(resolve_paper_data_path(job["alt_names"])),
        "permissible_values": str(resolve_paper_data_path(job["permissible_values"])),
        "reference_documents": str(resolve_paper_data_path(job["reference_documents"])),
        "allowance": rate,
        "exact_match_seed": int(job["exact_match_seed"]),
        "fuzzy_fallback": job["fuzzy_fallback"],
        "eligibility": job["eligibility"],
        "out_dir": str(resolve_paper_artifact_path(
            job["out_dir_cadsr"] if rate < 1.0 else job["out_dir_external"])),
        "tag": "fuzzy_a070" if rate < 1.0 else "fuzzy_a10",
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return 0

    delegated = [
        "--splits", split,
        "--split-dir", resolved["split_dir"],
        "--cde-master", resolved["cde_master"],
        "--alt-names", resolved["alt_names"],
        "--pvs", resolved["permissible_values"],
        "--ref-docs", resolved["reference_documents"],
        "--exact-query-match-allow-rate", str(rate),
        "--exact-match-seed", str(resolved["exact_match_seed"]),
        "--exact-match-scope", "gold",
        "--eligibility", resolved["eligibility"],
        "--fuzzy-fallback", resolved["fuzzy_fallback"],
        "--fuzzy-top-k-per-rule", str(job["top_k_per_rule"]),
        "--out-dir", resolved["out_dir"],
        "--tag", resolved["tag"],
    ]
    if args.overwrite:
        delegated.append("--overwrite")
    # The migration-locked builder's temp helper uses this module-level legacy
    # default. Point it at the already-resolved paper artifact directory before
    # delegation so paper mode never writes under a stale data-root default.
    build_candidates.DEFAULT_OUT_DIR = Path(resolved["out_dir"])
    return build_candidates.main(delegated)


if __name__ == "__main__":
    raise SystemExit(main())
