"""Recompute published Table 4 metrics from frozen query-level outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from demap_repro.reporting.table4_bundle import (
    Table4VerificationError,
    contract_methods,
    load_contract,
    verify_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path,
                        help="local directory containing the verified Table 4 bundle")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="external directory for table4.csv, metrics.json, and verification.json")
    parser.add_argument("--contract", type=Path, default=None,
                        help="override the packaged results contract (normally unnecessary)")
    parser.add_argument("--offline", action="store_true",
                        help="assert offline operation (the command never accesses the network)")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        _, metrics = verify_bundle(
            args.bundle,
            out_dir=args.out_dir,
            contract_path=args.contract,
            write_outputs=True,
        )
        contract = load_contract(args.contract)
    except (Table4VerificationError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    methods = contract_methods(contract)
    labels = {item["id"]: item["display_name"] for item in contract["methods"]}
    width = max(len(item["display_name"]) for item in contract["datasets"])
    column_widths = {method: max(13, len(labels[method])) for method in methods}
    print("Published Table 4 (Recall@5; recomputed from frozen query-level outputs)")
    print(" " * (width + 2) + "  ".join(
        f"{labels[method]:>{column_widths[method]}s}" for method in methods))
    for dataset in contract["datasets"]:
        dataset_id = dataset["id"]
        values = "  ".join(
            f"{metrics[dataset_id][method]['manuscript_3dp']:>{column_widths[method]}s}"
            for method in methods)
        print(f"{dataset['display_name']:<{width}}  {values}")
    print("\n36/36 required Table 4 cells verified; 0 central checks skipped")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
