#!/usr/bin/env python3
"""Figure S6 — broad evidence-family ablation of the final HGBC.

S5.7's question is narrower than it first looks: *once the candidate pool is
fixed, what evidence drives the ranking?* It is not a statement about candidate
generation. Every condition scores the **identical** Stage-1 pool; only the
feature columns available to the reranker change. S5.6 and Table S5 answer the
generation question separately.

The partition
-------------
The 117 features are split into three mutually exclusive, exhaustive families,
grouped by the **type of evidence** they encode, not by which software component
emitted them:

======================================  ===  =====================================
Lexical/keyword-oriented evidence        98  G3 keyword exact/synonym tier, G4
                                             permissible-value overlap, G5
                                             query/CDE text characteristics, G6
                                             keyword-rule evidence, plus the two
                                             keyword-side source indicators
FT-MPNet evidence                        13  G2 scores/ranks + its source indicator
FT-MedCPT evidence                        6  G7 cross-encoder scores/ranks
======================================  ===  =====================================

**G4 and G5 are grouped with the lexical family by type of evidence, not by
provenance.** They are computed for every pooled candidate from the public
catalog and are *not* outputs of CDE Match-Fuzzy. That is precisely why the
family is called "Lexical/keyword-oriented evidence" and must not be renamed
"CDE Match-Fuzzy evidence".

The protocol, identical for every condition
-------------------------------------------
* the same fixed Stage-1 candidate pool (423,395 rows, row-multiset identical
  across conditions);
* the HGBC is **retrained from scratch** on ``val_train`` — no condition is a
  rescore of the shipped model;
* the shipped final hyperparameters are **frozen**, not re-selected
  (``max_iter=200, max_depth=3, learning_rate=0.05, min_samples_leaf=30,
  random_state=42``), which is what ``demap train-hgbc --fixed-config`` is for;
* only the available feature family changes. No candidate is ever removed.

The full-model condition reproduces the shipped HGBC byte-identically, which is
the control that makes the deltas meaningful.

Reading the results
-------------------
Read absolute values from ``broad_family_recall{5,1}_display.csv``, which are
rounded once from full precision. Do **not** re-round the 4-dp
``*_absolute.csv`` / ``*_delta.csv`` files to 3 dp: that double-rounds, the same
failure that put 0.769 and 0.175 into Table 4.

Two interpretations to avoid:

* **Do not** claim a small delta is explained by proximity to the candidate-pool
  ceiling. The ceiling bounds how much a model could *improve*, not how far it
  can *deteriorate* once information is removed. A near-ceiling dataset can
  still fall a long way.
* Arm-level effects are **conservative**: because the pool is a union of two
  arms, arm membership is partly inferable by negation from the complementary
  source indicator (``corr(in_biencoder_topk, in_keyword_topk) = -0.862``).
* GDC is n = 72, so one query is 0.014 Recall@5. Do not read per-dataset GDC
  deltas.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from demap_repro.utils.paths import artifact_root, repo_root

#: The three families, in the order Figure S6 plots them.
FAMILY_ORDER = ("lexical_keyword", "ft_mpnet", "ft_medcpt")

#: Condition name -> the family it removes. ``full`` removes nothing.
CONDITIONS = {
    "full": None,
    "minus_lexical_keyword": "lexical_keyword",
    "minus_ft_mpnet": "ft_mpnet",
    "minus_ft_medcpt": "ft_medcpt",
}

#: Shipped final hyperparameters, held frozen across every condition.
FROZEN_HYPERPARAMETERS = {
    "max_iter": 200, "max_depth": 3, "learning_rate": 0.05, "min_samples_leaf": 30,
}

#: The committed partition. Self-contained: it carries the member list of each
#: family, so it does not depend on the fine-grained ablation that produced the
#: seven-group map originally and is not part of the current manuscript.
FAMILY_MAP = repo_root() / "tests" / "fixtures" / "broad_family_map.json"

DEFAULT_FEATURE_TABLE = (
    "artifacts/final_reranker/hgbc_features_v2_eligible/"
    "feature_table_fixedk30_crossenc_v2.parquet")


def load_family_map(path: Path | None = None) -> dict:
    """The verified three-family partition of the 117 features."""
    return json.loads((path or FAMILY_MAP).read_text())


def verify_partition(family_map: dict) -> dict:
    """Check the partition is exhaustive, mutually exclusive, and 117 features.

    Returned so a caller can log it; raises on any violation, because a broken
    partition would silently make the deltas incomparable.
    """
    families = family_map["families"]
    members = {f: set(families[f]["members"]) for f in FAMILY_ORDER}

    sizes = {f: len(m) for f, m in members.items()}
    total = sum(sizes.values())
    if total != family_map["n_features"]:
        raise ValueError(f"families cover {total} features, expected {family_map['n_features']}")

    overlaps = {}
    for i, a in enumerate(FAMILY_ORDER):
        for b in FAMILY_ORDER[i + 1:]:
            shared = members[a] & members[b]
            if shared:
                overlaps[f"{a}|{b}"] = sorted(shared)
    if overlaps:
        raise ValueError(f"families are not mutually exclusive: {overlaps}")

    return {"sizes": sizes, "total": total, "overlap": 0}


def drop_list(family_map: dict, condition: str) -> list[str]:
    """Features to exclude for one condition: the baseline five, plus the family.

    The baseline five are excluded from the 122-column superset in every
    condition including ``full``, exactly as the shipped model excludes them.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {list(CONDITIONS)}")
    drop = list(family_map["baseline_excluded"])
    family = CONDITIONS[condition]
    if family is not None:
        drop += sorted(family_map["families"][family]["members"])
    return drop


def write_condition_specs(out_dir: Path, family_map: dict | None = None) -> dict[str, Path]:
    """Emit one drop-list JSON per condition, for ``train-hgbc --exclude-features-file``."""
    fm = family_map or load_family_map()
    verify_partition(fm)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for condition in CONDITIONS:
        path = out_dir / f"{condition}.json"
        path.write_text(json.dumps({
            "condition": condition,
            "removed_family": CONDITIONS[condition],
            "excluded_features": drop_list(fm, condition),
            "fixed_config": FROZEN_HYPERPARAMETERS,
        }, indent=2) + "\n")
        written[condition] = path
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None,
                    help="write one drop-list JSON per condition here")
    ap.add_argument("--family-map", default=None,
                    help=f"partition to use (default: {FAMILY_MAP})")
    ap.add_argument("--print-commands", action="store_true",
                    help="print the train-hgbc command for each condition and exit")
    args = ap.parse_args(argv)

    fm = load_family_map(Path(args.family_map) if args.family_map else None)
    report = verify_partition(fm)
    print(f"partition verified: {report['sizes']} total={report['total']} overlap=0")

    if args.print_commands:
        table = artifact_root() / DEFAULT_FEATURE_TABLE
        cfg = json.dumps(FROZEN_HYPERPARAMETERS, separators=(",", ":"))
        out = Path(args.out_dir or "<out-dir>")
        for condition in CONDITIONS:
            print(f"\n# {condition}"
                  f"{'' if CONDITIONS[condition] is None else '  (removes ' + CONDITIONS[condition] + ')'}")
            print(f"demap train-hgbc --feature-table {table} \\\n"
                  f"    --exclude-features-file {out / (condition + '.json')} \\\n"
                  f"    --fixed-config '{cfg}' \\\n"
                  f"    --out <conditions>/{condition}")
        return 0

    if args.out_dir:
        written = write_condition_specs(Path(args.out_dir), fm)
        for condition, path in written.items():
            n = len(json.loads(path.read_text())["excluded_features"])
            print(f"  {condition:<24} drops {n:>3} features -> {path}")
    else:
        print("\nNo --out-dir given; nothing written. "
              "Use --print-commands to see the per-condition training commands.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
