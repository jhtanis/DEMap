#!/usr/bin/env python3
"""Remove evaluation queries that recur in training with the same gold CDE.

S1.5 reports two kinds of train/evaluation overlap. Query-identifier overlap is
almost nil. The one that matters is **content** overlap: an evaluation query whose
normalized text already appears in Train mapped to the same gold CDE. Such a query
tests memorization rather than generalization, so S6.1 and Table S6 re-report every
external result with those queries removed.

The rule
--------
Exclude an external evaluation query ``q`` when there exists a training pair ``t``
with::

    norm(q.query_text_raw) == norm(t.query_text_raw)
      AND publicid(q.cde_publicid) == publicid(t.cde_publicid)

    norm(x)     = collapse_ws(sub(r'[^a-z0-9]+', ' ', lower(x)))
    publicid(x) = x.split('::')[0]

Removal is whole-query: every row of an excluded ``query_id`` goes, so a
multi-gold query is never left with a partial gold set.

The internal **Test** split passes through untouched. It is in-distribution by
design — filtering it would not measure leakage, it would just shrink the split.

A second clause, ``q.query_id in train.query_id``, is evaluated and reported but
never fires on the canonical inputs: all 51 removals come from the text+gold
clause. The effective criterion is therefore exactly the one S1.5 states. The
clause is retained because it is part of the executed rule and because a future
benchmark rebuild could reuse identifiers.

Migrated verbatim (logic unchanged) from
``.scratch/demap/paper_v11_validation/build_eval_canonical_v2.py`` — the confirmed
producer of ``data/processed/eval_canonical_v2/``. Changes here: hardcoded absolute
roots became arguments, the rule became the importable
:func:`build_leakage_filtered_sets`, and the refuse-to-overwrite guard is now an
explicit ``--overwrite`` flag rather than an unconditional abort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from demap_repro.utils.paths import data_root

__all__ = [
    "EXTERNAL_DATASETS",
    "PASSTHROUGH_DATASETS",
    "PAPER_COUNTS",
    "norm",
    "publicid",
    "build_leakage_filtered_sets",
]

#: Distribution-shifted holdouts the filter applies to.
EXTERNAL_DATASETS = ("cctg", "oid_alt", "cdash", "gdc_combined", "cimac_v2")

#: In-distribution split copied through unchanged.
PASSTHROUGH_DATASETS = ("test",)

#: Query counts before and after filtering, as reported in S1.5 / Table S6.
PAPER_COUNTS = {
    "cctg": (1097, 8, 1089),
    "oid_alt": (1766, 7, 1759),
    "cdash": (324, 14, 310),
    "gdc_combined": (72, 4, 68),
    "cimac_v2": (131, 18, 113),
    "test": (3959, 0, 3959),
}

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def norm(text: object) -> str:
    """Lowercase, collapse every non-alphanumeric run to a single space, strip."""
    return _WS.sub(" ", _NON_ALNUM.sub(" ", str(text).lower())).strip()


def publicid(cde_id: object) -> str:
    """CDE public identifier: the part before the ``::`` version suffix."""
    return str(cde_id).split("::")[0].strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_leakage_filtered_sets(
    src_dir: Path,
    train_path: Path,
    *,
    datasets: Sequence[str] = EXTERNAL_DATASETS,
    passthrough: Sequence[str] = PASSTHROUGH_DATASETS,
    out_dir: Optional[Path] = None,
    manifest_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Tuple[List[Dict], pd.DataFrame]:
    """Apply the rule and, when ``out_dir`` is given, write the filtered sets.

    Returns ``(summary_rows, removed_queries)``. With ``out_dir=None`` nothing is
    written, which is how the regression tests replay the rule against the
    published inputs without touching them.
    """
    train = pd.read_parquet(train_path)
    train_key: Dict[Tuple[str, str], Tuple[str, str, str]] = {}
    for text, pid, qid, family, source in zip(
        train["query_text_raw"].map(norm),
        train["cde_publicid"].map(publicid),
        train["query_id"].astype(str),
        train["family"],
        train["query_source"],
    ):
        train_key.setdefault((text, pid), (qid, str(family), str(source)))
    train_qids = set(train["query_id"].astype(str))

    summary: List[Dict] = []
    removed_rows: List[Dict] = []

    for name in list(datasets) + list(passthrough):
        source_path = src_dir / f"{name}.parquet"
        df = pd.read_parquet(source_path)
        qids = df["query_id"].astype(str)

        reasons: Dict[str, List[str]] = {}
        if name in passthrough:
            drop_q: set = set()
        else:
            keys = list(zip(df["query_text_raw"].map(norm),
                            df["cde_publicid"].map(publicid)))
            for i, key in enumerate(keys):
                qid = qids.iloc[i]
                if key in train_key:
                    reasons.setdefault(qid, []).append(
                        f"norm_text+gold_in_train(gold={key[1]},"
                        f"train_qid={train_key[key][0]},"
                        f"train_family={train_key[key][1]})")
                if qid in train_qids:
                    reasons.setdefault(qid, []).append("query_id_in_train")
            drop_q = set(reasons)

        keep = df[~qids.isin(drop_q)].reset_index(drop=True)
        dropped = df[qids.isin(drop_q)].reset_index(drop=True)

        summary.append(dict(
            dataset=name,
            source=str(source_path),
            source_sha256=_sha256(source_path),
            v1_rows=len(df), v1_queries=int(qids.nunique()),
            removed_queries=len(drop_q), removed_rows=len(dropped),
            v2_rows=len(keep), v2_queries=int(keep["query_id"].nunique()),
            rule=("passthrough (in-distribution)" if name in passthrough else "Rule E"),
        ))
        for _, row in dropped.iterrows():
            removed_rows.append(dict(
                dataset=name,
                query_id=str(row["query_id"]),
                query_text_raw=str(row["query_text_raw"]),
                cde_publicid=str(row["cde_publicid"]),
                family=str(row.get("family", "")),
                reason=" ; ".join(sorted(set(reasons[str(row["query_id"])]))),
            ))

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{name}.parquet"
            if out_path.exists() and not overwrite:
                raise FileExistsError(f"refusing to overwrite {out_path}; pass overwrite=True")
            keep.to_parquet(out_path, index=False)

    removed = pd.DataFrame(removed_rows)

    if out_dir is not None and manifest_dir is not None:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        removed.to_csv(manifest_dir / "removed_queries.csv", index=False)
        for row in summary:
            row["v2_sha256"] = _sha256(out_dir / f"{row['dataset']}.parquet")
        (manifest_dir / "eval_canonical_v2_manifest.json").write_text(json.dumps({
            "rule": __doc__,
            "train_reference": str(train_path),
            "train_sha256": _sha256(train_path),
            "datasets": summary,
        }, indent=2))

    return summary, removed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = data_root()
    ap.add_argument("--src-dir", default=str(root / "data/processed/eval_canonical"))
    ap.add_argument("--train", default=str(root / "data/processed/splits/train.parquet"))
    ap.add_argument("--out-dir", default=str(root / "data/processed/eval_canonical_v2"))
    ap.add_argument("--manifest-dir", default=str(root / "artifacts/manifests/eval_canonical_v2"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = ap.parse_args(argv)

    summary, _removed = build_leakage_filtered_sets(
        Path(args.src_dir), Path(args.train),
        out_dir=None if args.dry_run else Path(args.out_dir),
        manifest_dir=None if args.dry_run else Path(args.manifest_dir),
        overwrite=args.overwrite,
    )
    print(pd.DataFrame(summary)[[
        "dataset", "v1_rows", "v1_queries", "removed_queries",
        "removed_rows", "v2_rows", "v2_queries"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
