#!/usr/bin/env python3
"""Reachable-denominator, by-dataset evaluation of an HGBC scored-rankings parquet.

Recomputes per-dataset Recall@K / MRR@100 from an existing
``hgbc_scored_rankings.parquet`` (written by ``train_hgbc_reranker.py``) — NO
retrain, NO rescore. For each split the denominator is the *reachable* query set
(``data/processed/eval_canonical/<split>.parquet``); a query is a hit@k iff its
gold (``is_label``) minimum ``hgbc_rank`` <= k, and queries with no gold candidate
in the pool are misses (rank inf). Primary ranking
is no-pin (uses ``hgbc_rank`` as written). MRR is MRR@100 to match
``train_hgbc_reranker.py``.

Optionally also emits the CIMAC v2 131 / 92 / 39 strata (gold-present /
exact-reachable / non-exact-reachable), reusing the canonical strata definition
from ``eval_hgbc_cimac131._strata``. The strata membership is gold-based and is
therefore unaffected by the CIMAC PV correction.

This module is import-friendly: ``by_dataset()`` and ``cimac_strata()`` can be
called directly (e.g. by ``build_medcpt_stage2_final_tables.py``).

Denominators
------------
The default query set is ``data/processed/eval_canonical`` — the canonical
evaluation population declared in ``configs/paper/eval_datasets_v1.yaml`` and the
one the manuscript reports against (Test 3,959 / CCTG 1,097 / OID ALT 1,766 /
CDASH 324 / GDC 72 / CIMAC 131).

It used to default to ``splits_v3_cdisc_reachable_2026-06-18_cimacpv``, a
pre-canonicalisation scheme that answers a different question, and the mismatch
was silent in two ways. Its ``test`` split holds 3,968 queries, not 3,959, and it
holds no ``cctg``, ``oid_alt``, ``cdash`` or ``gdc_combined`` parquet at all — it
carries the legacy ``external_holdout_org`` / ``external_holdout_refslice`` /
``external_holdout_gdc_*`` names those datasets were later carved from. Under the
canonical split names a run therefore printed a wrong denominator for ``test`` and
``val_dev`` and quietly *dropped four of the six* evaluation datasets.

No published number came from here: ``hgbc_eval_by_split.csv`` is written by
``demap_repro.reranker.train`` from the feature table, and reports the paper's
denominators. This was a stale default, not a corrupted result. A missing split
is now an error rather than a printed note, so the same mismatch cannot recur
silently.

The scored rankings you pass should already be on the corrected canonical fixed-K
base (``…_fixedk30_cimacpv``) for CIMAC.

Usage:
    PYTHONPATH=src python scripts/eval_hgbc_reachable_by_dataset.py \
        --scored-rankings <train/hgbc_scored_rankings.parquet> \
        --label medcpt_stage2_ce \
        --out-csv <by_dataset_metrics.csv> \
        --cimac-strata-out <cimac_strata_metrics.csv>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from demap_repro.utils.paths import data_root

REPO_ROOT = data_root()

#: The canonical evaluation population — see the module docstring.
DEFAULT_REACH_DIR = REPO_ROOT / "data/processed/eval_canonical"
REQUIRED_COLS = ("split", "query_id", "cde_id", "hgbc_rank", "is_label")
#: The paper's reporting depths (Table 4 quotes Recall@5; S5 quotes 1/5/10).
KS = (1, 5, 10)
CIMAC_STRATA_N = {"full_131": 131, "exact_92": 92, "non_exact_39": 39}


def _metrics(rank_map: dict, qset: set) -> dict:
    """Recall@K + MRR@100 over a query set; misses (absent gold) count as rank inf."""
    n = len(qset)
    r = np.array([rank_map.get(q, np.inf) for q in qset], dtype=float)
    d = {"n": n}
    for k in KS:
        d[f"R@{k}"] = round(float((r <= k).mean()), 4) if n else 0.0
    inv = np.where(np.isfinite(r) & (r <= 100), 1.0 / r, 0.0)
    d["MRR@100"] = round(float(inv.mean()), 4) if n else 0.0
    d["gold_in_pool"] = int(np.isfinite(r).sum())
    return d


def _rank_map(df: pd.DataFrame) -> dict:
    """query_id -> min hgbc_rank among that query's gold (is_label) candidate rows."""
    gold = df[df["is_label"] == True]  # noqa: E712 (pandas mask)
    if gold.empty:
        return {}
    return gold.groupby("query_id")["hgbc_rank"].min().astype(int).to_dict()


def _reachable_qids(reach_dir: Path, split: str):
    p = Path(reach_dir) / f"{split}.parquet"
    if not p.exists():
        return None
    return set(pd.read_parquet(p, columns=["query_id"])["query_id"].astype(str))


def by_dataset(scored: pd.DataFrame, reach_dir: Path, label: str,
               *, allow_missing: bool = False) -> pd.DataFrame:
    """Reachable by-dataset metrics for every split in ``scored``.

    A split present in ``scored`` with no query-set parquet under ``reach_dir``
    is an error: it means the denominators come from a different population than
    the rankings, and skipping it silently drops a whole evaluation dataset from
    the output table. Pass ``allow_missing=True`` (``--allow-missing-splits``)
    when that is genuinely intended, e.g. scoring ``val_train``/``val_dev``
    against the canonical six.
    """
    rows = []
    missing = []
    for split in sorted(scored["split"].unique()):
        qset = _reachable_qids(reach_dir, split)
        if qset is None:
            missing.append(split)
            continue
        sdf = scored[scored["split"] == split]
        rows.append({"split": split, "model": label, **_metrics(_rank_map(sdf), qset)})
    if missing and not allow_missing:
        raise FileNotFoundError(
            f"no query-set parquet under {reach_dir} for split(s) {missing}; "
            f"the denominators would come from a different population than the "
            f"rankings. Point --reachable-splits-dir at the matching query sets, "
            f"or pass --allow-missing-splits if the omission is intended.")
    if missing:
        print(f"  (skipped, no query-set parquet under {reach_dir}: {missing})")
    return pd.DataFrame(rows, columns=["split", "model", "n", "R@1", "R@5", "R@10",
                                       "MRR@100", "gold_in_pool"])


def cimac_strata(scored: pd.DataFrame, label: str,
                 cde_master: Path = None, interim: Path = None,
                 cimac_split: Path = None) -> pd.DataFrame:
    """CIMAC 131/92/39 strata metrics (gold-based membership; corrected ranks).

    Requires ``eval_hgbc_cimac131``, a research-repository script that was not
    migrated. :func:`by_dataset` — the function this module exists for — does not
    need it.
    """
    try:
        from eval_hgbc_cimac131 import _strata, DEF_ELIG, DEF_INTERIM, DEF_SPLIT
    except ImportError as exc:  # pragma: no cover - exercised by the guard test
        raise RuntimeError(
            "cimac_strata() needs 'eval_hgbc_cimac131', a research-repository "
            "script that was not migrated to this repository, so the CIMAC "
            "131/92/39 strata cannot be computed here. by_dataset() does not "
            "need it and is unaffected."
        ) from exc
    full, exa, nex = _strata(Path(cde_master or DEF_ELIG),
                             Path(interim or DEF_INTERIM),
                             Path(cimac_split or DEF_SPLIT))
    assert (len(full), len(exa), len(nex)) == (131, 92, 39), (len(full), len(exa), len(nex))
    c = scored[scored["split"] == "cimac_v2"]
    if c.empty:
        raise SystemExit("no cimac_v2 rows in the scored rankings; cannot compute strata")
    rm = _rank_map(c)
    rows = []
    for name, qs in [("full_131", full), ("exact_92", exa), ("non_exact_39", nex)]:
        rows.append({"stratum": name, "system": label, **_metrics(rm, qs)})
    return pd.DataFrame(rows, columns=["stratum", "system", "n", "R@1", "R@5", "R@10",
                                       "MRR@100", "gold_in_pool"])


def load_scored(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: scored rankings {path} missing columns: {missing}")
    df["query_id"] = df["query_id"].astype(str)
    return df


def main(argv=None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored-rankings", default=None)
    ap.add_argument("--reachable-splits-dir", default=str(DEFAULT_REACH_DIR))
    ap.add_argument("--label", default="hgbc")
    ap.add_argument("--allow-missing-splits", action="store_true",
                    help="skip splits with no query-set parquet instead of failing")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--cimac-strata-out", default=None,
                    help="if set and cimac_v2 present, also write 131/92/39 strata metrics")
    ap.add_argument("--paper-config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv_list)
    if args.paper_config:
        from demap_repro.config.paper import (
            PaperConfigError, reject_conflicting_flags, resolve_paper_artifact_path,
            resolve_paper_data_path, stage_job,
        )
        try:
            reject_conflicting_flags(
                argv_list,
                ["--scored-rankings", "--reachable-splits-dir", "--label",
                 "--allow-missing-splits", "--out-csv", "--cimac-strata-out"],
                context="eval-by-dataset --paper-config")
            eval_job = stage_job("evaluation", args.paper_config)
            hgbc_job = stage_job("hgbc", args.paper_config)
        except PaperConfigError as exc:
            ap.error(str(exc))
        out_root = resolve_paper_artifact_path(hgbc_job["out"])
        args.scored_rankings = str(out_root / "hgbc_scored_rankings.parquet")
        args.reachable_splits_dir = str(resolve_paper_data_path(eval_job["eval_dir"]))
        args.label = "hgbc_final_system_v1"
        args.allow_missing_splits = False
        args.out_csv = str(out_root / "hgbc_eval_by_dataset.csv")
        args.cimac_strata_out = None
        if args.dry_run:
            print(json.dumps({
                "scored_rankings": args.scored_rankings,
                "reachable_splits_dir": args.reachable_splits_dir,
                "datasets": eval_job["datasets"], "metrics": eval_job["metrics"],
                "out_csv": args.out_csv, "allow_missing_splits": False,
            }, indent=2, sort_keys=True))
            return 0
    elif args.dry_run:
        ap.error("--dry-run requires --paper-config")
    if not args.scored_rankings or not args.out_csv:
        ap.error("--scored-rankings and --out-csv are required")

    scored = load_scored(Path(args.scored_rankings))
    bd = by_dataset(scored, Path(args.reachable_splits_dir), args.label,
                    allow_missing=args.allow_missing_splits)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    bd.to_csv(args.out_csv, index=False)
    pd.set_option("display.width", 200)
    print(bd.to_string(index=False))
    print(f"\nwrote: {args.out_csv}")

    if args.cimac_strata_out and (scored["split"] == "cimac_v2").any():
        st = cimac_strata(scored, args.label)
        Path(args.cimac_strata_out).parent.mkdir(parents=True, exist_ok=True)
        st.to_csv(args.cimac_strata_out, index=False)
        print("\n" + st.to_string(index=False))
        print(f"wrote: {args.cimac_strata_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
