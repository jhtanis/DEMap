#!/usr/bin/env python3
"""Reachable-denominator, by-dataset evaluation of an HGBC scored-rankings parquet.

Recomputes per-dataset Recall@K / MRR@100 from an existing
``hgbc_scored_rankings.parquet`` (written by ``train_hgbc_reranker.py``) — NO
retrain, NO rescore. For each split the denominator is the *reachable* query set
(``data/processed/splits_v3_cdisc_reachable_2026-06-18_cimacpv/<split>.parquet``);
a query is a hit@k iff its gold (``is_label``) minimum ``hgbc_rank`` <= k, and
queries with no gold candidate in the pool are misses (rank inf). Primary ranking
is no-pin (uses ``hgbc_rank`` as written). MRR is MRR@100 to match
``train_hgbc_reranker.py``.

Optionally also emits the CIMAC v2 131 / 92 / 39 strata (gold-present /
exact-reachable / non-exact-reachable), reusing the canonical strata definition
from ``eval_hgbc_cimac131._strata``. The strata membership is gold-based and is
therefore unaffected by the CIMAC PV correction.

This module is import-friendly: ``by_dataset()`` and ``cimac_strata()`` can be
called directly (e.g. by ``build_medcpt_stage2_final_tables.py``).

Defaults assume the corrected CIMAC-PV world:
  * reachable splits dir: ``…_reachable_2026-06-18_cimacpv``
  * the scored rankings you pass should already be on the corrected canonical
    fixed-K base (``…_fixedk30_cimacpv``) for CIMAC.

Usage:
    PYTHONPATH=src python scripts/eval_hgbc_reachable_by_dataset.py \
        --scored-rankings <train/hgbc_scored_rankings.parquet> \
        --label medcpt_stage2_ce \
        --out-csv <by_dataset_metrics.csv> \
        --cimac-strata-out <cimac_strata_metrics.csv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from demap_repro.utils.paths import data_root

REPO_ROOT = data_root()

DEFAULT_REACH_DIR = REPO_ROOT / "data/processed/splits_v3_cdisc_reachable_2026-06-18_cimacpv"
REQUIRED_COLS = ("split", "query_id", "cde_id", "hgbc_rank", "is_label")
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


def by_dataset(scored: pd.DataFrame, reach_dir: Path, label: str) -> pd.DataFrame:
    """Reachable by-dataset metrics for every split that has a reachable parquet."""
    rows = []
    for split in sorted(scored["split"].unique()):
        qset = _reachable_qids(reach_dir, split)
        if qset is None:
            print(f"  (skip {split}: no reachable split parquet under {reach_dir})")
            continue
        sdf = scored[scored["split"] == split]
        rows.append({"split": split, "model": label, **_metrics(_rank_map(sdf), qset)})
    return pd.DataFrame(rows, columns=["split", "model", "n", "R@1", "R@5", "R@10",
                                       "MRR@100", "gold_in_pool"])


def cimac_strata(scored: pd.DataFrame, label: str,
                 cde_master: Path = None, interim: Path = None,
                 cimac_split: Path = None) -> pd.DataFrame:
    """CIMAC 131/92/39 strata metrics (gold-based membership; corrected ranks)."""
    from eval_hgbc_cimac131 import _strata, DEF_ELIG, DEF_INTERIM, DEF_SPLIT
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored-rankings", required=True)
    ap.add_argument("--reachable-splits-dir", default=str(DEFAULT_REACH_DIR))
    ap.add_argument("--label", default="hgbc")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--cimac-strata-out", default=None,
                    help="if set and cimac_v2 present, also write 131/92/39 strata metrics")
    args = ap.parse_args(argv)

    scored = load_scored(Path(args.scored_rankings))
    bd = by_dataset(scored, Path(args.reachable_splits_dir), args.label)
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
