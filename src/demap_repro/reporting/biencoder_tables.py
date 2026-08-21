#!/usr/bin/env python3
"""Regenerate the publication bi-encoder tables (Tables C1/C2/C3, E, F) + raw-CE table.

Durable promotion of the one-off scratch aggregator. **Reads existing artifacts only** —
it never recomputes model scores, never launches jobs, never touches
``artifacts_v3_cdisc``, and writes nothing under ``/tmp``. All **four** bi-encoders are
preserved (all-mpnet, sapbert, biosimcse, pubmedbert); the pipeline down-selects to one
production embedder but the publication comparison keeps all four.

Outputs (default under ``artifacts/publication_artifact_package_20260713/tables/``):
  bi_encoder/
    biencoder_offtheshelf_best_by_model.csv           (C1) off-the-shelf best cell/model
    biencoder_phase1_best_by_model.csv                (C2) phase-1 winner canonical eval
    biencoder_phase2_best_by_model.csv                (C3) phase-2 winner canonical eval
    offtheshelf_reppair_<slug>__val_dev_recallat5.csv (F)  4x10 query x CDE rep grid
    offtheshelf_reppair_<slug>__test_recallat5.csv    (F)  4x10 grid (test)
  pretraining_representation/
    phase0_representation_sweep_best_by_model.csv     (E)  best representation per model
  cross_encoder/
    ce_raw_offtheshelf_val_dev.csv                    raw CE (val_dev only — see note)

RAW CROSS-ENCODER CANONICAL-SCORE GAP (documented, NOT filled here): no standalone
raw/off-the-shelf CE evaluation exists across the 6 canonical datasets. The only
untrained-backbone numbers are the epoch-0 entries in each CE run's
``training_summary.json`` history, evaluated on the val_dev se20∪kwfuzzy10 pool
(recall@1/5/10 + mrr@100; no recall@100). This script emits exactly those and labels
them clearly; it does NOT run any new CE evaluation. Fill the gap only via a separately
approved eval job.

Usage::
    PYTHONPATH=src python scripts/aggregate_publication_biencoder_tables.py
    PYTHONPATH=src python scripts/aggregate_publication_biencoder_tables.py --out-root <dir> --check
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from demap_repro.utils.paths import data_root

#: Data and artifact tree (see demap_repro.utils.paths). Formerly
#: ``parents[1]``, which after packaging resolves inside ``src/demap_repro``.
REPO_ROOT = data_root()

# The four bi-encoders (publication comparison set). Order is stable & load-bearing.
MODELS: Dict[str, str] = {
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "sapbert-pubmedbert-fulltext": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "biosimcse-biolinkbert-base": "kamalkraj/BioSimCSE-BioLinkBERT-BASE",
    "pubmedbert-base-embeddings": "NeuML/pubmedbert-base-embeddings",
}
CANON = ["test", "cctg", "oid_alt", "cdash", "gdc_combined", "cimac_v2"]
QUERY_VARIANTS = ["Q1", "Q2", "Q3", "Q4"]
CDE_RECIPES = ["v1", "v3", "v5", "v6", "v2b", "v1_v6", "v1_v6_v2b",
               "v1_v6_v2b_v3", "v1_v6_v2b_v5", "v1_v6_v2b_v3_v5"]
CE_MODELS = ["cross-encoder_ms-marco-MiniLM-L-6-v2", "ncbi_MedCPT-Cross-Encoder",
             "BAAI_bge-reranker-base"]
CE_PRETTY = {
    "cross-encoder_ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "ncbi_MedCPT-Cross-Encoder": "ncbi/MedCPT-Cross-Encoder",
    "BAAI_bge-reranker-base": "BAAI/bge-reranker-base",
}

# --- Source artifact locations (all repo-relative; asserted present) ----------------
PHASE0_ROOT = "artifacts/phase0_representation_sweep"
CANON_EVAL_ROOT = "artifacts/evaluation/phase1_phase2_seedavg_model_winners_canonical_eval"
PHASE0_SELECTION = ".scratch/demap/phase0_representation_sweep/phase0_selection.json"
PHASE2_SEEDAVG = ".scratch/demap/phase1_phase2_seedavg/PHASE2_seedavg_winners.csv"
CE_TREE = "artifacts/final_reranker/crossencoder_fulltrain"

_FORBIDDEN = "artifacts_v3_cdisc"


class MissingArtifact(RuntimeError):
    pass


def _p(rel: str) -> Path:
    return REPO_ROOT / rel


def _require(rel: str) -> Path:
    p = _p(rel)
    if _FORBIDDEN in p.parts:
        raise MissingArtifact(f"refusing to read from {_FORBIDDEN}: {rel}")
    if not p.exists():
        raise MissingArtifact(f"required source artifact missing: {rel} -> {p}")
    return p


def _round(x, n=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return round(float(x), n)


# ---------------------------------------------------------------------------
def load_phase0_cells(slug: str) -> Dict[Tuple[str, str], dict]:
    root = _require(f"{PHASE0_ROOT}/{slug}")
    cells: Dict[Tuple[str, str], dict] = {}
    for mj in glob.glob(str(root / "*/*/metrics.json")):
        d = json.load(open(mj))
        sp = d.get("spec", {})
        qv, rec = sp.get("query_variant"), sp.get("recipe")
        if qv and rec:
            cells[(qv, rec)] = d.get("metrics_by_split", {})
    if not cells:
        raise MissingArtifact(f"no phase0 metrics.json cells found for {slug} under {root}")
    return cells


def write_F_and_C1_and_E(be_dir: Path, pr_dir: Path) -> List[str]:
    written: List[str] = []
    phase0 = {slug: load_phase0_cells(slug) for slug in MODELS}

    # F: 4x10 grids per model, per metric
    for slug in MODELS:
        cells = phase0[slug]
        for tag, (split, met) in {
            "val_dev_recallat5": ("val_dev", "recall@5"),
            "test_recallat5": ("test", "recall@5"),
        }.items():
            out = be_dir / f"offtheshelf_reppair_{slug}__{tag}.csv"
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["query_variant \\ cde_recipe"] + CDE_RECIPES)
                for qv in QUERY_VARIANTS:
                    row = [qv]
                    for rec in CDE_RECIPES:
                        m = cells.get((qv, rec), {}).get(split, {})
                        row.append(_round(m.get(met)))
                    w.writerow(row)
            written.append(str(out))

    # C1: off-the-shelf best cell per model (by val_dev recall@5)
    c1 = be_dir / "biencoder_offtheshelf_best_by_model.csv"
    with open(c1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "best_query_variant", "best_cde_recipe", "selection_metric",
                    "split", "recall@1", "recall@5", "recall@10", "recall@100", "mrr@100", "n"])
        for slug, hf in MODELS.items():
            cells = phase0[slug]
            best = max(cells, key=lambda k: cells[k].get("val_dev", {}).get("recall@5", -1))
            for split in ["val_dev", "test"]:
                m = cells[best].get(split, {})
                w.writerow([hf, best[0], best[1], "val_dev recall@5", split,
                            _round(m.get("recall@1")), _round(m.get("recall@5")),
                            _round(m.get("recall@10")), _round(m.get("recall@100")),
                            _round(m.get("mrr@100")), int(m.get("n", 0) or 0)])
    written.append(str(c1))

    # E: representation sweep best-per-model
    sel = json.load(open(_require(PHASE0_SELECTION)))
    e = pr_dir / "phase0_representation_sweep_best_by_model.csv"
    with open(e, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "selection_metric", "best_query_variant", "best_cde_recipe",
                    "best_val_dev_recall@5", "best_test_recall@5",
                    "anchor_query_variant", "anchor_cde_recipe", "anchor_val_dev_recall@5",
                    "anchor_is_alias_of_best"])
        for slug, hf in MODELS.items():
            if slug not in sel:
                raise MissingArtifact(f"phase0_selection.json missing model {slug}")
            s = sel[slug]
            best, anchor = s["top"][0], s["anchor"]
            arow = next((t for t in s["top"] if t["query_variant"] == anchor["query_variant"]
                         and t["recipe"] == anchor["recipe"]), {})
            w.writerow([hf, "val_dev recall@5", best["query_variant"], best["recipe"],
                        _round(best["val_dev_recall@5"]), _round(best.get("test_recall@5")),
                        anchor["query_variant"], anchor["recipe"], _round(anchor.get("score")),
                        arow.get("alias_equivalent_to_anchor", "")])
    written.append(str(e))
    return written


def write_C2_C3(be_dir: Path) -> List[str]:
    written: List[str] = []
    _require(CANON_EVAL_ROOT)
    rep_seed: Dict[str, int] = {}
    with open(_require(PHASE2_SEEDAVG)) as f:
        for row in csv.DictReader(f):
            rep_seed[row["base_model_id"]] = int(row["rep_seed"])

    def write_phase(phase: str, out_name: str, seed_of):
        out = be_dir / out_name
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "phase", "seed", "dataset", "recall@1", "recall@5",
                        "recall@10", "recall@100", "mrr@100", "n"])
            for slug, hf in MODELS.items():
                seed = seed_of(slug)
                pat = (f"{phase}__{slug}__*__seed{seed}" if phase == "phase2"
                       else f"{phase}__{slug}__seed{seed}")
                dirs = sorted(glob.glob(str(_p(CANON_EVAL_ROOT) / pat)))
                if not dirs:
                    raise MissingArtifact(f"no canonical eval dir matching {pat}")
                d = json.load(open(Path(dirs[0]) / "metrics.json"))
                mbs = d.get("metrics_by_split", {})
                for ds in CANON:
                    m = mbs.get(ds, {})
                    w.writerow([hf, phase, seed, ds, _round(m.get("recall@1")),
                                _round(m.get("recall@5")), _round(m.get("recall@10")),
                                _round(m.get("recall@100")), _round(m.get("mrr@100")),
                                int(m.get("n", 0) or 0)])
        return str(out)

    written.append(write_phase("phase1", "biencoder_phase1_best_by_model.csv", lambda slug: 0))
    written.append(write_phase("phase2", "biencoder_phase2_best_by_model.csv",
                               lambda slug: rep_seed[slug]))
    return written


def write_raw_ce(ce_dir: Path) -> List[str]:
    """RAW / off-the-shelf CE (val_dev only). Documents the canonical-score gap; does
    NOT run any new eval."""
    models_rel = {m: f"{CE_TREE}/runs/{m}/training_summary.json" for m in CE_MODELS}
    e0 = {}
    for m, rel in models_rel.items():
        d = json.load(open(_require(rel)))
        row = [h for h in d.get("history", []) if h.get("epoch") == 0]
        if not row:
            raise MissingArtifact(f"no epoch-0 (raw) entry in {rel}")
        e0[m] = row[0]
    out = ce_dir / "ce_raw_offtheshelf_val_dev.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "split", "recall@1", "recall@5", "recall@10", "recall@100",
                    "mrr@100", "n_queries", "note"])
        for m in CE_MODELS:
            r = e0[m]
            w.writerow([CE_PRETTY[m], "val_dev", _round(r["recall@1"]), _round(r["recall@5"]),
                        _round(r["recall@10"]), "", _round(r["mrr@100"]), r.get("n_queries", ""),
                        "RAW backbone (epoch-0 val_dev se20 U kwfuzzy10 pool); "
                        "NO canonical-set raw CE eval exists — documented gap, not filled"])
    return [str(out)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root",
                    default="artifacts/publication_artifact_package_20260713/tables",
                    help="tables/ root of the publication package")
    ap.add_argument("--check", action="store_true",
                    help="verify source artifacts resolve and exit (writes nothing)")
    args = ap.parse_args(argv)

    if "/tmp" in str(args.out_root).split("/")[:2] or str(args.out_root).startswith("/tmp"):
        ap.error("refusing to write under /tmp (Biowulf scratch rule)")

    out_root = _p(args.out_root) if not Path(args.out_root).is_absolute() else Path(args.out_root)
    be_dir = out_root / "bi_encoder"
    pr_dir = out_root / "pretraining_representation"
    ce_dir = out_root / "cross_encoder"

    # Resolve every source up front; fail clearly before writing anything.
    for slug in MODELS:
        _require(f"{PHASE0_ROOT}/{slug}")
    _require(PHASE0_SELECTION)
    _require(PHASE2_SEEDAVG)
    _require(CANON_EVAL_ROOT)
    for m in CE_MODELS:
        _require(f"{CE_TREE}/runs/{m}/training_summary.json")

    if args.check:
        print("OK: all source artifacts resolve; 4 bi-encoders:", list(MODELS.values()))
        return 0

    for d in (be_dir, pr_dir, ce_dir):
        d.mkdir(parents=True, exist_ok=True)

    written = []
    written += write_F_and_C1_and_E(be_dir, pr_dir)
    written += write_C2_C3(be_dir)
    written += write_raw_ce(ce_dir)

    print(f"wrote {len(written)} tables under {out_root} (4 bi-encoders preserved):")
    for w in sorted(written):
        print("  ", Path(w).relative_to(out_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
