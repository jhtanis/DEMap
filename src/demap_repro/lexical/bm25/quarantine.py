"""Quarantine guard for the BM25 baseline.

The BM25 code is sound; its historical *inputs* are not. Every saved BM25 artifact under
``artifacts/paper_bm25_*`` was produced against the January benchmark-construction catalog
(79,479 rows, retired CDEs included) and against split files that no longer denote the canonical
evaluation datasets. Those numbers are NOT valid for the paper.

This module blocks the unsafe paths by default while preserving every historical artifact and
every line of historical code. Legacy execution remains possible for reproduction, but only with
an explicit ``--allow-legacy`` and only after a prominent warning.

Canonical paper BM25 route:

    PYTHONPATH=src .venv/bin/python -m demap paper bm25 --run
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

CANONICAL_RUNNER = "demap paper bm25"
CANONICAL_COMMAND = "PYTHONPATH=src .venv/bin/python -m demap paper bm25 --run"
CANONICAL_PROTOCOL = "configs/paper/bm25_protocol_v1.yaml"
CANONICAL_WRAPPER = "scripts/paper_bm25_canonical.py"  # thin wrapper; not the documented route

# The June production catalog: the ONLY catalog valid for paper BM25 runs.
CANONICAL_CATALOG = (
    "data/processed/cadsr_xml_2026-06-18/"
    "cde_master_enriched_eval_production_cde_match.parquet"
)
CANONICAL_CATALOG_SHA256 = "2e360368307c7589325a43eebcbd6f6ec22090530ccf3521862cc1b9cb62a200"
CANONICAL_CATALOG_ROWS = 62976

# The January benchmark-construction catalog. Legitimate upstream source of queries, gold links
# and splits -- never a retrieval catalog for reported results.
LEGACY_CATALOG = "data/processed/cde_master_enriched.parquet"
LEGACY_CATALOG_ROWS = 79479

BM25_INVENTORY = {
    # ---------------------------------------------------------------- canonical
    CANONICAL_RUNNER: {
        "kind": "cli", "status": "canonical",
        "note": "fail-closed canonical BM25: June catalog, val_dev selection, canonical eval sets",
    },
    # ------------------------------------------------- reusable internal components
    "src/demap/experiments/bm25_baseline.py::build_bm25_model": {
        "kind": "lib", "status": "reusable_component",
        "note": "Okapi BM25 index builder; used unchanged by the canonical runner",
    },
    "src/demap/experiments/bm25_baseline.py::tokenize_baseline": {
        "kind": "lib", "status": "reusable_component",
    },
    "src/demap/experiments/bm25_baseline.py::_rank_order": {
        "kind": "lib", "status": "reusable_component",
        "note": "deterministic tie-break: -score, then catalog row index",
    },
    "src/demap/text/recipes.py::build_catalog": {"kind": "lib", "status": "reusable_component"},
    "scripts/keyword_stage_a_probe.py": {
        "kind": "diagnostic", "status": "reusable_component",
        "note": "imports build_bm25_model; not a paper-reporting path",
    },
    # ------------------------------------------- historical reproduction only
    "demap bm25-baseline": {
        "kind": "cli", "status": "historical_reproduction",
        "note": "pair-level, version-exact metrics; requires --allow-legacy for a legacy catalog",
    },
    "src/demap/pipeline/paper_baselines.py::bm25-heatmap-run": {
        "kind": "stage", "status": "historical_reproduction",
    },
    "src/demap/pipeline/paper_baselines.py::bm25-final-report-run": {
        "kind": "stage", "status": "historical_reproduction",
    },
    "src/demap/paper_bm25_gdc_alt_diagnostic.py": {
        "kind": "diagnostic", "status": "historical_reproduction", "note": "never executed",
    },
    "configs/paper/bm25_protocol_v1.yaml": {
        "kind": "protocol", "status": "canonical",
        "note": "locks catalog+sha, 4x10 grid, val_dev-only selection, k1/b, metrics, depth",
    },
    "notebooks/dataset_publication_tables_figures.ipynb::inline_bm25": {
        "kind": "diagnostic", "status": "historical_reproduction",
        "note": "third inline BM25 (k1=1.2, b=0.75); near-tie confusability tables only. Cannot be code-guarded; must never be promoted to a reporting path.",
    },
    "scripts/alt_lexical_diagnostics.py": {
        "kind": "diagnostic", "status": "historical_reproduction",
        "note": "independent BM25 re-implementation; diagnostic only. Routed through require_catalog(): a legacy catalog needs --allow-legacy.",
    },
    # ------------------------------------------------- conflicting / unsafe
    "configs/experiments/paper_bm25_heatmap.yaml": {
        "kind": "config", "status": "conflicting_unsafe",
        "note": "January 79,479 catalog + legacy split scheme",
    },
    "configs/experiments/paper_bm25_final_report.yaml": {
        "kind": "config", "status": "conflicting_unsafe",
        "note": "January 79,479 catalog + non-canonical eval splits incl. the leakage-excluded set",
    },
    "scripts/paper_select_bm25_winner.py": {
        "kind": "selector", "status": "conflicting_unsafe",
        "note": "rewrites the report config in place; selected over 16 of 40 cells on the legacy val split",
    },
    "src/demap/pipeline/paper_baselines.py::bm25-select-winner": {
        "kind": "stage", "status": "conflicting_unsafe",
    },
    "artifacts/paper_bm25_heatmap": {
        "kind": "artifact_root", "status": "conflicting_unsafe",
        "note": "40 runs on the January catalog; PRESERVE, never report",
    },
    "artifacts/paper_bm25_final_report": {
        "kind": "artifact_root", "status": "conflicting_unsafe",
        "note": "test n=6405 on the January catalog; PRESERVE, never report",
    },
    "artifacts/summaries/paper_bm25_winner": {
        "kind": "artifact_root", "status": "conflicting_unsafe",
        "note": "partial-grid selection on the legacy val split; PRESERVE, never report",
    },
}


class Bm25LegacyBlocked(RuntimeError):
    """Raised when an unsafe legacy BM25 path is invoked without explicit authorization."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _norm(path: str | os.PathLike) -> str:
    return str(path).replace("\\", "/").lstrip("./")


def warn_legacy(what: str) -> None:
    """Print a prominent, unmissable warning that legacy BM25 output is not paper-valid."""
    bar = "*" * 78
    print(
        f"\n{bar}\n"
        f"*** HISTORICAL-ONLY BM25 PATH: {what}\n"
        f"***\n"
        f"*** Results from this path are NOT VALID FOR THE PAPER. They are produced against\n"
        f"*** the January benchmark-construction catalog ({LEGACY_CATALOG_ROWS:,} rows, retired CDEs\n"
        f"*** included) and/or non-canonical evaluation splits, and are scored with pair-level,\n"
        f"*** version-exact metrics rather than the query-level, public-id metrics used for\n"
        f"*** every reported result.\n"
        f"***\n"
        f"*** Canonical BM25 for the paper:\n"
        f"***     {CANONICAL_COMMAND}\n"
        f"{bar}\n",
        flush=True,
    )


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_legacy_catalog(path: str | os.PathLike) -> bool:
    return _norm(path).endswith(_norm(LEGACY_CATALOG))


def is_canonical_catalog(path: str | os.PathLike) -> bool:
    return _norm(path).endswith(_norm(CANONICAL_CATALOG))


def require_catalog(catalog: Optional[str], *, allow_legacy: bool = False,
                    context: str = "bm25-baseline") -> str:
    """Resolve and validate the BM25 retrieval catalog.

    Fails closed. There is deliberately NO default: a missing catalog is an error rather than a
    silent fall back to the January master (which is what the historical code did).
    """
    if not catalog:
        raise Bm25LegacyBlocked(
            f"{context}: no CDE catalog configured.\n"
            f"There is no default. Pass --cde-master-enriched or set "
            f"`bm25_baseline.cde_master_enriched` in a config.\n"
            f"For paper runs use the canonical runner instead:\n"
            f"    {CANONICAL_COMMAND}"
        )
    if is_legacy_catalog(catalog) and not allow_legacy:
        raise Bm25LegacyBlocked(
            f"{context}: refusing the January benchmark-construction catalog\n"
            f"    {catalog}\n"
            f"It contains retired CDEs and is not the retrieval catalog used by any reported\n"
            f"result. Paper BM25 must retrieve from\n"
            f"    {CANONICAL_CATALOG}\n"
            f"Use the canonical runner:\n"
            f"    {CANONICAL_COMMAND}\n"
            f"To reproduce the historical numbers anyway, re-run with --allow-legacy."
        )
    if is_legacy_catalog(catalog) and allow_legacy:
        warn_legacy(f"{context} on {catalog}")
    return str(catalog)


def assert_canonical_catalog(catalog: str | os.PathLike, *, verify_sha256: bool = True) -> None:
    """Hard requirement for paper BM25: the June catalog, by path and (optionally) content."""
    if not is_canonical_catalog(catalog):
        raise Bm25LegacyBlocked(
            f"paper BM25 requires the canonical catalog {CANONICAL_CATALOG}, got {catalog}")
    p = Path(catalog)
    if not p.is_absolute():
        p = _repo_root() / p
    if not p.exists():
        raise Bm25LegacyBlocked(f"canonical catalog not found: {p}")
    if verify_sha256:
        got = sha256_file(p)
        if got != CANONICAL_CATALOG_SHA256:
            raise Bm25LegacyBlocked(
                f"canonical catalog sha256 mismatch: {got} != {CANONICAL_CATALOG_SHA256}")


def block_unsafe(name: str, *, allow_legacy: bool = False) -> None:
    """Gate a `conflicting_unsafe` or `historical_reproduction` entry point."""
    entry = BM25_INVENTORY.get(name, {})
    status = entry.get("status", "conflicting_unsafe")
    if status in {"canonical", "reusable_component"}:
        return
    if not allow_legacy:
        raise Bm25LegacyBlocked(
            f"{name} is classified '{status}' and is blocked by default.\n"
            f"{entry.get('note', '')}\n"
            f"Its output is NOT valid for the paper. Canonical BM25:\n"
            f"    {CANONICAL_COMMAND}\n"
            f"Re-run with --allow-legacy to reproduce the historical behaviour."
        )
    warn_legacy(name)
