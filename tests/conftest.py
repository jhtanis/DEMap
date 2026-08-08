"""Shared fixtures and artifact discovery.

Most tests here run offline against small committed fixtures. A few can do more
when the frozen experiment artifacts happen to be present — point
``DEMAP_ARTIFACT_ROOT`` at the research repository to enable them; they skip
otherwise, so the suite is green on a clean clone.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"


def load_fixture(name: str):
    """Load a committed JSON fixture by file name."""
    return json.loads((FIXTURE_DIR / name).read_text())


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def artifact_root():
    """Root of the frozen experiment artifacts, or skip.

    These are large and are not distributed with the repository.
    """
    raw = os.environ.get("DEMAP_ARTIFACT_ROOT")
    if not raw:
        pytest.skip("DEMAP_ARTIFACT_ROOT is not set; frozen artifacts unavailable")
    root = Path(raw)
    if not root.exists():
        pytest.skip(f"DEMAP_ARTIFACT_ROOT={root} does not exist")
    return root


@pytest.fixture(scope="session")
def fixed_k_summary():
    """Committed summary of the authoritative fixed-K feature table."""
    return load_fixture("fixed_k_feature_table.json")


@pytest.fixture(scope="session")
def source_parity():
    """Committed AST hashes pinning migrated implementations."""
    return load_fixture("source_parity.json")
