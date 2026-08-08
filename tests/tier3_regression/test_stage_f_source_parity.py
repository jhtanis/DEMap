"""Migrated Stage-F logic still matches the implementation that produced the paper.

Every function listed in ``tests/fixtures/source_parity.json`` was copied from the
research repository and then refactored in place — renamed, unindented from a
script into a module, and in a few cases pointed at an adapter instead of a
directly-imported dependency. The fixture records the hash of each function's
abstract syntax tree at the moment the migration was verified against source.

A hash change means the code now does something different. That may be intended,
but it cannot happen silently: regenerate the fixture from the research
repository and re-verify, don't edit the expected hash.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from demap_repro.utils.source_parity import function_ast_hash

pytestmark = [pytest.mark.tier3, pytest.mark.parity]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _component_cases(parity):
    for component, spec in parity["components"].items():
        for func, entry in spec["functions"].items():
            yield component, spec["dst_module"], func, entry


def test_fixture_covers_stage_f(source_parity):
    comp = source_parity["components"]["stage_f_base_features"]
    assert comp["dst_module"] == "src/demap_repro/reranker/base_features.py"
    assert set(comp["functions"]) == {
        "load_biencoder_long",
        "keyword_provenance",
        "apply_keyword_exact_control",
        "compute_all_features",
        "audit_feature_table",
        "feature_summary",
    }


def test_migrated_functions_match_pinned_ast(source_parity):
    """The whole point: refactoring must not have changed behaviour."""
    failures = []
    for component, module, func, entry in _component_cases(source_parity):
        source = (REPO_ROOT / module).read_text(encoding="utf-8")
        actual = function_ast_hash(source, func)
        if actual != entry["ast_sha256"]:
            failures.append(
                f"{component}:{module}::{func}\n"
                f"    expected {entry['ast_sha256']}\n"
                f"    actual   {actual}\n"
                f"    migrated from {entry['migrated_from']['path']}"
                f"::{entry['migrated_from']['function']}"
            )
    assert not failures, "migrated logic changed:\n" + "\n".join(failures)


def test_seams_are_declared_not_assumed(source_parity):
    """A function that differs from its source must carry a written reason.

    Five of the six Stage-F functions are byte-identical in AST to the research
    implementation. The sixth differs only in the text of a diagnostic message,
    and says so.
    """
    for component, module, func, entry in _component_cases(source_parity):
        if entry.get("identical_to_source") is False:
            assert entry.get("reviewed_seam"), (
                f"{component}::{func} differs from source with no reviewed_seam note"
            )
            assert entry.get("source_normalized_ast_sha256")


def test_every_pinned_function_still_exists(source_parity):
    for component, module, func, _entry in _component_cases(source_parity):
        path = REPO_ROOT / module
        assert path.exists(), f"{module} is missing (referenced by {component})"
        function_ast_hash(path.read_text(encoding="utf-8"), func)
