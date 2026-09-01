"""The keyword arm ships here, and it is the implementation that ran.

Both CDE Match implementations were cleared for public release on 2026-09-01.
Until then they were reached through a runtime adapter
(``lexical/cde_match_interface.py``) that resolved them from whatever
``demap.features`` happened to be importable. That adapter is gone, and these
tests hold the migration to the standard the rest of the repository uses:

* the modules import and expose the symbols the pipeline calls;
* nothing reaches back into the research package or edits ``sys.path``;
* the migrated bytes are the bytes that ran, checked by AST equivalence against
  the recorded source digest rather than by eye;
* the gate is retired in the manifests, and the material that is still
  restricted has not moved.
"""
from __future__ import annotations

import ast
import pathlib

import pytest
import yaml

pytestmark = pytest.mark.tier2

ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG = ROOT / "src" / "demap_repro" / "lexical" / "cde_match"
MODULES = ("clone.py", "keyword_retriever.py", "build_candidates.py")


# --------------------------------------------------------------------------
# the code is here and callable
# --------------------------------------------------------------------------

def test_the_three_modules_are_present():
    for name in MODULES:
        assert (PKG / name).is_file(), f"{name} missing from {PKG}"


def test_clone_exposes_the_symbols_the_pipeline_calls():
    from demap_repro.lexical.cde_match import clone

    for symbol in ("ExactMatchControl", "FuzzyFallback"):
        assert hasattr(clone, symbol), f"clone.{symbol} missing"


def test_keyword_retriever_exposes_generate_candidates():
    from demap_repro.lexical.cde_match import keyword_retriever

    assert callable(keyword_retriever.generate_candidates)


def test_exact_match_control_is_constructible():
    """non_exact_subset builds this directly now, not through an adapter."""
    from demap_repro.lexical.cde_match.clone import ExactMatchControl

    assert ExactMatchControl(allow_rate=0.70, seed=42) is not None


def test_base_features_reaches_the_retriever_without_the_adapter():
    """keyword_provenance imports the retriever lazily, as the source did.

    The import is deferred inside the function because it pulls in sklearn, so
    it is asserted on the source text rather than as a module attribute.
    """
    src = (ROOT / "src" / "demap_repro" / "reranker" / "base_features.py").read_text()
    assert "from demap_repro.lexical.cde_match import keyword_retriever as kr" in src
    assert "kr.generate_candidates(" in src
    assert "cde_match_interface" not in src


# --------------------------------------------------------------------------
# nothing reaches back into the research package
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", MODULES)
def test_no_import_of_the_research_package(name):
    tree = ast.parse((PKG / name).read_text())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "demap" or node.module.startswith("demap."):
                bad.append(node.module)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "demap" or a.name.startswith("demap."):
                    bad.append(a.name)
    assert not bad, f"{name} imports the research package: {bad}"


@pytest.mark.parametrize("name", MODULES)
def test_no_sys_path_manipulation(name):
    """Both files inserted into sys.path to find the research tree."""
    src = (PKG / name).read_text()
    assert "sys.path.insert" not in src, f"{name} still edits sys.path"
    assert "sys.path.append" not in src, f"{name} still edits sys.path"


def test_the_adapter_is_gone():
    assert not (ROOT / "src" / "demap_repro" / "lexical" / "cde_match_interface.py").exists()


def test_nothing_still_refers_to_the_adapter():
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        if "cde_match_interface" in path.read_text():
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"still reference the deleted adapter: {offenders}"


# --------------------------------------------------------------------------
# the migrated code is the code that ran
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ledger():
    return yaml.safe_load((ROOT / "manifests" / "source_migration.yaml").read_text())


@pytest.mark.parametrize("name", MODULES)
def test_migration_is_recorded_with_its_source_digest(ledger, name):
    dst = f"src/demap_repro/lexical/cde_match/{name}"
    entry = next((e for e in ledger["entries"] if e.get("dst") == dst), None)
    assert entry is not None, f"{dst} has no migration ledger entry"
    assert entry["migration_state"] == "parity_established"
    assert entry.get("release_gate") is None
    assert len(entry["worktree_sha256"]) == 64


@pytest.mark.parametrize("name", MODULES)
def test_only_imports_changed_from_the_source(ledger, name):
    """With imports and docstrings stripped, the AST must be identical.

    The copy-first rule: the bytes that ran were copied, then only import
    statements were repointed and the docstrings that named the old module
    paths corrected. Rather than allow-listing the functions that carry a lazy
    import - which would quietly excuse a real edit in the same function - this
    removes all Import/ImportFrom nodes and all docstrings from both trees and
    demands the remainder match exactly. Anything left is a logic change.

    Docstrings are covered separately by
    ``test_no_docstring_points_at_the_research_package``.

    Skips when the research checkout is unavailable.
    """
    dst = f"src/demap_repro/lexical/cde_match/{name}"
    entry = next(e for e in ledger["entries"] if e.get("dst") == dst)
    src_repo = pathlib.Path(ledger["source_repository"]["path"])
    src_file = src_repo / entry["src"]
    if not src_file.is_file():
        pytest.skip(f"research checkout unavailable: {src_file}")

    class Strip(ast.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

        def _drop_docstring(self, node):
            self.generic_visit(node)
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
            return node

        visit_FunctionDef = _drop_docstring
        visit_AsyncFunctionDef = _drop_docstring
        visit_ClassDef = _drop_docstring
        visit_Module = _drop_docstring

    def normalized(source: str) -> dict:
        tree = Strip().visit(ast.parse(source))
        ast.fix_missing_locations(tree)
        return {n.name: ast.dump(n) for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}

    ours = normalized((PKG / name).read_text())
    theirs = normalized(src_file.read_text())

    assert set(ours) == set(theirs), (
        f"{name}: definitions differ\n"
        f"  only here:   {sorted(set(ours) - set(theirs))}\n"
        f"  only source: {sorted(set(theirs) - set(ours))}")

    differing = sorted(k for k in ours if ours[k] != theirs[k])
    assert not differing, (
        f"{name}: these definitions changed beyond their imports: {differing}")


@pytest.mark.parametrize("name", MODULES)
def test_no_docstring_points_at_the_research_package(name):
    """Prose must not send a reader to a module path that does not exist here."""
    import re

    src = (PKG / name).read_text()
    stale = re.findall(r"\bdemap\.(?!features\b)[a-z_]+(?:\.[a-z_]+)*", src)
    stale += re.findall(r"\bdemap\.features\.[a-z_]+", src)
    stale += re.findall(r"src/demap/[a-z_/]+\.py", src)
    assert not stale, f"{name} still points at the research package: {sorted(set(stale))}"


# --------------------------------------------------------------------------
# what is still restricted has not moved
# --------------------------------------------------------------------------

def test_the_derivative_gate_is_retired(ledger):
    gate = ledger["release_gates"]["cde_match_derivative_unresolved"]
    assert gate["status"] == "resolved"
    assert not gate["affects_stages"]


def test_no_stage_is_gated_on_the_retired_gate():
    from demap_repro.cli import STAGES

    assert not [s.name for s in STAGES if s.gated]


def test_the_nci_supplied_material_is_still_excluded(ledger):
    """The clearance covered our implementations, not NCI's source material."""
    assert ledger["release_gates"]["nci_cde_match_source"]["status"] == "unresolved"
    still = {e["src"] for e in ledger["gated_material"]
             if e["release_gate"] == "nci_cde_match_source"}
    assert len(still) == 7, still
    assert all(e["migration_state"] == "not_copied" for e in ledger["gated_material"]
               if e["release_gate"] == "nci_cde_match_source")


def test_the_live_service_output_is_still_excluded(ledger):
    assert ledger["release_gates"]["official_nci_cde_match_frozen"]["status"] == "unresolved"


def test_cimac_is_still_gated(ledger):
    """The keyword-algorithm clearance did not extend to the CIMAC workbook."""
    assert ledger["release_gates"]["cimac_unresolved"]["status"] == "unresolved"
    assert any(e["release_gate"] == "cimac_unresolved" for e in ledger["gated_material"])


def test_no_nci_source_material_is_in_the_repository():
    """The PL/SQL and the logic PDF have never been here; keep it that way."""
    for pattern in ("*.sql", "CDE_Match_Logic*.pdf", "S74_NCI_DS*"):
        found = [p for p in ROOT.rglob(pattern)
                 if ".git" not in p.parts and ".venv" not in p.parts
                 and ".scratch" not in p.parts]
        assert not found, f"NCI-supplied material present: {found}"
