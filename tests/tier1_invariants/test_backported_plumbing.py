"""Plumbing defects found by the later metamodel line, backported here.

Every test below fails against this repository as it stood at 55e8765. None of
them is about a scientific result: they are the difference between code that
runs from a clean clone and code that only ever ran on one machine, with the
paths and imports of a research tree baked in.

The audit that decided which of those defects apply to the non-metamodel paper,
and which do not, is recorded per file in ``manifests/source_migration.yaml``.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import pathlib

import pytest

pytestmark = pytest.mark.tier1

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


# --------------------------------------------------------------------------
# Bug 1 — the curator allowlists were never migrated
# --------------------------------------------------------------------------

def test_curator_allowlists_are_tracked_inputs():
    """``demap build-queries`` resolves three allowlists that were not shipped.

    ``manifests/dataset_build_manifest_paper_era.json`` passes two of them on the
    executed build's command line, so their absence made the benchmark build
    unreproducible from a clean clone.
    """
    from demap_repro.data.queries import ALLOWLIST_SHA256

    missing, wrong = [], []
    for name, expected in sorted(ALLOWLIST_SHA256.items()):
        path = ROOT / "configs" / "allowlists" / name
        if not path.exists():
            missing.append(name)
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            wrong.append(f"{name}: {digest} != {expected}")
    assert not missing, f"curator allowlists absent from the repository: {missing}"
    assert not wrong, f"curator allowlists edited without updating the pin: {wrong}"


def test_allowlist_defaults_resolve_to_the_shipped_files():
    """The module defaults must name files that exist, not files that should."""
    from demap_repro.data import queries

    for constant in ("DEFAULT_ALT_ALLOWLIST", "DEFAULT_ALT_RECIPE_ALLOWLIST",
                     "DEFAULT_REFDOC_ALLOWLIST"):
        path = pathlib.Path(getattr(queries, constant))
        assert path.exists(), f"{constant} -> {path} does not exist"


# --------------------------------------------------------------------------
# Bug 2 — the canonical registry default named a path that does not exist
# --------------------------------------------------------------------------

def test_canonical_registry_loads_with_no_arguments():
    """Formerly ``configs/evaluation/canonical_eval_datasets.yaml``, CWD-relative:
    every no-argument call raised ``FileNotFoundError``."""
    from demap_repro.evaluation import canonical_datasets as cd

    assert pathlib.Path(cd.REGISTRY_PATH).is_absolute()
    assert cd.canonical_split_names() == list(cd.PAPER_EVAL_DATASETS)


# --------------------------------------------------------------------------
# Bug 3 / 13 — data roots resolved inside the installed package
# --------------------------------------------------------------------------

#: ``Path(__file__).resolve().parents[1]`` meant "repository root" while these
#: modules lived in ``scripts/``. After packaging it resolves to
#: ``src/demap_repro`` (or ``src/demap_repro/data``), so every default path
#: underneath pointed into the source tree.
_DATA_ROOT_MODULES = [
    ("demap_repro.biencoder.deep_retrieval", "REPO_ROOT"),
    ("demap_repro.reporting.biencoder_tables", "REPO_ROOT"),
    ("demap_repro.data.cimac.corrected_pv_split", "REPO"),
    ("demap_repro.data.cimac.appendix_a_v2", "REPO"),
]


@pytest.mark.parametrize("module_name,attribute", _DATA_ROOT_MODULES,
                         ids=[m for m, _ in _DATA_ROOT_MODULES])
def test_no_module_resolves_its_data_root_inside_the_package(module_name, attribute):
    module = importlib.import_module(module_name)
    root = pathlib.Path(getattr(module, attribute)).resolve()
    assert root != SRC and SRC not in root.parents, (
        f"{module_name}.{attribute} resolves to {root}, inside the source tree")


def test_deep_retrieval_defaults_land_outside_the_source_tree():
    from demap_repro.biencoder import deep_retrieval

    for name in ("SPLITS_DIR", "CDE_MASTER", "OUT_DIR", "DEFAULT_RUN_DIR"):
        path = pathlib.Path(getattr(deep_retrieval, name)).resolve()
        assert SRC not in path.parents, f"{name} -> {path} points into src/"


def test_splits_dir_names_the_corrected_canonical_tree():
    """``splits_v3_cdisc`` is the superseded scheme (test 3,986 against the
    paper's 3,959) and had never resolved anywhere real."""
    from demap_repro.biencoder import deep_retrieval

    assert deep_retrieval.SPLITS_DIR.name == "splits_v3_cdisc_reachable_2026-06-18_cimacpv"


# --------------------------------------------------------------------------
# Bug 11 — absolute research-repository paths, and the sys.path inserts
# --------------------------------------------------------------------------

def _tracked_sources():
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_module_hardcodes_an_absolute_machine_path():
    """A released repository cannot carry ``/vf/users/...`` or ``/data/...``
    prefixes into the research tree: they make the module unusable anywhere else
    and silently re-couple this repository to that one.

    ``demap_repro.utils.paths.data_root`` and ``DEMAP_DATA_ROOT`` are the seam.
    """
    offenders = []
    for path in _tracked_sources():
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue          # a comment recording the old value is fine
            for prefix in ('"/vf/users/', "'/vf/users/", '"/data/nextgen2/', "'/data/nextgen2/"):
                if prefix in line:
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno}")
                    break
    assert not offenders, f"absolute machine paths in tracked source: {offenders}"


def test_no_module_manipulates_sys_path():
    """Every one of these inserted a directory *ahead of* the installed package,
    so an unrelated checkout on that path could shadow ``demap_repro`` itself.
    The imports beneath them all resolve from the installed distribution."""
    offenders = []
    for path in _tracked_sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (isinstance(func, ast.Attribute) and func.attr in {"insert", "append"}
                    and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "path"
                    and isinstance(func.value.value, ast.Name)
                    and func.value.value.id == "sys"):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, f"sys.path manipulation in tracked source: {offenders}"


def test_no_module_imports_an_unmigrated_research_script():
    """Three modules still imported research-repository scripts by bare name.

    Each has an equivalent here — ``paper_figure_style`` ->
    ``reporting.figures.style``, ``train_hgbc_reranker`` -> ``reranker.train``,
    ``evaluate_non_exact_subset`` -> ``lexical.non_exact_subset`` — so from a
    clean clone the originals were simply ``ModuleNotFoundError``.

    ``eval_hgbc_cimac131`` has no equivalent and is deliberately exempt: it is
    guarded at its call site instead (see the bug 4 tests).
    """
    unmigrated = {"paper_figure_style", "train_hgbc_reranker", "evaluate_non_exact_subset"}
    offenders = []
    for path in _tracked_sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = {node.module.split(".")[0]}
            else:
                continue
            for name in names & unmigrated:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    assert not offenders, f"imports of unmigrated research scripts: {offenders}"
