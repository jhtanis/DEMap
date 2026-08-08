"""Importing a module must never run an experiment.

Much of this code was migrated from research scripts whose work happened at module
level — reading artifacts, fitting models, writing figures. That is fine for a
script invoked once; it is not fine for an importable package. A test collector, an
IDE, or ``demap_repro.cli`` listing the stages would silently overwrite results.

During migration this bit twice: importing the modules to check them regenerated
figures and validation reports in the source repository. Nothing scientific
drifted, but the failure mode is real, so it is now a test.

Every module must be importable with no side effects, and everything that does
work must live behind ``main()`` or another callable.
"""
from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

pytestmark = pytest.mark.tier1

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"

#: Operations that must not happen while a module is being imported. Building a
#: constant lookup table or setting matplotlib rcParams is fine; touching the
#: filesystem, loading an artifact or fitting a model is not.
FORBIDDEN_CALLS = {
    # filesystem
    "open", "mkdir", "makedirs", "write_text", "write_bytes", "unlink",
    "rmtree", "copy", "copytree", "rename", "remove", "touch",
    # data i/o
    "read_csv", "read_parquet", "read_json", "read_excel", "load", "loads_file",
    "to_csv", "to_parquet", "to_json", "savefig", "save", "dump",
    # compute
    "fit", "predict", "predict_proba", "encode", "train", "main", "run",
    # process
    "system", "check_call", "check_output", "Popen", "exit",
}


def _modules():
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        yield path


def _module_name(path: pathlib.Path) -> str:
    return str(path.relative_to(SRC)).replace("/", ".")[:-3]


def _toplevel_work(path: pathlib.Path):
    """Forbidden operations reachable while the module body executes.

    Statement *shape* is not the criterion — a module-level loop that fills a
    constant dict is harmless. What matters is whether importing the module can
    read an artifact, write a file, or fit a model.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in tree.body:
        # Definitions are not executed beyond binding a name.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        # The entry-point guard is the correct place for work.
        if isinstance(node, ast.If):
            test = ast.dump(node.test)
            if "__main__" in test or "__name__" in test:
                continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            func = sub.func
            name = getattr(func, "attr", getattr(func, "id", ""))
            if name in FORBIDDEN_CALLS:
                offenders.append((f"{name}()", getattr(sub, "lineno", node.lineno)))
    return offenders


@pytest.mark.parametrize("path", list(_modules()), ids=_module_name)
def test_module_does_no_work_at_import_time(path):
    offenders = _toplevel_work(path)
    assert not offenders, (
        f"{path.relative_to(SRC)} runs at import time: {offenders}. "
        "Move it into main() or another function — importing this package must "
        "never read artifacts, fit models or write files."
    )


@pytest.mark.parametrize("path", list(_modules()), ids=_module_name)
def test_module_imports_cleanly(path):
    """Also catches broken intra-package imports left over from migration."""
    module = _module_name(path)
    try:
        importlib.import_module(module)
    except ImportError as exc:
        # Optional heavy extras (torch, transformers, matplotlib) may be absent.
        missing = str(exc).lower()
        if any(dep in missing for dep in
               ("torch", "transformers", "sentence_transformers", "matplotlib",
                "datasets", "accelerate", "lxml", "joblib")):
            pytest.skip(f"optional dependency unavailable: {exc}")
        raise


def test_entry_points_expose_main():
    """Every stage the CLI dispatches to must have a main() to dispatch to."""
    from demap_repro.cli import STAGES

    missing = []
    for stage in STAGES:
        try:
            module = importlib.import_module(stage.module)
        except ImportError:
            continue          # optional extra; covered above
        if not callable(getattr(module, "main", None)):
            missing.append(stage.name)
    assert not missing, f"CLI stages without main(): {missing}"
