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
import re

import pytest

pytestmark = pytest.mark.tier1

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
ROOT = SRC.parent

#: Top-level module name -> the ``[project.optional-dependencies]`` extra that
#: installs it. A module-level import may only be forgiven when it appears here,
#: which forces every optional dependency to be declared in ``pyproject.toml``.
OPTIONAL_MODULE_EXTRA = {
    "torch": "neural",
    "transformers": "neural",
    "sentence_transformers": "neural",
    "accelerate": "neural",
    "datasets": "neural",
    "huggingface_hub": "neural",
    "safetensors": "neural",
    "matplotlib": "figures",
    "lxml": "extract",
    "openpyxl": "excel",
    "pptx": "documents",
    "docx": "documents",
}

#: Distribution name on PyPI, where it differs from the import name.
_DISTRIBUTION_NAME = {
    "sentence_transformers": "sentence-transformers",
    "pptx": "python-pptx",
    "docx": "python-docx",
}

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
        # Optional extras may be absent. The forgiveness list is keyed on the
        # *declared* extra, not on a substring of the message: a module whose
        # dependency nobody declared is a packaging defect, and matching loosely
        # on the text is what let ``pptx``/``docx`` fail here unnoticed.
        name = getattr(exc, "name", None)
        extra = OPTIONAL_MODULE_EXTRA.get(name)
        if extra is not None:
            pytest.skip(f"optional dependency {name!r} unavailable; "
                        f"install it with: pip install -e '.[{extra}]'")
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


# --------------------------------------------------------------------------
# The guards on the forgiveness list itself.
#
# ``test_module_imports_cleanly`` skips when an optional dependency is absent.
# That escape hatch is only honest if every module it forgives is actually
# installable from this project's own metadata — otherwise a module with an
# undeclared dependency is silently reclassified as "optional" and can never be
# imported from a clean clone. Both ``pptx`` and ``docx`` reached the released
# repository that way.
# --------------------------------------------------------------------------

def _optional_dependency_extras():
    """``[project.optional-dependencies]`` as ``{extra: [requirement, ...]}``.

    ``tomllib`` is 3.11+, and this project supports 3.10, so fall back to a
    reader for the one section shape this file uses: ``name = [ "a", "b" ]``,
    on one line or spread over several.
    """
    try:
        import tomllib
    except ModuleNotFoundError:
        pass
    else:
        with (ROOT / "pyproject.toml").open("rb") as fh:
            return tomllib.load(fh)["project"]["optional-dependencies"]

    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = re.search(
        r"^\[project\.optional-dependencies\]\n(.*?)(?=^\[|\Z)",
        text, flags=re.S | re.M)
    assert section, "pyproject.toml has no [project.optional-dependencies]"
    body = re.sub(r"#[^\n]*", "", section.group(1))
    # Split on each ``name = [`` header and take everything up to the next one;
    # bracket matching would otherwise trip over "DEMap[neural]" inside a list.
    starts = list(re.finditer(r"^(\w[\w.-]*)\s*=\s*\[", body, flags=re.M))
    extras = {}
    for i, match in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(body)
        extras[match.group(1)] = re.findall(r'"([^"]+)"', body[match.end():end])
    return extras


def test_optional_modules_are_declared_in_pyproject():
    """Every forgiven import must be pinned under the extra that claims it."""
    extras = _optional_dependency_extras()
    undeclared = []
    for module, extra in sorted(OPTIONAL_MODULE_EXTRA.items()):
        requirements = extras.get(extra)
        if requirements is None:
            undeclared.append(f"{module}: extra '{extra}' does not exist")
            continue
        distribution = _DISTRIBUTION_NAME.get(module, module)
        # Requirement strings carry version specifiers; compare on the name.
        names = {re.split(r"[<>=!~\[ ]", r, maxsplit=1)[0].lower().replace("_", "-")
                 for r in requirements}
        if distribution.lower().replace("_", "-") not in names:
            undeclared.append(f"{module}: '{distribution}' not pinned under [{extra}]")
    assert not undeclared, (
        "modules forgiven by test_module_imports_cleanly but not installable "
        f"from pyproject.toml: {undeclared}")


def test_all_extra_includes_every_optional_extra():
    """``pip install -e '.[all]'`` must actually install everything optional."""
    extras = _optional_dependency_extras()
    referenced = {re.search(r"\[([^\]]+)\]", r).group(1)
                  for r in extras["all"] if "[" in r}
    optional = set(extras) - {"all"}
    assert optional <= referenced, (
        f"extras missing from [all]: {sorted(optional - referenced)}")
