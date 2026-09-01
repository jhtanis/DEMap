"""Every registered stage must actually be callable through ``demap``.

``demap_repro.cli`` dispatches with ``entry(rest)`` — it hands the remaining
argv to the stage's ``main()``. A module whose ``main()`` takes no parameter
therefore raises ``TypeError`` the moment it is invoked through the dispatcher,
including for ``--help``.

This is not hypothetical. ``materialize-eval``, ``non-exact-eval`` and
``leakage-sensitivity`` all shipped in that state, and ``demap
leakage-sensitivity`` was a documented command in ``docs/reproducing.md`` that
could not run. The stage worked when invoked as ``python -m``, which is how they
were exercised during migration, so nothing caught it.

The signature check is static and needs no imports. The ``--help`` check is the
end-to-end one: it proves the dispatcher can reach the parser.
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.tier1

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

from demap_repro.cli import STAGES  # noqa: E402


def _module_path(dotted: str) -> pathlib.Path:
    return SRC / (dotted.replace(".", "/") + ".py")


@pytest.mark.parametrize("stage", STAGES, ids=lambda s: s.name)
def test_stage_main_accepts_argv(stage):
    """``main()`` must take an argv parameter, because the dispatcher passes one."""
    path = _module_path(stage.module)
    assert path.exists(), f"{stage.name}: {stage.module} has no file at {path}"

    tree = ast.parse(path.read_text())
    mains = [n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "main"]
    assert mains, f"{stage.name}: {stage.module} defines no module-level main()"

    args = mains[0].args
    positional = args.posonlyargs + args.args
    assert positional or args.vararg, (
        f"{stage.name}: {stage.module}.main() takes no arguments, but "
        f"demap_repro.cli calls it as main(rest). Give it 'argv=None' and pass "
        f"argv to parse_args()."
    )


@pytest.mark.parametrize("stage", STAGES, ids=lambda s: s.name)
def test_stage_responds_to_help(stage):
    """``demap <stage> --help`` must print usage, or say which extra it needs.

    What must never happen is a bare traceback. A reader who installed the
    documented quick start (`pip install -e '.[dev]'`) and asks a neural or
    figure stage for help should be told to install that extra, not shown a
    ModuleNotFoundError. Both outcomes are acceptable; a traceback is not.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "demap_repro.cli", stage.name, "--help"],
        capture_output=True, text=True, cwd=ROOT, timeout=180,
    )
    combined = proc.stdout + proc.stderr
    assert "Traceback" not in combined, (
        f"{stage.name}: --help raised\n{combined[-2000:]}"
    )
    if "optional dependency" in combined:
        assert "pip install -e" in combined, (
            f"{stage.name}: reported a missing extra without saying how to install it\n"
            f"{combined[-2000:]}")
        assert proc.returncode == 2, f"{stage.name}: expected exit 2, got {proc.returncode}"
        return
    assert proc.returncode == 0, (
        f"{stage.name}: --help exited {proc.returncode}\n{combined[-2000:]}"
    )
    assert "usage" in combined.lower(), (
        f"{stage.name}: --help printed no usage line\n{combined[-2000:]}"
    )
