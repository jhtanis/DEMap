"""Pin migrated logic to the implementation that produced the published results.

Most of this repository was migrated from a research repository under a
copy-first rule: copy the exact bytes that ran, prove parity, and only then
refactor. Re-running a stage to prove parity is often impossible for a reader —
it needs the full artifact tree, GPUs, and in some cases components whose
redistribution is unresolved.

This module provides the offline half of that guarantee. For every migrated
function it records a hash of the function's **abstract syntax tree**, stripped
of docstrings and of the names that renaming legitimately changes. The hash is
insensitive to comments, formatting, blank lines and the function's own name; it
is sensitive to any change in what the code actually does.

The hashes were computed from the research-repository sources at migration time
and are stored in ``tests/fixtures/source_parity.json``. The test suite recomputes
them from this repository and fails on any difference, so a later "cleanup" that
alters behaviour cannot pass review unnoticed — and the check runs anywhere, with
no data, no weights and no network.

It is a guard against silent drift, not a proof of correctness: two functions
with the same AST hash do the same thing, but agreeing hashes say nothing about
whether the original was right. Numerical parity against the published artifacts
is asserted separately, in ``tests/tier3_regression/``.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Dict, Iterable, Optional

__all__ = ["function_ast_hash", "module_function_hashes", "normalize_function"]


def _strip_docstring(node: ast.AST) -> None:
    """Drop a leading string-literal expression from a function/class/module body."""
    body = getattr(node, "body", None)
    if not body:
        return
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
            and isinstance(first.value.value, str):
        del body[0]


class _AliasRewriter(ast.NodeTransformer):
    """Rewrite declared symbol aliases to a canonical name.

    Migration moved some call targets behind a seam — a gated dependency reached
    through an adapter, or a helper promoted out of a private module. Where the
    *called implementation is the same*, the alias map says so explicitly, one
    entry per rename, and the hash ignores the spelling. Anything not declared in
    the map still counts as a difference, so the check cannot be widened by
    accident.
    """

    def __init__(self, aliases: Dict[str, str]):
        self.aliases = aliases

    @staticmethod
    def _dotted(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = _AliasRewriter._dotted(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    def _canonical(self, node: ast.AST) -> Optional[ast.AST]:
        dotted = self._dotted(node)
        if dotted is not None and dotted in self.aliases:
            return ast.copy_location(ast.Name(id=self.aliases[dotted], ctx=ast.Load()), node)
        return None

    def visit_Call(self, node: ast.Call):  # noqa: N802
        # Collapse a declared zero-argument accessor to the constant it returns:
        # `iface.cdematch_feature_columns()` and `CDEMATCH_FEATURE_COLUMNS` denote
        # the same value, and the migration replaced one spelling with the other.
        dotted = self._dotted(node.func)
        if dotted is not None and not node.args and not node.keywords:
            canonical = self.aliases.get(f"{dotted}()")
            if canonical is not None:
                return ast.copy_location(ast.Name(id=canonical, ctx=ast.Load()), node)
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):  # noqa: N802
        replacement = self._canonical(node)
        if replacement is not None:
            return replacement
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name):  # noqa: N802
        replacement = self._canonical(node)
        return replacement if replacement is not None else node


def _drop_inner_imports(node: ast.AST) -> None:
    """Remove ``import`` statements from inside a function body.

    Hoisting a function-local import to module scope binds the same symbol and
    cannot change behaviour, so it is normalized away. The *call sites* are still
    compared, so importing a different symbol remains visible.
    """
    for sub in ast.walk(node):
        body = getattr(sub, "body", None)
        if isinstance(body, list):
            sub.body = [s for s in body if not isinstance(s, (ast.Import, ast.ImportFrom))]


def normalize_function(node: ast.AST, aliases: Optional[Dict[str, str]] = None,
                       drop_inner_imports: bool = True) -> ast.AST:
    """Return ``node`` with docstrings removed and its own name neutralized.

    Renaming a function during migration (``_feature_summary`` ->
    ``feature_summary``) is a naming change, not a behaviour change, so the
    function's own name is excluded from the hash. Names it *references* are
    kept unless ``aliases`` declares the rename: calling a different function IS
    a behaviour change.
    """
    for sub in ast.walk(node):
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            _strip_docstring(sub)
    if drop_inner_imports:
        _drop_inner_imports(node)
    if aliases:
        node = _AliasRewriter(aliases).visit(node)
        ast.fix_missing_locations(node)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        node.name = "_"
    return node


def function_ast_hash(source: str, func_name: str,
                      aliases: Optional[Dict[str, str]] = None) -> str:
    """SHA256 of the normalized AST of ``func_name`` defined in ``source``.

    Only top-level definitions are considered, which is what every migrated
    module uses.
    """
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            dumped = ast.dump(normalize_function(node, aliases), annotate_fields=True,
                              include_attributes=False)
            return hashlib.sha256(dumped.encode("utf-8")).hexdigest()
    raise LookupError(f"no top-level function {func_name!r} in the given source")


def module_function_hashes(path: Path, names: Optional[Iterable[str]] = None) -> Dict[str, str]:
    """Map function name -> AST hash for ``path``.

    ``names`` selects a subset; omit it to hash every top-level function.
    """
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    available = [n.name for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    wanted = list(names) if names is not None else available
    missing = [n for n in wanted if n not in available]
    if missing:
        raise LookupError(f"{path}: missing top-level functions {missing}")
    return {n: function_ast_hash(source, n) for n in wanted}
