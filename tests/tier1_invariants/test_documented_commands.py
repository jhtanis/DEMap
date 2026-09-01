"""Every ``demap`` command the documentation names must be a real stage.

Documentation that names a command which does not exist is worse than no
documentation: the reader assumes the fault is theirs. This drifts silently
whenever a stage is renamed, so it is a test rather than a review item.

It also catches the reverse for the workflow pages — a stage that exists but is
mentioned nowhere is either undocumented or dead.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from demap_repro.cli import BY_NAME, STAGES

pytestmark = pytest.mark.tier1

REPO = pathlib.Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"

#: ``demap <word>`` where the word is not a flag.
COMMAND = re.compile(r"demap ([a-zA-Z][a-zA-Z0-9-]*)")

#: Stages a reader is not expected to invoke directly from the workflow pages.
#: Each is reachable, but through another documented stage or a manifest.
NOT_REQUIRED_IN_DOCS: set[str] = set()


def _doc_files() -> list[pathlib.Path]:
    return sorted(p for p in DOCS.rglob("*.md")) + [REPO / "README.md"]


def _mentions() -> dict[str, set[str]]:
    """command -> the docs that name it."""
    found: dict[str, set[str]] = {}
    for path in _doc_files():
        if not path.is_file():
            continue
        for match in COMMAND.finditer(path.read_text()):
            name = match.group(1)
            found.setdefault(name, set()).add(str(path.relative_to(REPO)))
    return found


def test_documentation_names_only_real_stages():
    mentions = _mentions()
    bogus = {
        name: sorted(where) for name, where in mentions.items()
        if name not in BY_NAME and name not in {"list", "help"}
    }
    assert not bogus, (
        "documentation names commands that are not stages:\n  "
        + "\n  ".join(f"demap {n} -> {w}" for n, w in sorted(bogus.items())))


def test_the_check_is_not_vacuous():
    """If the regex ever stops matching, this test would pass on nothing."""
    mentions = _mentions()
    real = {n for n in mentions if n in BY_NAME}
    assert len(real) >= 20, f"only found {len(real)} documented commands; regex likely broken"


def test_every_stage_is_documented_somewhere():
    """A stage nobody documents is either undocumented or dead."""
    mentions = _mentions()
    undocumented = sorted(
        s.name for s in STAGES
        if s.name not in mentions and s.name not in NOT_REQUIRED_IN_DOCS)
    assert not undocumented, (
        "stages that appear in no documentation:\n  " + "\n  ".join(undocumented))


def test_the_core_docs_exist():
    """The README points at these; a dangling link is a broken front door."""
    for name in ("data_sources.md", "building_datasets.md", "environment.md",
                 "running_experiments.md", "reproducing.md", "manuscript_map.md",
                 "provenance.md", "slurm.md"):
        assert (DOCS / name).is_file(), f"docs/{name} is missing"


def test_no_doc_hardcodes_a_developer_path():
    """No documented path may resolve into the authors' workspace."""
    offenders = []
    for path in _doc_files():
        if not path.is_file():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for needle in ("/data/nextgen2", "/vf/users"):
                if needle in line:
                    offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()[:80]}")
    assert not offenders, (
        "documentation hardcodes developer paths:\n  " + "\n  ".join(offenders))


def test_internal_doc_links_resolve():
    """Relative markdown links between docs must point at files that exist."""
    link = re.compile(r"\[[^\]]+\]\((?!https?:)([^)#]+)(?:#[^)]*)?\)")
    broken = []
    for path in _doc_files():
        if not path.is_file():
            continue
        for match in link.finditer(path.read_text()):
            target = (path.parent / match.group(1)).resolve()
            if not target.exists():
                broken.append(f"{path.relative_to(REPO)} -> {match.group(1)}")
    assert not broken, "broken internal links:\n  " + "\n  ".join(broken)
