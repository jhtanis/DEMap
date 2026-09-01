"""Every config a manifest declares as required must actually be present.

``manifests/paper_scope.yaml`` names, per result, the configuration a reader
needs to reproduce it. Three items declared ``configs/paper/dataset_v1.yaml``,
which has never existed in this repository or the research one — the real
contract is ``configs/pipeline.yaml``, which is what ``demap make-dataset``
defaults to and which was itself missing here.

A declared-but-absent config is worse than an undeclared one: it reads as a
promise. Nothing caught it because no test related the manifests to the files on
disk, and because the only stage that reads it needs a data tree to run.
"""
from __future__ import annotations

import pathlib

import pytest
import yaml

pytestmark = pytest.mark.tier1

REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def scope():
    return yaml.safe_load((REPO / "manifests" / "paper_scope.yaml").read_text())


def _declared_configs(scope) -> set[str]:
    out = set()
    for item in scope["items"]:
        for cfg in item.get("required_config") or []:
            out.add(str(cfg))
    return out


def test_every_required_config_exists(scope):
    missing = sorted(c for c in _declared_configs(scope) if not (REPO / c).is_file())
    assert not missing, (
        "paper_scope declares configs that do not exist:\n  " + "\n  ".join(missing))


def test_required_configs_are_actually_declared(scope):
    """Guards the reverse: the set is non-empty, so the test cannot pass vacuously."""
    assert len(_declared_configs(scope)) >= 5


def test_the_dataset_pipeline_contract_is_present():
    """`demap make-dataset` defaults to this path, so a clean clone needs it."""
    cfg = REPO / "configs" / "pipeline.yaml"
    assert cfg.is_file(), "configs/pipeline.yaml is the make-dataset default and must ship"
    spec = yaml.safe_load(cfg.read_text())["pipeline"]
    for stage in ("extract", "merge", "enrich", "build_queries", "build_pairs",
                  "split", "catalog", "emit"):
        assert stage in spec, f"pipeline.yaml is missing the {stage} stage"


def test_the_pipeline_config_only_references_files_that_ship():
    """Its allowlist paths must resolve inside the repository."""
    spec = yaml.safe_load((REPO / "configs" / "pipeline.yaml").read_text())["pipeline"]
    for key in ("alt_allowlist", "refdoc_allowlist"):
        rel = spec["build_queries"][key]
        assert (REPO / rel).is_file(), f"pipeline.yaml build_queries.{key} -> {rel} is absent"


#: Keys whose value is a historical provenance record - where migrated bytes came
#: from - rather than a path anything resolves at runtime.
PROVENANCE_KEYS = ("source_repository",)


def test_no_config_hardcodes_a_developer_path():
    """No RUNTIME path in configs/ may resolve into the authors' workspace.

    Provenance blocks are exempt and enumerated above: recording which checkout
    a byte was copied from is the point of a migration ledger. Everything else
    must resolve through DEMAP_DATA_ROOT or be repo-relative.
    """
    offenders = []
    for path in sorted((REPO / "configs").rglob("*")):
        if not (path.is_file() and path.suffix in {".yaml", ".yml", ".json"}):
            continue
        in_provenance = False
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            stripped = line.strip()
            if any(stripped.startswith(k + ":") for k in PROVENANCE_KEYS):
                in_provenance = True
                continue
            if in_provenance and (not line.startswith((" ", "\t")) or not stripped):
                in_provenance = False
            if in_provenance:
                continue
            for needle in ("/data/nextgen2", "/vf/users", "/home/"):
                if needle in line:
                    offenders.append(f"{path.relative_to(REPO)}:{lineno}: {stripped[:80]}")
    assert not offenders, (
        "configs resolve absolute developer paths at runtime:\n  " + "\n  ".join(offenders))
