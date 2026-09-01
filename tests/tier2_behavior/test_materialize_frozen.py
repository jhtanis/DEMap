"""``demap materialize-eval`` puts the shipped frozen inputs where evaluation reads.

Two of the six canonical evaluation datasets cannot be rebuilt from public
inputs in the form the manuscript used, so they ship in ``data/frozen/``.
Getting them into the data tree used to be a manual ``cp``, which is easy to
skip and easy to do with the wrong file. It is now part of the stage whose job
that already was.

What matters here is not that the copy happens but that it cannot happen
*wrongly*: digests are verified on the way in and on the way out, an existing
destination that differs is refused rather than overwritten, and repeating the
command is safe.
"""
from __future__ import annotations

import hashlib
import shutil

import pytest

from demap_repro.data.cli.materialize_eval import (
    DERIVED,
    FROZEN_INPUTS,
    frozen_dir,
    materialize_frozen,
)

pytestmark = pytest.mark.tier2


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# the shipped artifacts
# --------------------------------------------------------------------------

def test_frozen_dir_resolves_into_the_source_tree():
    """Repo-relative, so it works from any working directory."""
    d = frozen_dir()
    assert d.is_dir(), f"{d} should be the committed data/frozen/"
    assert d.name == "frozen" and d.parent.name == "data"
    assert (d / "cimac_v2.parquet").is_file()
    assert (d / "gdc_combined.parquet").is_file()


def test_the_two_frozen_datasets_are_declared():
    assert set(FROZEN_INPUTS) == {"cimac_v2", "gdc_combined"}
    assert set(DERIVED) == {"test", "cctg", "oid_alt", "cdash"}
    assert not set(FROZEN_INPUTS) & set(DERIVED), "a dataset cannot be both"


@pytest.mark.parametrize("name", sorted(FROZEN_INPUTS))
def test_shipped_artifacts_match_their_pinned_digests(name):
    filename, want = FROZEN_INPUTS[name]
    assert _sha256(frozen_dir() / filename) == want


# --------------------------------------------------------------------------
# materialization
# --------------------------------------------------------------------------

def test_materializes_both_to_the_canonical_location(tmp_path):
    out = tmp_path / "eval_canonical"
    rows = materialize_frozen(out)

    assert {r["dataset"] for r in rows} == {"cimac_v2", "gdc_combined"}
    assert all(r["action"] == "copied" for r in rows)
    for name, (filename, want) in FROZEN_INPUTS.items():
        dst = out / f"{name}.parquet"
        assert dst.is_file(), f"{name} was not materialized"
        assert _sha256(dst) == want, f"{name} materialized with the wrong bytes"


def test_it_creates_the_destination_directory(tmp_path):
    out = tmp_path / "deep" / "nested" / "eval_canonical"
    assert not out.exists()
    materialize_frozen(out)
    assert (out / "cimac_v2.parquet").is_file()


def test_repeated_execution_is_safe(tmp_path):
    out = tmp_path / "eval_canonical"
    materialize_frozen(out)
    first = {n: _sha256(out / f"{n}.parquet") for n in FROZEN_INPUTS}

    rows = materialize_frozen(out)
    assert all(r["action"] == "skipped" for r in rows), "a matching destination is left alone"
    assert {n: _sha256(out / f"{n}.parquet") for n in FROZEN_INPUTS} == first


def test_a_conflicting_destination_fails_clearly(tmp_path):
    """Silently overwriting would change the evaluated population without saying so."""
    out = tmp_path / "eval_canonical"
    out.mkdir(parents=True)
    (out / "cimac_v2.parquet").write_bytes(b"a different evaluation set")

    with pytest.raises(SystemExit) as exc:
        materialize_frozen(out)
    msg = str(exc.value)
    assert "differs from the shipped frozen input" in msg
    assert "Refusing to overwrite" in msg
    # and the pre-existing file is untouched
    assert (out / "cimac_v2.parquet").read_bytes() == b"a different evaluation set"


def test_a_tampered_source_is_refused(tmp_path):
    """The digest is checked on the way in, not only on the way out."""
    src = tmp_path / "frozen"
    src.mkdir()
    for filename, _ in FROZEN_INPUTS.values():
        shutil.copy2(frozen_dir() / filename, src / filename)
    (src / "gdc_combined.parquet").write_bytes(b"substituted")

    with pytest.raises(SystemExit) as exc:
        materialize_frozen(tmp_path / "out", src_dir=src)
    assert "does not match its pinned digest" in str(exc.value)


def test_a_missing_source_fails_clearly(tmp_path):
    with pytest.raises(SystemExit) as exc:
        materialize_frozen(tmp_path / "out", src_dir=tmp_path / "nonexistent")
    assert "shipped frozen input missing" in str(exc.value)


def test_dry_run_writes_nothing(tmp_path):
    out = tmp_path / "eval_canonical"
    materialize_frozen(out, dry_run=True)
    assert not out.exists() or not list(out.glob("*.parquet"))


# --------------------------------------------------------------------------
# data/frozen/ stays immutable
# --------------------------------------------------------------------------

def test_materializing_does_not_touch_data_frozen(tmp_path):
    before = {p.name: _sha256(p) for p in sorted(frozen_dir().glob("*.parquet"))}
    materialize_frozen(tmp_path / "out")
    after = {p.name: _sha256(p) for p in sorted(frozen_dir().glob("*.parquet"))}
    assert after == before, "data/frozen/ is a read-only distribution location"


def test_the_destination_is_never_data_frozen(tmp_path):
    """Guards against a caller pointing the output back at the source."""
    out = frozen_dir()
    # Materializing into data/frozen/ finds matching digests and no-ops, rather
    # than rewriting the shipped files.
    rows = materialize_frozen(out)
    assert all(r["action"] == "skipped" for r in rows)


# --------------------------------------------------------------------------
# evaluation consumes what was materialized
# --------------------------------------------------------------------------

def test_the_materialized_names_are_the_registry_names(tmp_path):
    """The filenames written must be the ones the evaluation registry declares."""
    import yaml
    from demap_repro.utils.paths import repo_root

    reg = yaml.safe_load((repo_root() / "configs/paper/eval_datasets_v1.yaml").read_text())
    declared = {d["name"]: d["path"] for d in reg["canonical"]}

    out = tmp_path / "eval_canonical"
    materialize_frozen(out)
    for name in FROZEN_INPUTS:
        assert declared[name].endswith(f"eval_canonical/{name}.parquet")
        assert (out / f"{name}.parquet").is_file()


def test_evaluation_reads_the_materialized_files(tmp_path):
    """A materialized tree satisfies the reachable-query lookup evaluation uses."""
    import pandas as pd
    from demap_repro.evaluation.by_dataset import by_dataset

    out = tmp_path / "eval_canonical"
    materialize_frozen(out)

    cimac = pd.read_parquet(out / "cimac_v2.parquet", columns=["query_id"])
    scored = pd.DataFrame({
        "split": ["cimac_v2"] * 3,
        "query_id": cimac["query_id"].astype(str).head(3).tolist(),
        "cde_id": ["1::1", "2::1", "3::1"],
        "is_label": [True, True, True],
        "hgbc_rank": [1, 2, 3],
    })
    res = by_dataset(scored, out, label="probe")
    assert len(res) == 1
    assert res.iloc[0]["split"] == "cimac_v2"
    # denominator is the materialized query set, not the scored rows
    assert res.iloc[0]["n"] == 131
