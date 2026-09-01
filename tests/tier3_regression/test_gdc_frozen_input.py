"""The frozen GDC evaluation input.

GDC's gold CDE mappings were **manually curated**: a curator assigned, for each
GDC property, the caDSR CDE it maps to. That assignment is the benchmark label,
and it is not recoverable from any public GDC source — the GDC Data Dictionary
publishes properties and permissible values, so the query side has a public
analogue, but it publishes no property→CDE linkage and has no endpoint for one.

So this artifact is shipped, not regenerated. Without it the GDC column of every
reported table is unreproducible. The raw curation submissions are deliberately
not distributed: they carry workflow metadata and 1,822 rows for what is 120
curated queries, none of which reproduces anything.
"""
from __future__ import annotations

import hashlib
import pathlib

import pandas as pd
import pytest

pytestmark = [pytest.mark.tier3, pytest.mark.parity]

REPO = pathlib.Path(__file__).resolve().parents[2]
FROZEN = REPO / "data" / "frozen" / "gdc_combined.parquet"

#: The artifact as shipped.
SHA256 = "289c4f5fc966748c54fc3d4c8c0edda28937b55910796a65db7834034bf86f21"

#: Columns evaluation and reporting depend on.
REQUIRED_COLUMNS = {
    "query_id", "query_source", "query_field", "family",
    "query_text_raw", "query_text_q3",
    "cde_publicid", "cde_version", "cde_id",
}

#: The two GDC query styles the set combines, and their counts.
STYLE_COUNTS = {"GDC_QTXT": 39, "GDC_ALT": 33}


@pytest.fixture(scope="module")
def gdc():
    return pd.read_parquet(FROZEN)


def test_the_frozen_artifact_ships():
    assert FROZEN.is_file(), f"{FROZEN} is missing; it is a required evaluation input"


def test_digest_is_pinned():
    got = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    assert got == SHA256, (
        f"data/frozen/gdc_combined.parquet changed.\n  expected {SHA256}\n  got      {got}\n"
        "This is the exact evaluation input the manuscript used. If it must change, "
        "the GDC column of every reported table changes with it.")


def test_it_is_the_72_query_paper_population(gdc):
    from demap_repro.evaluation.canonical_datasets import PAPER_EVAL_QUERY_COUNTS

    assert len(gdc) == 72
    assert PAPER_EVAL_QUERY_COUNTS["gdc_combined"] == 72


def test_query_ids_are_unique(gdc):
    assert gdc["query_id"].nunique() == 72, "a duplicated query id would double-count"
    assert gdc["query_id"].notna().all()
    assert gdc["pair_id"].nunique() == 72


def test_schema_carries_everything_evaluation_needs(gdc):
    missing = REQUIRED_COLUMNS - set(gdc.columns)
    assert not missing, f"frozen GDC input is missing columns: {sorted(missing)}"


def test_it_combines_the_two_query_styles(gdc):
    """'combined' is question-text plus alternate-name queries."""
    assert gdc["query_source"].value_counts().to_dict() == STYLE_COUNTS
    assert set(gdc["query_field"]) == {"preferred_question_text", "alternate_name"}


def test_every_gold_is_populated_and_versioned(gdc):
    """The curated gold is the benchmark label; none may be missing."""
    assert gdc["cde_id"].notna().all(), "every GDC query must carry a gold CDE"
    assert gdc["cde_publicid"].notna().all()
    pattern = r"^\d+::\d+(\.\d+)?$"
    ok = gdc["cde_id"].astype(str).str.match(pattern)
    assert ok.all(), f"malformed gold ids: {sorted(set(gdc['cde_id'].astype(str)[~ok]))}"


def test_the_representation_evaluation_consumes_is_present(gdc):
    q3 = gdc["query_text_q3"].astype(str)
    assert (q3.str.strip() != "").all(), "every query needs a q3 representation"


def test_no_curation_workflow_metadata_travelled_with_it(gdc):
    """The raw submissions carry submitting user, comments, tips and 'do not
    use' flags. None of that is needed to reproduce anything, and none of it is
    here."""
    forbidden = {"Batch User", "batch_user", "Comments", "comments",
                 "Do Not Use", "do_not_use", "Entity User Tip", "PV VM User Tip"}
    present = forbidden & set(gdc.columns)
    assert not present, f"curation workflow metadata leaked into the artifact: {sorted(present)}"
    # And no cell should carry the submitting curator's identity.
    blob = gdc.astype(str).agg(" ".join, axis=1).str.cat(sep=" ").lower()
    for needle in ("warzel", "batch user", "do not use"):
        assert needle not in blob, f"{needle!r} appears in the shipped artifact"


# --------------------------------------------------------------------------
# downstream consumption
# --------------------------------------------------------------------------

def test_the_evaluation_registry_declares_this_dataset():
    """The artifact must be the one the pipeline actually reads."""
    import yaml

    reg = yaml.safe_load((REPO / "configs/paper/eval_datasets_v1.yaml").read_text())
    entry = next(d for d in reg["canonical"] if d["name"] == "gdc_combined")
    assert entry["path"].endswith("eval_canonical/gdc_combined.parquet")
    assert entry["reachable_rows"] == 72
    assert entry["query_text_q3"] is True


def test_downstream_code_consumes_gdc_combined():
    """It is a canonical evaluation dataset, not an auxiliary one."""
    from demap_repro.evaluation.canonical_datasets import PAPER_EVAL_DATASETS
    from demap_repro.data.leakage_filter import EXTERNAL_DATASETS

    assert "gdc_combined" in PAPER_EVAL_DATASETS
    assert "gdc_combined" in EXTERNAL_DATASETS


def test_the_leakage_filter_expects_the_documented_counts():
    """§S1.5: 4 of 72 GDC queries recur in training, leaving 68."""
    from demap_repro.data.leakage_filter import PAPER_COUNTS

    assert PAPER_COUNTS["gdc_combined"] == (72, 4, 68)


def test_both_frozen_inputs_are_accounted_for():
    """data/frozen/ holds exactly the two sets that cannot be rebuilt publicly."""
    frozen = sorted(p.name for p in (REPO / "data" / "frozen").glob("*.parquet"))
    assert frozen == ["cimac_v2.parquet", "gdc_combined.parquet"], (
        "data/frozen/ should hold exactly the two non-regenerable evaluation sets; "
        f"found {frozen}")


# --------------------------------------------------------------------------
# data/frozen/ is a distribution location; these guard the hand-off to the
# data tree the pipeline actually reads
# --------------------------------------------------------------------------

#: The two frozen evaluation inputs, and the canonical name each is copied to.
FROZEN_INPUTS = {
    "cimac_v2.parquet": "1d0797330864cb8a73d277c1885fa5bba41a63d3a011cfaaa148e59c6dce4fe0",
    "gdc_combined.parquet": "289c4f5fc966748c54fc3d4c8c0edda28937b55910796a65db7834034bf86f21",
}


def test_only_the_materializer_references_data_frozen():
    """One reader, and it reads. The shipped artifacts are immutable inputs.

    ``demap materialize-eval`` copies them into the data tree; nothing else in
    the package should know the path exists, and nothing at all should write to
    it. The behavioural half of this guarantee is in
    ``tests/tier2_behavior/test_materialize_frozen.py``.
    """
    import re

    referencing = set()
    for path in (REPO / "src").rglob("*.py"):
        for line in path.read_text().splitlines():
            if re.search(r"""["'][^"']*data/frozen[^"']*["']""", line) or '"frozen"' in line:
                if "data" in line or "frozen_dir" in line:
                    referencing.add(str(path.relative_to(REPO)))
    assert referencing <= {"src/demap_repro/data/cli/materialize_eval.py"}, (
        "data/frozen/ should be reachable only from the materialization stage; "
        f"also referenced by {sorted(referencing - {'src/demap_repro/data/cli/materialize_eval.py'})}")


def test_the_materializer_never_opens_the_frozen_source_for_writing():
    """Static check that the copy is one-directional."""
    src = (REPO / "src/demap_repro/data/cli/materialize_eval.py").read_text()
    # the only filesystem writes are to the destination
    assert "shutil.copy2(src, dst)" in src, "the copy must go source -> destination"
    assert "copy2(dst, src)" not in src
    for bad in ('open(src, "w"', "open(src, 'w'", "src.write_", "src.unlink", "src.rename"):
        assert bad not in src, f"materialize_eval writes to the frozen source: {bad}"


def test_a_missing_canonical_split_fails_loudly_not_silently(tmp_path):
    """If a reader skips the documented copy, evaluation must stop, not guess.

    ``by_dataset`` returning partial results would silently drop a whole
    evaluation dataset from the reported table while looking successful.
    """
    import pandas as pd
    from demap_repro.evaluation.by_dataset import by_dataset

    scored = pd.DataFrame({
        "split": ["cimac_v2", "cimac_v2"],
        "query_id": ["q1", "q2"],
        "cde_id": ["1::1", "2::1"],
        "is_label": [True, False],
        "hgbc_rank": [1, 2],
    })
    empty = tmp_path / "eval_canonical"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="cimac_v2"):
        by_dataset(scored, empty, label="probe")

    # ...and only an explicit opt-in may proceed without it.
    out = by_dataset(scored, empty, label="probe", allow_missing=True)
    assert out.empty


@pytest.mark.needs_artifacts
def test_the_data_tree_holds_the_shipped_frozen_inputs(artifact_root):
    """Guards the copy step: the data tree's copy must BE the shipped artifact.

    Removing the frozen file is caught by the test above. This catches the other
    half — a stale or substituted copy in the data tree, which would evaluate
    successfully against a different population and report different numbers.
    """
    import hashlib

    canonical = artifact_root / "data" / "processed" / "eval_canonical"
    if not canonical.is_dir():
        pytest.skip(f"no canonical evaluation directory at {canonical}")

    for name, want in FROZEN_INPUTS.items():
        target = canonical / name
        if not target.is_file():
            pytest.skip(f"{name} not present in the data tree")
        got = hashlib.sha256(target.read_bytes()).hexdigest()
        assert got == want, (
            f"{target} is not the shipped frozen input.\n"
            f"  expected {want}\n  got      {got}\n"
            f"Copy it from data/frozen/{name}; a different file evaluates a "
            f"different population and changes the reported numbers.")
