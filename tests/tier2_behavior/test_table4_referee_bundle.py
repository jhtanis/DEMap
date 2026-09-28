"""Strict Stage-1 Table 4 contract and bundle-verifier behavior."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from demap_repro.reporting.table4_bundle import (
    MANIFEST_FILE,
    QUERY_FILE,
    RANKING_FILE,
    Table4VerificationError,
    canonical_hgbc_order,
    contract_methods,
    default_contract_path,
    load_contract,
    query_set_identity_sha256,
    sha256_file,
    validate_ft_medcpt_order,
    verify_bundle,
    write_query_parquet,
    write_ranking_parquet,
)

pytestmark = pytest.mark.tier2
REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_BUNDLE = REPO_ROOT / "data/frozen/table4_v1"


def _synthetic_bundle(directory: Path) -> Path:
    """Create a full-cardinality but compact six-by-six contract fixture."""
    directory.mkdir(parents=True, exist_ok=True)
    contract_path = default_contract_path()
    contract = load_contract(contract_path)
    query_rows = []
    rank_rows = []
    public_id = 1_000_000
    for dataset in contract["datasets"]:
        dataset_id = dataset["id"]
        denominator = int(dataset["denominator"])
        for index in range(denominator):
            query_id = f"{dataset_id}:{index:05d}"
            gold = str(public_id)
            public_id += 1
            query_rows.append({
                "dataset": dataset_id,
                "query_id": query_id,
                "gold_public_ids": [gold],
            })
            for method in contract["methods"]:
                method_id = method["id"]
                hit = index < int(dataset["results"][method_id]["hits_at_5"])
                ids = [gold] if hit else []
                ranks = [1] if hit else []
                scores = ([1.0] if hit else []) if method_id in {
                    "final_reranker", "ft_medcpt"} else None
                rank_rows.append({
                    "dataset": dataset_id,
                    "method": method_id,
                    "query_id": query_id,
                    "ranked_public_ids": ids,
                    "source_ranks": ranks,
                    "scores": scores,
                    "returned_depth": len(ids),
                    "ranking_policy": method["ranking_policy"],
                    "source_artifact_id": f"synthetic_{method_id}",
                })
    write_query_parquet(query_rows, directory / QUERY_FILE)
    write_ranking_parquet(rank_rows, directory / RANKING_FILE)
    queries = pq.read_table(directory / QUERY_FILE).to_pandas()
    rankings = pq.read_table(directory / RANKING_FILE).to_pandas()
    manifest = {
        "schema_version": 1,
        "bundle_id": "synthetic_table4_v1",
        "results_contract": {
            "contract_id": contract["contract_id"],
            "sha256": sha256_file(contract_path),
        },
        "query_set_identity_sha256": query_set_identity_sha256(queries),
        "query_rows": len(queries),
        "ranking_rows": len(rankings),
        "dataset_query_counts": {
            item["id"]: int(item["denominator"]) for item in contract["datasets"]},
        "method_row_counts": {method: 7349 for method in contract_methods(contract)},
        "method_maximum_depths": {
            item["id"]: int(item["maximum_depth"]) for item in contract["methods"]},
        "ranking_policies": {
            item["id"]: item["ranking_policy"] for item in contract["methods"]},
        "files": {
            QUERY_FILE: {
                "sha256": sha256_file(directory / QUERY_FILE),
                "bytes": (directory / QUERY_FILE).stat().st_size,
                "rows": len(queries),
            },
            RANKING_FILE: {
                "sha256": sha256_file(directory / RANKING_FILE),
                "bytes": (directory / RANKING_FILE).stat().st_size,
                "rows": len(rankings),
            },
        },
        "source_artifacts": [
            {"artifact_id": f"synthetic_{method}", "sha256": "0" * 64,
             "bytes": 0, "role": f"rankings:{method}"}
            for method in contract_methods(contract)
        ],
        "expected_results": {
            item["id"]: item["results"] for item in contract["datasets"]},
    }
    (directory / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return directory


@pytest.fixture(scope="module")
def base_bundle(tmp_path_factory) -> Path:
    return _synthetic_bundle(tmp_path_factory.mktemp("table4-bundle"))


def _copy(base: Path, target: Path) -> Path:
    shutil.copytree(base, target)
    return target


def _manifest(bundle: Path) -> dict:
    return json.loads((bundle / MANIFEST_FILE).read_text(encoding="utf-8"))


def _write_manifest(bundle: Path, manifest: dict) -> None:
    (bundle / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_file(bundle: Path, name: str, *, update_rows: bool = True) -> None:
    manifest = _manifest(bundle)
    path = bundle / name
    manifest["files"][name]["sha256"] = sha256_file(path)
    manifest["files"][name]["bytes"] = path.stat().st_size
    if update_rows:
        manifest["files"][name]["rows"] = pq.ParquetFile(path).metadata.num_rows
    _write_manifest(bundle, manifest)


def _read_rankings(bundle: Path) -> pd.DataFrame:
    return pq.read_table(bundle / RANKING_FILE).to_pandas()


def _rewrite_rankings(bundle: Path, frame: pd.DataFrame) -> None:
    write_ranking_parquet(frame.to_dict("records"), bundle / RANKING_FILE)
    _refresh_file(bundle, RANKING_FILE)


def _rewrite_queries(bundle: Path, frame: pd.DataFrame, *, refresh_identity: bool = False) -> None:
    write_query_parquet(frame.to_dict("records"), bundle / QUERY_FILE)
    _refresh_file(bundle, QUERY_FILE)
    if refresh_identity:
        manifest = _manifest(bundle)
        manifest["query_set_identity_sha256"] = query_set_identity_sha256(frame)
        _write_manifest(bundle, manifest)


def test_results_contract_is_complete_and_lossless():
    contract = load_contract()
    assert sum(item["denominator"] for item in contract["datasets"]) == 7349
    assert sum(len(item["results"]) for item in contract["datasets"]) == 36
    assert contract["metric"]["unreachable_gold_queries_excluded"] is True


def test_happy_path_bundle_verification(base_bundle):
    manifest, metrics = verify_bundle(base_bundle, write_outputs=False)
    assert manifest["query_rows"] == 7349
    assert manifest["ranking_rows"] == 44094
    assert metrics["test"]["final_reranker"]["hits_at_5"] == 3846


def test_committed_bundle_verifies_all_published_cells():
    manifest, metrics = verify_bundle(COMMITTED_BUNDLE, write_outputs=False)
    assert manifest["query_rows"] == 7349
    assert manifest["ranking_rows"] == 44094
    assert metrics["test"]["final_reranker"]["hits_at_5"] == 3846
    assert metrics["cimac"]["final_reranker"]["hits_at_5"] == 105


def test_all_36_cells_are_checked_exactly(base_bundle):
    contract = load_contract()
    _, metrics = verify_bundle(base_bundle, write_outputs=False)
    observed = {
        (dataset, method): values["hits_at_5"]
        for dataset, methods in metrics.items() for method, values in methods.items()
    }
    expected = {
        (dataset["id"], method): cell["hits_at_5"]
        for dataset in contract["datasets"] for method, cell in dataset["results"].items()
    }
    assert observed == expected


def test_missing_bundle_file_is_a_hard_failure(base_bundle, tmp_path):
    bundle = _copy(base_bundle, tmp_path / "bundle")
    (bundle / QUERY_FILE).unlink()
    with pytest.raises(Table4VerificationError, match="missing required bundle file"):
        verify_bundle(bundle, write_outputs=False)


def test_changed_file_byte_fails_checksum(base_bundle, tmp_path):
    bundle = _copy(base_bundle, tmp_path / "bundle")
    with (bundle / QUERY_FILE).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(Table4VerificationError, match="byte-size mismatch|SHA-256 mismatch"):
        verify_bundle(bundle, write_outputs=False)


def test_wrong_manifest_row_count_is_a_hard_failure(base_bundle, tmp_path):
    bundle = _copy(base_bundle, tmp_path / "bundle")
    manifest = _manifest(bundle)
    manifest["ranking_rows"] = 44093
    _write_manifest(bundle, manifest)
    with pytest.raises(Table4VerificationError, match="manifest ranking row count mismatch"):
        verify_bundle(bundle, write_outputs=False)


def test_missing_query_is_a_hard_failure(base_bundle, tmp_path):
    bundle = _copy(base_bundle, tmp_path / "bundle")
    queries = pq.read_table(bundle / QUERY_FILE).to_pandas().iloc[:-1].copy()
    _rewrite_queries(bundle, queries)
    with pytest.raises(Table4VerificationError, match="query row count mismatch"):
        verify_bundle(bundle, write_outputs=False)


def test_extra_query_is_a_hard_failure(base_bundle, tmp_path):
    bundle = _copy(base_bundle, tmp_path / "bundle")
    queries = pq.read_table(bundle / QUERY_FILE).to_pandas()
    extra = pd.DataFrame([{
        "dataset": "test", "query_id": "test:extra", "gold_public_ids": ["99999999"]}])
    _rewrite_queries(bundle, pd.concat([queries, extra], ignore_index=True))
    with pytest.raises(Table4VerificationError, match="query row count mismatch"):
        verify_bundle(bundle, write_outputs=False)


def test_dataset_reassignment_is_a_hard_failure(base_bundle, tmp_path):
    bundle = _copy(base_bundle, tmp_path / "bundle")
    queries = pq.read_table(bundle / QUERY_FILE).to_pandas()
    queries.loc[0, "dataset"] = "cctg"
    _rewrite_queries(bundle, queries, refresh_identity=True)
    with pytest.raises(Table4VerificationError, match="membership/count mismatch"):
        verify_bundle(bundle, write_outputs=False)


def test_missing_method_is_a_hard_failure(base_bundle, tmp_path):
    bundle = _copy(base_bundle, tmp_path / "bundle")
    rankings = _read_rankings(bundle)
    rankings.loc[rankings["method"] == "bm25", "method"] = "ft_mpnet"
    _rewrite_rankings(bundle, rankings)
    with pytest.raises(Table4VerificationError, match="missing or extra ranking method"):
        verify_bundle(bundle, write_outputs=False)


def test_duplicate_candidate_is_a_hard_failure(base_bundle, tmp_path):
    bundle = _copy(base_bundle, tmp_path / "bundle")
    rankings = _read_rankings(bundle)
    index = rankings.index[rankings["method"] == "final_reranker"][0]
    rankings.at[index, "ranked_public_ids"] = ["2", "2"]
    rankings.at[index, "source_ranks"] = [1, 2]
    rankings.at[index, "scores"] = [1.0, 0.5]
    rankings.at[index, "returned_depth"] = 2
    _rewrite_rankings(bundle, rankings)
    with pytest.raises(Table4VerificationError, match="duplicate candidate"):
        verify_bundle(bundle, write_outputs=False)


def test_id_score_length_mismatch_is_a_hard_failure(base_bundle, tmp_path):
    bundle = _copy(base_bundle, tmp_path / "bundle")
    rankings = _read_rankings(bundle)
    index = rankings.index[rankings["method"] == "final_reranker"][0]
    rankings.at[index, "scores"] = [1.0, 0.5]
    _rewrite_rankings(bundle, rankings)
    with pytest.raises(Table4VerificationError, match="ID/score list-length mismatch"):
        verify_bundle(bundle, write_outputs=False)


def test_non_finite_score_is_a_hard_failure(base_bundle, tmp_path):
    bundle = _copy(base_bundle, tmp_path / "bundle")
    rankings = _read_rankings(bundle)
    index = rankings.index[rankings["method"] == "final_reranker"][0]
    rankings.at[index, "scores"] = [float("nan")]
    _rewrite_rankings(bundle, rankings)
    with pytest.raises(Table4VerificationError, match="non-finite score"):
        verify_bundle(bundle, write_outputs=False)


def test_invalid_depth_is_a_hard_failure(base_bundle, tmp_path):
    bundle = _copy(base_bundle, tmp_path / "bundle")
    rankings = _read_rankings(bundle)
    index = rankings.index[rankings["method"] == "python_cde_match_approx"][0]
    rankings.at[index, "source_ranks"] = [11]
    _rewrite_rankings(bundle, rankings)
    with pytest.raises(Table4VerificationError, match="invalid depth"):
        verify_bundle(bundle, write_outputs=False)


def test_expected_hit_count_mismatch_is_a_hard_failure(base_bundle, tmp_path):
    bundle = _copy(base_bundle, tmp_path / "bundle")
    manifest = _manifest(bundle)
    manifest["expected_results"]["test"]["final_reranker"]["hits_at_5"] -= 1
    _write_manifest(bundle, manifest)
    with pytest.raises(Table4VerificationError, match="manifest expected results"):
        verify_bundle(bundle, write_outputs=False)


def test_hgbc_numeric_tie_orders_2_before_10():
    ids, scores = canonical_hgbc_order(["10", "2"], [0.5, 0.5])
    assert ids == ["2", "10"]
    assert scores == [0.5, 0.5]


def test_hgbc_tie_result_is_invariant_to_incoming_order():
    assert canonical_hgbc_order(["10", "2"], [0.5, 0.5]) == \
        canonical_hgbc_order(["2", "10"], [0.5, 0.5])


def test_ft_medcpt_equal_scores_preserve_archived_order(base_bundle, tmp_path):
    bundle = _copy(base_bundle, tmp_path / "bundle")
    rankings = _read_rankings(bundle)
    candidates = rankings[(rankings["method"] == "ft_medcpt") &
                          (rankings["returned_depth"] == 0)]
    index = candidates.index[0]
    rankings.at[index, "ranked_public_ids"] = ["10", "2"]
    rankings.at[index, "source_ranks"] = [1, 2]
    rankings.at[index, "scores"] = [0.5, 0.5]
    rankings.at[index, "returned_depth"] = 2
    validate_ft_medcpt_order(["10", "2"], [0.5, 0.5])
    _rewrite_rankings(bundle, rankings)
    verify_bundle(bundle, write_outputs=False)
    reread = pq.read_table(bundle / RANKING_FILE).to_pandas()
    assert list(reread.at[index, "ranked_public_ids"]) == ["10", "2"]


def test_explicit_empty_ranking_is_valid_and_is_a_miss(base_bundle):
    rankings = _read_rankings(base_bundle)
    assert (rankings["returned_depth"] == 0).any()
    _, metrics = verify_bundle(base_bundle, write_outputs=False)
    assert metrics["test"]["bm25"]["hits_at_5"] == 1429


def test_missing_ranking_row_is_invalid(base_bundle, tmp_path):
    bundle = _copy(base_bundle, tmp_path / "bundle")
    rankings = _read_rankings(bundle).iloc[:-1].copy()
    _rewrite_rankings(bundle, rankings)
    with pytest.raises(Table4VerificationError, match="ranking row count mismatch"):
        verify_bundle(bundle, write_outputs=False)


def test_outputs_are_written_only_to_requested_external_directory(base_bundle, tmp_path):
    out = tmp_path / "reports"
    verify_bundle(base_bundle, out_dir=out, write_outputs=True)
    assert {path.name for path in out.iterdir()} == {
        "table4.csv", "metrics.json", "verification.json"}
