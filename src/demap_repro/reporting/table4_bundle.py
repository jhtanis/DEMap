"""Strict, offline validation of the frozen query-level Table 4 bundle.

The bundle is deliberately downstream of inference.  It contains only opaque
query identifiers, accepted gold public identifiers, and ranked public
identifiers.  Verification recomputes metrics from those query-level outputs;
it does not retrain models or rerun neural inference.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from demap_repro.ranking import canonical_score_order, public_id_sort_key
from demap_repro.utils.paths import repo_root

QUERY_FILE = "table4_queries.parquet"
RANKING_FILE = "table4_rankings.parquet"
MANIFEST_FILE = "table4_bundle_manifest.json"
BUNDLE_SCHEMA_VERSION = 1
QUERY_COLUMNS = ["dataset", "query_id", "gold_public_ids"]
RANKING_COLUMNS = [
    "dataset", "method", "query_id", "ranked_public_ids", "source_ranks", "scores",
    "returned_depth", "ranking_policy", "source_artifact_id",
]
SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")

QUERY_SCHEMA = pa.schema([
    ("dataset", pa.string()),
    ("query_id", pa.string()),
    ("gold_public_ids", pa.list_(pa.string())),
])
RANKING_SCHEMA = pa.schema([
    ("dataset", pa.string()),
    ("method", pa.string()),
    ("query_id", pa.string()),
    ("ranked_public_ids", pa.list_(pa.string())),
    ("source_ranks", pa.list_(pa.int16())),
    ("scores", pa.list_(pa.float64())),
    ("returned_depth", pa.int16()),
    ("ranking_policy", pa.string()),
    ("source_artifact_id", pa.string()),
])


class Table4VerificationError(ValueError):
    """A scientific-contract or bundle invariant failed."""


def default_contract_path() -> Path:
    return repo_root() / "configs/paper/table4_results_v1.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_ratio(hits: int, denominator: int, places: int = 3) -> str:
    quantum = Decimal(1).scaleb(-places)
    ratio = Decimal(hits) / Decimal(denominator)
    return format(ratio.quantize(quantum, rounding=ROUND_HALF_UP), f".{places}f")


def load_contract(path: Path | None = None) -> dict[str, Any]:
    contract_path = Path(path or default_contract_path())
    with contract_path.open(encoding="utf-8") as handle:
        contract = yaml.safe_load(handle)
    validate_contract(contract)
    return contract


def validate_contract(contract: dict[str, Any]) -> None:
    """Validate the complete, lossless 6 x 6 Table 4 scientific contract."""
    if contract.get("schema_version") != 1:
        raise Table4VerificationError("results contract schema_version must be 1")
    if contract.get("contract_id") != "table4_v1":
        raise Table4VerificationError("results contract_id must be table4_v1")
    metric = contract.get("metric", {})
    if metric.get("id") != "recall_at_5" or metric.get("cutoff") != 5:
        raise Table4VerificationError("results contract must define Recall@5")
    if metric.get("unreachable_gold_queries_excluded") is not True:
        raise Table4VerificationError("unreachable-gold exclusion must be explicit")
    if metric.get("multi_gold_semantics") != "any accepted gold public identifier is a hit":
        raise Table4VerificationError("multi-gold hit semantics are missing or changed")
    rendering = metric.get("rendering", {})
    if rendering != {"decimal_places": 3, "rounding": "ROUND_HALF_UP"}:
        raise Table4VerificationError("contract rendering must be 3dp ROUND_HALF_UP")

    methods = contract.get("methods", [])
    method_ids = [item.get("id") for item in methods]
    if len(method_ids) != 6 or len(set(method_ids)) != 6:
        raise Table4VerificationError("results contract must define six unique methods")
    datasets = contract.get("datasets", [])
    dataset_ids = [item.get("id") for item in datasets]
    if len(dataset_ids) != 6 or len(set(dataset_ids)) != 6:
        raise Table4VerificationError("results contract must define six unique datasets")
    total = sum(int(item.get("denominator", -1)) for item in datasets)
    if total != 7349 or int(contract.get("total_queries", -1)) != 7349:
        raise Table4VerificationError(f"dataset denominators must sum to 7349, got {total}")

    for dataset in datasets:
        denominator = int(dataset["denominator"])
        results = dataset.get("results", {})
        if set(results) != set(method_ids):
            raise Table4VerificationError(
                f"{dataset['id']} does not contain exactly all six method cells")
        for method_id in method_ids:
            cell = results[method_id]
            hits = int(cell["hits_at_5"])
            if not 0 <= hits <= denominator:
                raise Table4VerificationError(
                    f"invalid hit count {dataset['id']}/{method_id}: {hits}/{denominator}")
            derived = hits / denominator
            if not math.isclose(float(cell["recall_at_5"]), derived,
                                rel_tol=0.0, abs_tol=1e-15):
                raise Table4VerificationError(
                    f"full-precision ratio mismatch for {dataset['id']}/{method_id}")
            rendered = render_ratio(hits, denominator)
            if str(cell["manuscript_3dp"]) != rendered:
                raise Table4VerificationError(
                    f"3dp rendering mismatch for {dataset['id']}/{method_id}: "
                    f"{cell['manuscript_3dp']} != {rendered}")


def contract_methods(contract: dict[str, Any]) -> list[str]:
    return [item["id"] for item in contract["methods"]]


def contract_datasets(contract: dict[str, Any]) -> list[str]:
    return [item["id"] for item in contract["datasets"]]


def method_map(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in contract["methods"]}


def dataset_map(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in contract["datasets"]}


def _require_safe_token(value: Any, label: str, *, max_length: int = 160) -> str:
    text = str(value)
    if not text or len(text) > max_length or SAFE_TOKEN_RE.fullmatch(text) is None:
        raise Table4VerificationError(f"unsafe or non-opaque {label}: {text!r}")
    if "@" in text or "/" in text or "\\" in text or "://" in text:
        raise Table4VerificationError(f"unsafe or non-opaque {label}: {text!r}")
    return text


def _public_id_sort_key(public_id: str) -> tuple[int, int, str]:
    """Numeric IDs first numerically; deterministic lexical fallback otherwise."""
    return public_id_sort_key(public_id)


def sort_public_ids(values: Iterable[Any]) -> list[str]:
    result = [_require_safe_token(value, "CDE public identifier") for value in values]
    return sorted(set(result), key=_public_id_sort_key)


def canonical_hgbc_order(
    public_ids: Iterable[Any], scores: Iterable[Any]
) -> tuple[list[str], list[float]]:
    """Score descending, then numeric CDE public identifier ascending.

    A nonnumeric public identifier, if one is ever encountered, sorts after all
    numeric identifiers and uses a lexical fallback.  The released bundle is
    expected to contain numeric caDSR public identifiers.
    """
    ids = [_require_safe_token(value, "candidate public identifier") for value in public_ids]
    numeric_scores = [float(value) for value in scores]
    if len(ids) != len(numeric_scores):
        raise Table4VerificationError("candidate ID/score list-length mismatch")
    if len(ids) != len(set(ids)):
        raise Table4VerificationError("duplicate candidate public identifier")
    if any(not math.isfinite(value) for value in numeric_scores):
        raise Table4VerificationError("non-finite candidate score")
    try:
        return canonical_score_order(ids, numeric_scores)
    except ValueError as exc:
        raise Table4VerificationError(str(exc)) from exc


def validate_ft_medcpt_order(public_ids: list[str], scores: list[float]) -> None:
    """Validate archived score monotonicity without inventing a tie policy."""
    if len(public_ids) != len(scores):
        raise Table4VerificationError("candidate ID/score list-length mismatch")
    if any(not math.isfinite(value) for value in scores):
        raise Table4VerificationError("non-finite candidate score")
    if any(scores[index] < scores[index + 1] for index in range(len(scores) - 1)):
        raise Table4VerificationError("FT-MedCPT scores are not monotonic in archived order")


def query_set_identity_sha256(queries: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    ordered = queries.sort_values(["dataset", "query_id"], kind="stable")
    for row in ordered.itertuples(index=False):
        record = {
            "dataset": str(row.dataset),
            "query_id": str(row.query_id),
            "gold_public_ids": list(row.gold_public_ids),
        }
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _assert_external_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    repository = repo_root().resolve()
    try:
        resolved.relative_to(repository)
    except ValueError:
        return resolved
    raise Table4VerificationError(
        f"output directory must be outside the repository checkout: {resolved}")


def _load_manifest(bundle_dir: Path) -> dict[str, Any]:
    path = bundle_dir / MANIFEST_FILE
    if not path.is_file():
        raise Table4VerificationError(f"missing required bundle file: {MANIFEST_FILE}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise Table4VerificationError(f"invalid bundle manifest: {exc}") from exc
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise Table4VerificationError("bundle manifest schema_version mismatch")
    return manifest


def _verify_file_entry(bundle_dir: Path, name: str, entry: dict[str, Any]) -> Path:
    if Path(name).name != name or name not in {QUERY_FILE, RANKING_FILE}:
        raise Table4VerificationError(f"unexpected or unsafe manifest file name: {name!r}")
    path = bundle_dir / name
    if not path.is_file():
        raise Table4VerificationError(f"missing required bundle file: {name}")
    size = path.stat().st_size
    if int(entry.get("bytes", -1)) != size:
        raise Table4VerificationError(
            f"byte-size mismatch for {name}: manifest {entry.get('bytes')} != actual {size}")
    actual = sha256_file(path)
    if entry.get("sha256") != actual:
        raise Table4VerificationError(f"SHA-256 mismatch for {name}")
    return path


def _validate_queries(
    queries: pd.DataFrame, contract: dict[str, Any], manifest: dict[str, Any]
) -> dict[tuple[str, str], frozenset[str]]:
    if list(queries.columns) != QUERY_COLUMNS:
        raise Table4VerificationError(
            f"query schema mismatch: {list(queries.columns)} != {QUERY_COLUMNS}")
    if len(queries) != 7349:
        raise Table4VerificationError(f"query row count mismatch: {len(queries)} != 7349")
    dmap = dataset_map(contract)
    if set(queries["dataset"].astype(str)) != set(dmap):
        raise Table4VerificationError("query dataset identities do not match the contract")

    expected_counts = {dataset: int(item["denominator"]) for dataset, item in dmap.items()}
    actual_counts = queries.groupby("dataset", observed=True).size().to_dict()
    if actual_counts != expected_counts:
        raise Table4VerificationError(
            f"query dataset membership/count mismatch: {actual_counts} != {expected_counts}")

    gold_by_query: dict[tuple[str, str], frozenset[str]] = {}
    for row in queries.itertuples(index=False):
        dataset = str(row.dataset)
        query_id = _require_safe_token(row.query_id, "query_id")
        key = (dataset, query_id)
        if key in gold_by_query:
            raise Table4VerificationError(f"duplicate query identity: {key}")
        gold = [_require_safe_token(value, "gold public identifier")
                for value in list(row.gold_public_ids)]
        if not gold:
            raise Table4VerificationError(f"query has no accepted gold identifiers: {key}")
        if gold != sort_public_ids(gold):
            raise Table4VerificationError(f"gold identifiers are not sorted and duplicate-free: {key}")
        gold_by_query[key] = frozenset(gold)

    identity = query_set_identity_sha256(queries)
    if manifest.get("query_set_identity_sha256") != identity:
        raise Table4VerificationError("query-set identity hash mismatch")
    return gold_by_query


def _coerce_scores(value: Any, key: tuple[str, str, str]) -> list[float] | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        result = [float(item) for item in list(value)]
    except (TypeError, ValueError) as exc:
        raise Table4VerificationError(f"invalid scores list for {key}") from exc
    if any(not math.isfinite(item) for item in result):
        raise Table4VerificationError(f"non-finite score for {key}")
    return result


def _validate_rankings(
    rankings: pd.DataFrame,
    contract: dict[str, Any],
    manifest: dict[str, Any],
    gold_by_query: dict[tuple[str, str], frozenset[str]],
) -> dict[tuple[str, str, str], tuple[list[str], list[int]]]:
    if list(rankings.columns) != RANKING_COLUMNS:
        raise Table4VerificationError(
            f"ranking schema mismatch: {list(rankings.columns)} != {RANKING_COLUMNS}")
    if len(rankings) != 44094:
        raise Table4VerificationError(
            f"ranking row count mismatch: {len(rankings)} != 44094")

    methods = contract_methods(contract)
    method_specs = method_map(contract)
    datasets = contract_datasets(contract)
    if set(rankings["dataset"].astype(str)) != set(datasets):
        raise Table4VerificationError("ranking dataset identities do not match the contract")
    if set(rankings["method"].astype(str)) != set(methods):
        raise Table4VerificationError("missing or extra ranking method")

    manifest_depths = manifest.get("method_maximum_depths")
    expected_depths = {method: int(method_specs[method]["maximum_depth"]) for method in methods}
    if manifest_depths != expected_depths:
        raise Table4VerificationError("manifest method maximum depths do not match contract")
    manifest_policies = manifest.get("ranking_policies")
    expected_policies = {method: method_specs[method]["ranking_policy"] for method in methods}
    if manifest_policies != expected_policies:
        raise Table4VerificationError("manifest ranking policies do not match contract")

    source_artifacts = manifest.get("source_artifacts", [])
    source_ids: set[str] = set()
    for item in source_artifacts:
        if set(item) != {"artifact_id", "sha256", "bytes", "role"}:
            raise Table4VerificationError("invalid source artifact manifest entry")
        artifact_id = _require_safe_token(item["artifact_id"], "source artifact ID")
        _require_safe_token(item["role"], "source artifact role")
        if not re.fullmatch(r"[0-9a-f]{64}", str(item["sha256"])):
            raise Table4VerificationError(f"invalid source artifact SHA-256: {artifact_id}")
        if int(item["bytes"]) < 0 or artifact_id in source_ids:
            raise Table4VerificationError(f"invalid/duplicate source artifact: {artifact_id}")
        source_ids.add(artifact_id)
    if not source_ids:
        raise Table4VerificationError("manifest source artifact identities are missing")

    actual: dict[tuple[str, str, str], tuple[list[str], list[int]]] = {}
    for row in rankings.itertuples(index=False):
        dataset, method = str(row.dataset), str(row.method)
        query_id = _require_safe_token(row.query_id, "query_id")
        key = (dataset, method, query_id)
        if key in actual:
            raise Table4VerificationError(f"duplicate ranking row: {key}")
        if (dataset, query_id) not in gold_by_query:
            raise Table4VerificationError(f"ranking references an unknown query: {key}")
        if method not in method_specs:
            raise Table4VerificationError(f"unknown ranking method: {method}")
        ids = [_require_safe_token(value, "candidate public identifier")
               for value in list(row.ranked_public_ids)]
        if len(ids) != len(set(ids)):
            raise Table4VerificationError(f"duplicate candidate public identifier for {key}")
        returned_depth = int(row.returned_depth)
        if returned_depth != len(ids):
            raise Table4VerificationError(f"returned_depth/list-length mismatch for {key}")
        ranks = [int(value) for value in list(row.source_ranks)]
        if len(ranks) != len(ids):
            raise Table4VerificationError(f"candidate ID/source-rank list-length mismatch for {key}")
        maximum_depth = int(method_specs[method]["maximum_depth"])
        if returned_depth < 0 or returned_depth > maximum_depth or any(
            rank < 1 or rank > maximum_depth for rank in ranks
        ):
            raise Table4VerificationError(f"invalid depth for {key}: {returned_depth}")
        if ranks != sorted(set(ranks)):
            raise Table4VerificationError(f"source ranks are not strictly increasing for {key}")
        policy = str(row.ranking_policy)
        if policy != method_specs[method]["ranking_policy"]:
            raise Table4VerificationError(f"ranking-policy violation for {key}")
        source_id = _require_safe_token(row.source_artifact_id, "source_artifact_id")
        if source_id not in source_ids:
            raise Table4VerificationError(f"unknown source_artifact_id for {key}: {source_id}")

        scores = _coerce_scores(row.scores, key)
        if scores is not None and len(scores) != len(ids):
            raise Table4VerificationError(f"candidate ID/score list-length mismatch for {key}")
        if method == "final_reranker":
            if scores is None:
                raise Table4VerificationError(f"final reranker scores are required for {key}")
            canonical_ids, canonical_scores = canonical_hgbc_order(ids, scores)
            if ids != canonical_ids or scores != canonical_scores:
                raise Table4VerificationError(f"final-reranker ranking-policy violation for {key}")
            if ranks != list(range(1, len(ids) + 1)):
                raise Table4VerificationError(f"final-reranker ranks are not contiguous for {key}")
        elif method == "ft_medcpt":
            if scores is None:
                raise Table4VerificationError(f"FT-MedCPT scores are required for {key}")
            # Equal-score order is intentionally preserved.  The HGBC numeric-ID
            # tie rule must not leak into this archived cross-encoder ordering.
            validate_ft_medcpt_order(ids, scores)
            if ranks != list(range(1, len(ids) + 1)):
                raise Table4VerificationError(f"FT-MedCPT ranks are not contiguous for {key}")
        actual[key] = (ids, ranks)

    expected = {
        (dataset, method, query_id)
        for dataset, query_id in gold_by_query
        for method in methods
    }
    actual_keys = set(actual)
    if actual_keys != expected:
        missing = sorted(expected - actual_keys)[:3]
        extra = sorted(actual_keys - expected)[:3]
        raise Table4VerificationError(
            f"ranking query/method membership mismatch; missing={missing}, extra={extra}")
    return actual


def compute_metrics(
    contract: dict[str, Any],
    gold_by_query: dict[tuple[str, str], frozenset[str]],
    rankings: dict[tuple[str, str, str], tuple[list[str], list[int]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in contract["datasets"]:
        dataset_id = dataset["id"]
        query_ids = sorted(query_id for ds, query_id in gold_by_query if ds == dataset_id)
        result[dataset_id] = {}
        for method in contract["methods"]:
            method_id = method["id"]
            hits = {1: 0, 5: 0, 10: 0}
            reciprocal_sum = 0.0
            for query_id in query_ids:
                gold = gold_by_query[(dataset_id, query_id)]
                ranked, source_ranks = rankings[(dataset_id, method_id, query_id)]
                first = next((rank for public_id, rank in zip(ranked, source_ranks)
                              if public_id in gold), None)
                if first is not None:
                    reciprocal_sum += 1.0 / first
                    for cutoff in hits:
                        if first <= cutoff:
                            hits[cutoff] += 1
            denominator = len(query_ids)
            result[dataset_id][method_id] = {
                "denominator": denominator,
                "hits_at_1": hits[1],
                "recall_at_1": hits[1] / denominator,
                "hits_at_5": hits[5],
                "recall_at_5": hits[5] / denominator,
                "hits_at_10": hits[10],
                "recall_at_10": hits[10] / denominator,
                "mrr_at_available_depth": reciprocal_sum / denominator,
                "available_depth": int(method["maximum_depth"]),
                "manuscript_3dp": render_ratio(hits[5], denominator),
            }
    return result


def _compare_results(
    contract: dict[str, Any], metrics: dict[str, dict[str, dict[str, Any]]]
) -> None:
    for dataset in contract["datasets"]:
        dataset_id = dataset["id"]
        denominator = int(dataset["denominator"])
        for method_id, expected in dataset["results"].items():
            observed = metrics[dataset_id][method_id]
            if observed["denominator"] != denominator:
                raise Table4VerificationError(
                    f"denominator mismatch for {dataset_id}/{method_id}")
            if observed["hits_at_5"] != int(expected["hits_at_5"]):
                raise Table4VerificationError(
                    f"R@5 hit-count mismatch for {dataset_id}/{method_id}: "
                    f"{observed['hits_at_5']} != {expected['hits_at_5']}")
            if not math.isclose(observed["recall_at_5"],
                                float(expected["recall_at_5"]),
                                rel_tol=0.0, abs_tol=1e-15):
                raise Table4VerificationError(
                    f"full-precision R@5 mismatch for {dataset_id}/{method_id}")
            if observed["manuscript_3dp"] != str(expected["manuscript_3dp"]):
                raise Table4VerificationError(
                    f"manuscript rendering mismatch for {dataset_id}/{method_id}")


def _write_outputs(
    out_dir: Path,
    contract: dict[str, Any],
    manifest: dict[str, Any],
    metrics: dict[str, dict[str, dict[str, Any]]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = contract["methods"]
    with (out_dir / "table4.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset"] + [method["display_name"] for method in methods])
        for dataset in contract["datasets"]:
            dataset_id = dataset["id"]
            writer.writerow([dataset["display_name"]] + [
                metrics[dataset_id][method["id"]]["manuscript_3dp"] for method in methods
            ])
    (out_dir / "metrics.json").write_text(
        json.dumps({"contract_id": contract["contract_id"], "metrics": metrics},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verification = {
        "status": "PASS",
        "contract_id": contract["contract_id"],
        "bundle_id": manifest["bundle_id"],
        "checks": {
            "central_result_checks_skipped": 0,
            "table4_cells_verified": 36,
            "query_rows": 7349,
            "ranking_rows": 44094,
            "file_checksums_verified": 2,
        },
    }
    (out_dir / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_bundle(
    bundle: Path,
    *,
    out_dir: Path | None = None,
    contract_path: Path | None = None,
    write_outputs: bool = True,
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    """Strictly verify a local bundle and optionally write the three reports."""
    bundle_dir = Path(bundle).expanduser().resolve()
    if not bundle_dir.is_dir():
        raise Table4VerificationError(f"bundle directory does not exist: {bundle_dir}")
    contract_file = Path(contract_path or default_contract_path()).resolve()
    contract = load_contract(contract_file)
    manifest = _load_manifest(bundle_dir)
    if manifest.get("bundle_id") in (None, ""):
        raise Table4VerificationError("bundle_id is missing")
    contract_entry = manifest.get("results_contract", {})
    if contract_entry.get("contract_id") != contract["contract_id"]:
        raise Table4VerificationError("results-contract ID mismatch")
    if contract_entry.get("sha256") != sha256_file(contract_file):
        raise Table4VerificationError("results-contract SHA-256 mismatch")
    if int(manifest.get("query_rows", -1)) != 7349:
        raise Table4VerificationError("manifest query row count mismatch")
    if int(manifest.get("ranking_rows", -1)) != 44094:
        raise Table4VerificationError("manifest ranking row count mismatch")
    if manifest.get("dataset_query_counts") != {
        item["id"]: int(item["denominator"]) for item in contract["datasets"]
    }:
        raise Table4VerificationError("manifest dataset query counts mismatch")
    if manifest.get("method_row_counts") != {
        item["id"]: 7349 for item in contract["methods"]
    }:
        raise Table4VerificationError("manifest method row counts mismatch")

    files = manifest.get("files", {})
    if set(files) != {QUERY_FILE, RANKING_FILE}:
        raise Table4VerificationError("manifest must list exactly the two Parquet files")
    query_path = _verify_file_entry(bundle_dir, QUERY_FILE, files[QUERY_FILE])
    ranking_path = _verify_file_entry(bundle_dir, RANKING_FILE, files[RANKING_FILE])
    queries = pq.read_table(query_path).to_pandas()
    rankings_df = pq.read_table(ranking_path).to_pandas()
    if int(files[QUERY_FILE].get("rows", -1)) != len(queries):
        raise Table4VerificationError("manifest/file query row count mismatch")
    if int(files[RANKING_FILE].get("rows", -1)) != len(rankings_df):
        raise Table4VerificationError("manifest/file ranking row count mismatch")
    gold_by_query = _validate_queries(queries, contract, manifest)
    rankings = _validate_rankings(rankings_df, contract, manifest, gold_by_query)
    metrics = compute_metrics(contract, gold_by_query, rankings)
    _compare_results(contract, metrics)
    if manifest.get("expected_results") != {
        item["id"]: item["results"] for item in contract["datasets"]
    }:
        raise Table4VerificationError("manifest expected results do not match contract")

    if write_outputs:
        if out_dir is None:
            raise Table4VerificationError("an explicit output directory is required")
        external_out = _assert_external_output(Path(out_dir))
        _write_outputs(external_out, contract, manifest, metrics)
    return manifest, metrics


def write_query_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    table = pa.Table.from_pylist(rows, schema=QUERY_SCHEMA)
    pq.write_table(table, path, compression="zstd", version="2.6")


def write_ranking_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    table = pa.Table.from_pylist(rows, schema=RANKING_SCHEMA)
    pq.write_table(table, path, compression="zstd", version="2.6")
