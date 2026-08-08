"""Load and verify the Section 4.3 winner manifest (fail-closed)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List

import yaml


class WinnerError(SystemExit):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


REQUIRED_FIELDS = (
    "model_id", "query_id", "query_name", "query_col",
    "cde_recipe", "cde_representation", "cde_format", "loss",
)


def load_winner_manifest(path: str, *, verify_source_hash: bool = True) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise WinnerError(f"[winners] manifest not found: {path}")
    with open(p) as f:
        man = yaml.safe_load(f)
    if not isinstance(man, dict) or man.get("schema_version") != 1:
        raise WinnerError(f"[winners] {path}: expected dict with schema_version: 1")
    winners = man.get("winners") or []
    if not winners:
        raise WinnerError(f"[winners] {path}: no winners")

    ids = [w.get("model_id") for w in winners]
    if len(ids) != len(set(ids)):
        raise WinnerError(f"[winners] duplicate model_id in manifest: {ids}")
    for w in winners:
        missing = [k for k in REQUIRED_FIELDS if not w.get(k)]
        if missing:
            raise WinnerError(f"[winners] winner {w.get('model_id')} missing fields: {missing}")

    if verify_source_hash:
        src = man.get("source_results_csv")
        exp = man.get("source_results_sha256")
        if not src or not exp:
            raise WinnerError("[winners] manifest missing source_results_csv/source_results_sha256")
        if Path(src).exists():
            got = sha256_file(src)
            if got != exp:
                raise WinnerError(
                    f"[winners] source CSV hash mismatch: {got} != {exp} ({src}); "
                    "manifest is stale relative to its source results -- rebuild it."
                )
        else:
            # Source may be a scratch artifact absent in some environments; record but do not
            # silently pass a canonical run. The caller decides via allow_missing_source.
            man["_source_present"] = False
    man["_manifest_sha256"] = sha256_file(path)
    man["_manifest_path"] = str(path)
    return man


def get_winner(man: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    for w in man.get("winners", []):
        if w.get("model_id") == model_id:
            return w
    have = [w.get("model_id") for w in man.get("winners", [])]
    raise WinnerError(f"[winners] no winner row for model_id={model_id!r}; manifest has {have}")


def list_model_ids(man: Dict[str, Any]) -> List[str]:
    return [w.get("model_id") for w in man.get("winners", [])]
