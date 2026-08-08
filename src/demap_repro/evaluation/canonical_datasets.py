"""Canonical evaluation dataset registry loader.

Single source of truth for the reporting eval set (test, cctg, oid_alt, cdash, gdc_combined,
cimac_v2). All model families and reporting code should read split names / paths / display
names from here instead of hard-coding lists.

Registry file: configs/evaluation/canonical_eval_datasets.yaml
Materialized reachable-filtered parquets: data/processed/eval_canonical/<name>.parquet
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml

REGISTRY_PATH = Path('configs/evaluation/canonical_eval_datasets.yaml')


@lru_cache(maxsize=None)
def load_registry(path: str | Path = REGISTRY_PATH) -> Dict[str, Any]:
    """Load and cache the canonical eval registry YAML."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Canonical eval registry not found: {p}')
    with p.open('r', encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def _canonical_entries(path: str | Path = REGISTRY_PATH) -> List[Dict[str, Any]]:
    return list(load_registry(path).get('canonical') or [])


def canonical_split_names(path: str | Path = REGISTRY_PATH) -> List[str]:
    """Ordered canonical dataset names, e.g. ['test','cctg','oid_alt','cdash','gdc_combined','cimac_v2']."""
    return [str(e['name']) for e in _canonical_entries(path)]


def canonical_path_map(path: str | Path = REGISTRY_PATH) -> Dict[str, str]:
    """name -> materialized parquet path."""
    return {str(e['name']): str(e['path']) for e in _canonical_entries(path)}


def canonical_display_map(path: str | Path = REGISTRY_PATH) -> Dict[str, str]:
    """name -> human display name (falls back to the raw name)."""
    disp = dict(load_registry(path).get('display_names') or {})
    return {n: str(disp.get(n, n)) for n in canonical_split_names(path)}


def canonical_count_map(path: str | Path = REGISTRY_PATH) -> Dict[str, Dict[str, int]]:
    """name -> {original_rows, reachable_rows, dropped_rows}."""
    out: Dict[str, Dict[str, int]] = {}
    for e in _canonical_entries(path):
        out[str(e['name'])] = {
            'original_rows': int(e.get('original_rows', -1)),
            'reachable_rows': int(e.get('reachable_rows', -1)),
            'dropped_rows': int(e.get('dropped_rows', -1)),
        }
    return out


def materialized_dir(path: str | Path = REGISTRY_PATH) -> str:
    return str(load_registry(path).get('materialized_dir', 'data/processed/eval_canonical'))


def production_catalog(path: str | Path = REGISTRY_PATH) -> str:
    return str(load_registry(path)['production_catalog'])


__all__ = [
    'REGISTRY_PATH',
    'load_registry',
    'canonical_split_names',
    'canonical_path_map',
    'canonical_display_map',
    'canonical_count_map',
    'materialized_dir',
    'production_catalog',
]
