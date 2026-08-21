"""Canonical evaluation dataset registry loader.

Single source of truth for the reporting eval set (test, cctg, oid_alt, cdash, gdc_combined,
cimac_v2). All model families and reporting code should read split names / paths / display
names from here instead of hard-coding lists.

Registry file: ``configs/paper/eval_datasets_v1.yaml``
Materialized reachable-filtered parquets: ``data/processed/eval_canonical/<name>.parquet``

Two things this module is deliberately strict about.

**The default resolves.** It used to name ``configs/evaluation/canonical_eval_datasets.yaml``,
a path that does not exist in this repository, relative to the current working
directory — so every no-argument call raised ``FileNotFoundError`` and no call at
all worked from outside the repository root. The default is now the shipped
registry, resolved against the source tree.

**Only the paper population counts.** ``eval_canonical_v2`` is the Rule-E
train-decontaminated derivative retained for the S6.1 leakage sensitivity
analysis (Table S5). It is a paper artifact, but it is *not* the reporting
population, and reading it through this loader would silently swap the
denominators in Table 4. :func:`assert_paper_evaluation_population` rejects it,
along with the superseded ``splits_v3_cdisc`` scheme and any dataset name outside
the six the paper reports.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml

from demap_repro.utils.paths import repo_root

#: The shipped registry. Resolved against the source tree, not the working
#: directory, so a caller outside the checkout still gets a path that exists.
REGISTRY_PATH = repo_root() / 'configs' / 'paper' / 'eval_datasets_v1.yaml'

#: The six evaluation datasets the paper reports, in report order.
PAPER_EVAL_DATASETS = ('test', 'cctg', 'oid_alt', 'cdash', 'gdc_combined', 'cimac_v2')

#: Query-level denominators stated in the manuscript (Section 2.1 / Table 4).
PAPER_EVAL_QUERY_COUNTS = {
    'test': 3959, 'cctg': 1097, 'oid_alt': 1766,
    'cdash': 324, 'gdc_combined': 72, 'cimac_v2': 131,
}

#: Path fragments that name a *different* evaluation population. Any of these
#: appearing in a canonical entry means the wrong registry has been loaded.
FORBIDDEN_PATH_FRAGMENTS = ('eval_canonical_v2', 'eval_paper_v1', 'splits_v3_cdisc')


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


def assert_paper_evaluation_population(path: str | Path = REGISTRY_PATH) -> None:
    """Raise unless the registry is the paper's own evaluation population.

    Checks three things, each of which has a plausible way of going wrong:
    the six dataset names, the paper's query-level denominators, and that no
    entry points into a non-paper evaluation tree.
    """
    entries = _canonical_entries(path)
    names = tuple(str(e['name']) for e in entries)
    if names != PAPER_EVAL_DATASETS:
        raise ValueError(
            f'canonical registry {path} declares {names}, '
            f'not the paper population {PAPER_EVAL_DATASETS}')

    for entry in entries:
        entry_path = str(entry['path'])
        for fragment in FORBIDDEN_PATH_FRAGMENTS:
            if fragment in entry_path:
                raise ValueError(
                    f"canonical dataset '{entry['name']}' resolves to {entry_path}, "
                    f"which is not the paper evaluation population ('{fragment}')")

    counts = canonical_count_map(path)
    wrong = {n: counts[n]['reachable_rows'] for n in names
             if counts[n]['reachable_rows'] < PAPER_EVAL_QUERY_COUNTS[n]}
    if wrong:
        raise ValueError(
            f'canonical registry {path} declares fewer reachable rows than the '
            f'paper reports queries for: {wrong}')


__all__ = [
    'REGISTRY_PATH',
    'PAPER_EVAL_DATASETS',
    'PAPER_EVAL_QUERY_COUNTS',
    'FORBIDDEN_PATH_FRAGMENTS',
    'assert_paper_evaluation_population',
    'load_registry',
    'canonical_split_names',
    'canonical_path_map',
    'canonical_display_map',
    'canonical_count_map',
    'materialized_dir',
    'production_catalog',
]
