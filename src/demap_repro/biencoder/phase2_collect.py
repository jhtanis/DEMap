from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd
import yaml

from demap_repro.biencoder.constants import (
    PAPER_PHASE2_BATCH_SIZE,
    PAPER_PHASE2_LR_BY_PHASE1,
    PAPER_PHASE2_MINING_STRATEGIES,
    PAPER_PHASE2_TEMPERATURE_BY_PHASE1,
)
from demap_repro.biencoder.phase1_screening import extract_placeholder_strategy
from demap_repro.utils.config import load_config


DEFAULT_HARDNEG_STAGE_TAG = 'paper_step12_hardneg'
DEFAULT_PHASE2_STAGE_TAG = 'paper_step13_finetune_phase2_bf16'
DEFAULT_PHASE2_SEEDS = [0, 1]
DEFAULT_PHASE2_BATCH_SIZE = int(PAPER_PHASE2_BATCH_SIZE)
DEFAULT_PHASE2_MINING_STRATEGIES = list(PAPER_PHASE2_MINING_STRATEGIES)

PHASE2_TEMP_BY_PHASE1 = {float(k): [float(x) for x in v] for k, v in PAPER_PHASE2_TEMPERATURE_BY_PHASE1.items()}
PHASE2_LR_BY_PHASE1 = {float(k): [float(x) for x in v] for k, v in PAPER_PHASE2_LR_BY_PHASE1.items()}

PHASE2_GROUP_COLS = [
    'model_name',
    'base_model_id',
    'query_variant',
    'recipe',
    'cde_format',
    'rerank_mode',
    'hybrid_alpha',
    'placeholder_strategy',
    'loss',
    'lr',
    'batch_size',
    'temperature',
    'epochs',
    'mining_strategy',
    'strategy',
    'nneg',
    'hard_band',
    'semihard_band',
    'semihard_tight_band',
    'curriculum_bands',
    'hardcurr_bands',
]


def slug(s: object) -> str:
    return str(s).replace('/', '__').replace(' ', '_').replace(':', '_')


def load_yaml_block(path: str | Path, block_name: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    all_cfg = load_config(path)
    cfg = all_cfg.get(block_name, all_cfg)
    if not isinstance(cfg, dict):
        raise SystemExit(f'Could not resolve {block_name} block from template config')
    return all_cfg, cfg


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def derive_phase2_batch_size(phase1_batch_size: int, nneg: int) -> int:
    total_phase1_texts = 2.0 * float(int(phase1_batch_size))
    total_phase2_texts_per_anchor = float(int(nneg) + 2)
    return max(1, int(math.ceil(total_phase1_texts / total_phase2_texts_per_anchor)))


def _closest_anchor(value: float, anchors: Sequence[float]) -> float:
    vals = [float(x) for x in anchors]
    if not vals:
        return float(value)
    return min(vals, key=lambda x: (abs(float(value) - x), x))


def neighbor_values(center: float, grid: Sequence[float]) -> List[float]:
    vals = sorted({float(x) for x in grid})
    if not vals:
        return [float(center)]
    if float(center) in vals:
        idx = vals.index(float(center))
    else:
        vals.append(float(center))
        vals = sorted(set(vals))
        idx = vals.index(float(center))
    keep = [vals[idx]]
    if idx - 1 >= 0:
        keep.insert(0, vals[idx - 1])
    if idx + 1 < len(vals):
        keep.append(vals[idx + 1])
    return keep


def derive_phase2_lr_grid(phase1_lr: float, anchors: Sequence[float]) -> List[float]:
    anchor = _closest_anchor(float(phase1_lr), anchors)
    grid = PHASE2_LR_BY_PHASE1.get(anchor)
    if grid is not None:
        return [float(x) for x in grid]
    return [float(x) for x in neighbor_values(float(phase1_lr), anchors) if float(x) <= float(phase1_lr)] or [float(phase1_lr)]


def derive_phase2_temperature_grid(phase1_temp: float, anchors: Sequence[float]) -> List[float]:
    anchor = _closest_anchor(float(phase1_temp), anchors)
    grid = PHASE2_TEMP_BY_PHASE1.get(anchor)
    if grid is not None:
        return [float(x) for x in grid]
    return [float(x) for x in neighbor_values(float(phase1_temp), anchors)]


def _normalize_band_value(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (tuple, list)):
        if len(value) == 2 and not any(isinstance(x, (tuple, list, dict)) for x in value):
            try:
                return f'{int(value[0])}-{int(value[1])}'
            except Exception:
                return f'{value[0]}-{value[1]}'
        parts = [_normalize_band_value(x) for x in value]
        return '__'.join([p for p in parts if p])
    return str(value).strip()


def mining_strategy_label_from_parts(
    *,
    strategy: Any,
    hard_band: Any = None,
    semihard_band: Any = None,
    semihard_tight_band: Any = None,
    curriculum_bands: Any = None,
    hardcurr_bands: Any = None,
) -> str:
    strat = str(strategy or '').strip().lower()
    hb = _normalize_band_value(hard_band)
    shb = _normalize_band_value(semihard_band)
    stb = _normalize_band_value(semihard_tight_band)
    cb = _normalize_band_value(curriculum_bands)
    hcb = _normalize_band_value(hardcurr_bands)

    if strat == 'none':
        return 'none'
    if strat == 'hard':
        return 'hard_top25' if hb == '1-25' else f"hard_{hb.replace('-', '_')}"
    if strat == 'semihard':
        return 'semihard_1_50' if shb == '1-50' else f"semihard_{shb.replace('-', '_')}"
    if strat == 'semihard_tight':
        return 'semihard_tight_10_50' if stb == '10-50' else f"semihard_tight_{stb.replace('-', '_')}"
    if strat == 'curr':
        return 'curr_50_200__25_100__1_25' if cb == '50-200__25-100__1-25' else f"curr_{cb.replace('-', '_')}"
    if strat == 'hardcurr':
        return 'hardcurr_1_50__1_25__1_15' if hcb == '1-50__1-25__1-15' else f"hardcurr_{hcb.replace('-', '_')}"
    return strat or 'unknown'


def resolve_mining_strategy(name: str) -> Dict[str, Any]:
    key = str(name or '').strip().lower()
    aliases = {
        'none': 'none',
        'hard': 'hard_top25',
        'hard_top25': 'hard_top25',
        'hard_1_25': 'hard_top25',
        'semihard': 'semihard_1_50',
        'semihard_band': 'semihard_1_50',
        'semihard_1_50': 'semihard_1_50',
        'semihard_25_100': 'semihard_1_50',
        'hardcurr': 'hardcurr_1_50__1_25__1_15',
        'hardcurr_1_50__1_25__1_15': 'hardcurr_1_50__1_25__1_15',
    }
    canonical = aliases.get(key)
    if canonical is None:
        raise SystemExit(
            f'Unknown mining strategy: {name!r}. Supported values: {", ".join(DEFAULT_PHASE2_MINING_STRATEGIES)}'
        )
    if canonical == 'none':
        return {
            'name': canonical,
            'strategy': 'none',
            'requires_mined': False,
            'hard_band': None,
            'semihard_band': None,
            'semihard_tight_band': None,
            'curriculum_bands': None,
            'hardcurr_bands': None,
            'max_rank': 0,
        }
    if canonical == 'hard_top25':
        return {
            'name': canonical,
            'strategy': 'hard',
            'requires_mined': True,
            'hard_band': '1-25',
            'semihard_band': None,
            'semihard_tight_band': None,
            'curriculum_bands': None,
            'hardcurr_bands': None,
            'max_rank': 25,
        }
    if canonical == 'semihard_1_50':
        return {
            'name': canonical,
            'strategy': 'semihard',
            'requires_mined': True,
            'hard_band': None,
            'semihard_band': '1-50',
            'semihard_tight_band': None,
            'curriculum_bands': None,
            'hardcurr_bands': None,
            'max_rank': 50,
        }
    return {
        'name': 'hardcurr_1_50__1_25__1_15',
        'strategy': 'hardcurr',
        'requires_mined': True,
        'hard_band': None,
        'semihard_band': None,
        'semihard_tight_band': None,
        'curriculum_bands': None,
        'hardcurr_bands': ['1-50', '1-25', '1-15'],
        'max_rank': 50,
    }


def parse_mining_strategies(value: str | Sequence[str] | None) -> List[Dict[str, Any]]:
    if value is None:
        names = list(DEFAULT_PHASE2_MINING_STRATEGIES)
    elif isinstance(value, str):
        names = [x.strip() for x in value.split(',') if x.strip()]
    else:
        names = [str(x).strip() for x in value if str(x).strip()]
    presets: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for name in names:
        preset = resolve_mining_strategy(name)
        if preset['name'] in seen:
            continue
        presets.append(preset)
        seen.add(str(preset['name']))
    return presets


def max_required_rank(presets: Sequence[Mapping[str, Any]]) -> int:
    vals = [int(p.get('max_rank', 0)) for p in presets]
    return max(vals) if vals else 0


def collect_phase2_runs(stage_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rc_path in sorted(stage_root.glob('*/runs/*/run_config.json')):
        run_dir = rc_path.parent
        metrics_path = run_dir / 'metrics.json'
        if not metrics_path.exists():
            continue
        try:
            rc = json.loads(rc_path.read_text(encoding='utf-8'))
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        except Exception:
            continue
        # Selection metric: prefer the canonical `val_dev` split (recorded in run_config.data.
        # validation_split), fall back to legacy `val` for older runs.
        _mbs = metrics.get('metrics_by_split') or {}
        _val_split = str(((rc.get('data') or {}).get('validation_split'))
                         or ((rc.get('eval') or {}).get('validation_split')) or '')
        val = (_mbs.get(_val_split) if _val_split else None) or _mbs.get('val_dev') or _mbs.get('val') or {}
        rep = rc.get('representation') or {}
        train = rc.get('train') or {}
        hardneg = rc.get('hardneg') or {}
        model_name = str(rc.get('model_name') or rc.get('base_model_id') or rc.get('base_model') or '')
        base_model_id = str(rc.get('base_model_id') or rc.get('base_model') or model_name)
        hard_band = _normalize_band_value(hardneg.get('hard_band'))
        semihard_band = _normalize_band_value(hardneg.get('effective_semihard_band') or hardneg.get('semihard_band'))
        semihard_tight_band = _normalize_band_value(hardneg.get('semihard_tight_band'))
        curriculum_bands = _normalize_band_value(hardneg.get('curriculum_bands'))
        hardcurr_bands = _normalize_band_value(hardneg.get('effective_curriculum_bands') or hardneg.get('hardcurr_bands'))
        strategy = str(hardneg.get('strategy') or '')
        rows.append(
            {
                'run_dir': str(run_dir),
                'model_path': str(run_dir / 'model'),
                'model_name': model_name,
                'base_model_id': base_model_id,
                'query_variant': str(rep.get('query_variant') or rc.get('query_variant') or ''),
                'recipe': str(rep.get('recipe') or rc.get('recipe') or ''),
                'cde_format': str(rep.get('cde_format') or rc.get('cde_format') or ''),
                'rerank_mode': str(rep.get('rerank_mode') or rc.get('rerank_mode') or 'R0'),
                'hybrid_alpha': float(rep.get('hybrid_alpha') or rc.get('hybrid_alpha') or 0.5),
                'placeholder_strategy': extract_placeholder_strategy(rc),
                'seed': int(train.get('seed', -1)),
                'loss': str(train.get('loss') or ''),
                'lr': float(train.get('lr') or float('nan')),
                'batch_size': int(train.get('batch_size', -1)),
                'temperature': float(train.get('temperature') or float('nan')),
                'epochs': int(train.get('epochs', -1)),
                'mining_strategy': mining_strategy_label_from_parts(
                    strategy=strategy,
                    hard_band=hard_band,
                    semihard_band=semihard_band,
                    semihard_tight_band=semihard_tight_band,
                    curriculum_bands=curriculum_bands,
                    hardcurr_bands=hardcurr_bands,
                ),
                'strategy': strategy,
                'nneg': int(hardneg.get('effective_nneg', hardneg.get('nneg', 0))),
                'hard_band': hard_band,
                'semihard_band': semihard_band,
                'semihard_tight_band': semihard_tight_band,
                'curriculum_bands': curriculum_bands,
                'hardcurr_bands': hardcurr_bands,
                'val_recall5': float(val.get('recall@5', float('nan'))),
                'val_mrr100': float(val.get('mrr@100', float('nan'))),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    'DEFAULT_HARDNEG_STAGE_TAG',
    'DEFAULT_PHASE2_BATCH_SIZE',
    'DEFAULT_PHASE2_MINING_STRATEGIES',
    'DEFAULT_PHASE2_SEEDS',
    'DEFAULT_PHASE2_STAGE_TAG',
    'PHASE2_GROUP_COLS',
    'collect_phase2_runs',
    'derive_phase2_batch_size',
    'derive_phase2_lr_grid',
    'derive_phase2_temperature_grid',
    'load_yaml_block',
    'max_required_rank',
    'mining_strategy_label_from_parts',
    'neighbor_values',
    'parse_mining_strategies',
    'resolve_mining_strategy',
    'slug',
    'write_yaml',
]
