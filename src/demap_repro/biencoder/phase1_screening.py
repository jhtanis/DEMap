from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd
import yaml

from demap_repro.biencoder.constants import PAPER_PHASE1_BATCH_SIZES
from demap_repro.biencoder.selection_rules import order_candidates_with_tie_margin
from demap_repro.utils.config import load_config


DEFAULT_SCREEN_STAGE_TAG = 'paper_step11_4_screen_joint'
DEFAULT_SCREEN_REP_STAGE_TAG = DEFAULT_SCREEN_STAGE_TAG
DEFAULT_SCREEN_LOSS_STAGE_TAG = 'paper_step11_4_screen_loss'
DEFAULT_SCREEN_LR = 5e-5
DEFAULT_SCREEN_BATCH_SIZE = int(PAPER_PHASE1_BATCH_SIZES[0])
DEFAULT_SCREEN_TEMPERATURE = 0.05
DEFAULT_SCREEN_EPOCHS = 3
DEFAULT_SCREEN_SEEDS = [0, 1]
DEFAULT_SCREEN_BASELINE_LOSS = 'mnrl'
DEFAULT_SCREEN_LOSSES = ['mnrl', 'symmetric_mnrl']
DEFAULT_PLACEHOLDER_STRATEGY = 'placeholder'
DEFAULT_SHORT_NAME_PLACEHOLDER = '<MISSING_SHORT_NAME>'
DEFAULT_PV_PLACEHOLDER = '<MISSING_PV_SUMMARY>'


CONDITION_ORDER = [
    ('labeled', 'placeholder'),
    ('raw', 'placeholder'),
    ('labeled', 'omit'),
    ('raw', 'omit'),
]


def _condition_rank(cde_format: object, placeholder_strategy: object) -> int:
    key = (str(cde_format or ''), normalize_placeholder_strategy(placeholder_strategy))
    try:
        return CONDITION_ORDER.index(key)
    except ValueError:
        return len(CONDITION_ORDER)


def select_best_refinement_conditions_by_model(
    top2_manifest: pd.DataFrame,
    refinement_runs: pd.DataFrame,
    *,
    tie_margin: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if top2_manifest.empty:
        raise SystemExit('Top-2 representation manifest is empty.')
    if refinement_runs.empty:
        raise SystemExit('No Step 8 refinement runs found.')

    top2 = top2_manifest.copy()
    top2['base_model_id'] = top2['base_model_id'].astype(str)
    top2['query_variant'] = top2['query_variant'].astype(str)
    top2['recipe'] = top2['recipe'].astype(str)
    if 'selected_rank' not in top2.columns:
        top2['selected_rank'] = range(1, len(top2) + 1)
    top2['repr_key'] = top2['query_variant'] + '||' + top2['recipe']
    if 'model_name' not in top2.columns:
        top2['model_name'] = top2['base_model_id']

    runs = refinement_runs.copy()
    runs['query_variant'] = runs['query_variant'].astype(str)
    runs['recipe'] = runs['recipe'].astype(str)
    runs['repr_key'] = runs['query_variant'] + '||' + runs['recipe']
    runs['placeholder_strategy'] = runs['placeholder_strategy'].map(normalize_placeholder_strategy)

    top2_keys = top2[['base_model_id', 'model_name', 'selected_rank', 'repr_key', 'query_variant', 'recipe']].drop_duplicates().rename(
        columns={'base_model_id': 'phase1_base_model_id', 'model_name': 'phase1_model_name'}
    )
    matched = runs.merge(
        top2_keys,
        on=['repr_key', 'query_variant', 'recipe'],
        how='inner',
    )
    matched['base_model_id'] = matched['phase1_base_model_id'].astype(str)
    matched['model_name'] = matched['phase1_model_name'].astype(str)
    if matched.empty:
        raise SystemExit('No Step 8 refinement runs matched the selected top-2 representations.')

    expected_counts = top2.groupby('base_model_id', dropna=False)['repr_key'].nunique().rename('expected_repr_count').reset_index()
    coverage = (
        matched.groupby(['base_model_id', 'cde_format', 'placeholder_strategy', 'seed'], dropna=False)['repr_key']
        .nunique()
        .rename('matched_repr_count')
        .reset_index()
        .merge(expected_counts, on='base_model_id', how='left')
    )
    incomplete = coverage[coverage['matched_repr_count'] != coverage['expected_repr_count']]
    if not incomplete.empty:
        problems = []
        for row in incomplete.head(5).itertuples(index=False):
            problems.append(
                f"base_model_id={row.base_model_id}, cde_format={row.cde_format}, placeholder_strategy={row.placeholder_strategy}, seed={row.seed}, matched={row.matched_repr_count}, expected={row.expected_repr_count}"
            )
        preview = '; '.join(problems)
        if len(incomplete) > 5:
            preview += f' ... ({len(incomplete)} groups total)'
        raise SystemExit(f'Incomplete Step 8 refinement coverage for selected representations: {preview}')

    per_seed = (
        matched.groupby(['base_model_id', 'cde_format', 'placeholder_strategy', 'seed'], dropna=False)
        .agg(
            n_runs=('run_dir', 'count'),
            n_representations=('repr_key', 'nunique'),
            val_recall5=('val_recall5', 'mean'),
            val_mrr100=('val_mrr100', 'mean'),
        )
        .reset_index()
    )

    summary = (
        per_seed.groupby(['base_model_id', 'cde_format', 'placeholder_strategy'], dropna=False)
        .agg(
            n_runs=('seed', 'count'),
            n_representations=('n_representations', 'max'),
            val_recall5_mean=('val_recall5', 'mean'),
            val_recall5_std=('val_recall5', 'std'),
            val_mrr100_mean=('val_mrr100', 'mean'),
            val_mrr100_std=('val_mrr100', 'std'),
        )
        .reset_index()
    )
    for col in ['val_recall5_std', 'val_mrr100_std']:
        summary[col] = summary[col].fillna(0.0)
    summary['condition_label'] = summary['cde_format'].astype(str) + ' + ' + summary['placeholder_strategy'].astype(str)
    summary['condition_rank'] = [
        _condition_rank(cf, ps) for cf, ps in zip(summary['cde_format'], summary['placeholder_strategy'])
    ]

    ranked = order_candidates_with_tie_margin(
        summary,
        primary_col='val_recall5_mean',
        secondary_col='val_mrr100_mean',
        tie_margin=float(tie_margin),
        fallback_cols=['condition_rank', 'cde_format', 'placeholder_strategy'],
        fallback_ascending=[True, True, True],
        group_cols=['base_model_id'],
    )
    winners = ranked.groupby('base_model_id', dropna=False, sort=False).head(1).reset_index(drop=True)
    return matched, ranked, winners


def merge_top2_with_refinement_conditions(
    top2_manifest: pd.DataFrame,
    best_conditions: pd.DataFrame,
) -> pd.DataFrame:
    if top2_manifest.empty:
        raise SystemExit('Top-2 representation manifest is empty.')
    if best_conditions.empty:
        raise SystemExit('Best refinement-condition table is empty.')

    top2 = top2_manifest.copy()
    top2['base_model_id'] = top2['base_model_id'].astype(str)
    best = best_conditions.copy()
    best['base_model_id'] = best['base_model_id'].astype(str)

    keep_cols = [
        'base_model_id',
        'cde_format',
        'placeholder_strategy',
        'condition_label',
        'val_recall5_mean',
        'val_mrr100_mean',
        'selection_rank',
    ]
    merged = top2.rename(columns={
        'cde_format': 'step7_cde_format',
        'placeholder_strategy': 'step7_placeholder_strategy',
    }).merge(best[[c for c in keep_cols if c in best.columns]], on='base_model_id', how='left')
    missing = merged['cde_format'].isna() | merged['placeholder_strategy'].isna()
    if bool(missing.any()):
        models = sorted(merged.loc[missing, 'base_model_id'].astype(str).unique().tolist())
        raise SystemExit(f'Missing Step 8 best-condition rows for base models: {models}')
    merged['selection_source'] = 'step7_top2_plus_step8_condition'
    return merged


def slug(s: object) -> str:
    return str(s).replace('/', '__').replace(' ', '_').replace(':', '_')


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def normalize_placeholder_strategy(strategy: object, *, default: str = DEFAULT_PLACEHOLDER_STRATEGY) -> str:
    value = str(strategy or '').strip().lower()
    if not value or value in {'nan', 'none'}:
        value = str(default).strip().lower()
    if value not in {'omit', 'placeholder'}:
        raise SystemExit(
            f"Unknown placeholder strategy: {strategy!r}. Expected 'omit' or 'placeholder'."
        )
    return value


def extract_placeholder_strategy(record: Mapping[str, Any], *, default: str = DEFAULT_PLACEHOLDER_STRATEGY) -> str:
    representation = record.get('representation') if isinstance(record.get('representation'), Mapping) else {}
    rep_recipe_cfg = representation.get('recipe_configs') if isinstance(representation.get('recipe_configs'), Mapping) else {}
    top_recipe_cfg = record.get('recipe_configs') if isinstance(record.get('recipe_configs'), Mapping) else {}
    candidate = (
        record.get('placeholder_strategy')
        or record.get('placeholder_policy')
        or rep_recipe_cfg.get('placeholder_policy')
        or top_recipe_cfg.get('placeholder_policy')
        or default
    )
    return normalize_placeholder_strategy(candidate, default=default)


def apply_placeholder_strategy(
    recipe_configs: Mapping[str, Any] | None,
    *,
    placeholder_strategy: str,
) -> Dict[str, Any]:
    recipe_cfg = deepcopy(dict(recipe_configs or {}))
    recipe_cfg['placeholder_policy'] = normalize_placeholder_strategy(placeholder_strategy)
    recipe_cfg.setdefault('short_name_placeholder', DEFAULT_SHORT_NAME_PLACEHOLDER)
    recipe_cfg.setdefault('pv_placeholder', DEFAULT_PV_PLACEHOLDER)
    return recipe_cfg


def resolved_ft1_block(cfg_or_path: str | Path | Mapping[str, Any]) -> Dict[str, Any]:
    cfg_all: Dict[str, Any]
    if isinstance(cfg_or_path, (str, Path)):
        cfg_all = load_config(cfg_or_path)
    else:
        cfg_all = dict(cfg_or_path)
    cfg = cfg_all.get('finetune_phase1', cfg_all)
    if not isinstance(cfg, dict):
        raise SystemExit('Could not resolve finetune_phase1 block from template config')
    return cfg


def collect_phase1_runs(stage_root: Path) -> pd.DataFrame:
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
        val = (metrics.get('metrics_by_split') or {}).get('val') or {}
        rep = rc.get('representation') or {}
        train = rc.get('train') or {}
        model_name = str(rc.get('model_name') or rc.get('base_model_id') or rc.get('base_model') or '')
        base_model_id = str(rc.get('base_model_id') or rc.get('base_model') or model_name)
        rows.append(
            {
                'run_dir': str(run_dir),
                'model_path': str(run_dir / 'model'),
                'model_name': model_name,
                'base_model_id': base_model_id,
                'query_variant': str(rep.get('query_variant') or rc.get('query_variant') or ''),
                'recipe': str(rep.get('recipe') or rc.get('recipe') or ''),
                'cde_format': str(rep.get('cde_format') or rc.get('cde_format') or 'labeled'),
                'rerank_mode': str(rep.get('rerank_mode') or rc.get('rerank_mode') or 'R0'),
                'hybrid_alpha': float(rep.get('hybrid_alpha') or rc.get('hybrid_alpha') or 0.5),
                'placeholder_strategy': extract_placeholder_strategy(rc),
                'seed': int(train.get('seed', rc.get('seed', -1))),
                'loss': str(train.get('loss') or rc.get('loss') or ''),
                'lr': float(train.get('lr') or rc.get('lr') or float('nan')),
                'batch_size': int(train.get('batch_size', rc.get('batch_size', -1))),
                'temperature': float(train.get('temperature') or rc.get('temperature') or float('nan')),
                'epochs': int(train.get('epochs', rc.get('epochs', -1))),
                'val_recall5': float(val.get('recall@5', float('nan'))),
                'val_mrr100': float(val.get('mrr@100', float('nan'))),
            }
        )
    return pd.DataFrame(rows)


def summarize_runs(df: pd.DataFrame, *, group_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        raise SystemExit('No runs found under the supplied stage root.')
    summ = df.groupby(list(group_cols), dropna=False).agg(
        n_runs=('run_dir', 'count'),
        val_recall5_mean=('val_recall5', 'mean'),
        val_recall5_std=('val_recall5', 'std'),
        val_mrr100_mean=('val_mrr100', 'mean'),
        val_mrr100_std=('val_mrr100', 'std'),
    ).reset_index()
    for col in ['val_recall5_std', 'val_mrr100_std']:
        summ[col] = summ[col].fillna(0.0)
    return summ


def build_seed_coverage_table(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    expected_seeds: Sequence[int],
) -> pd.DataFrame:
    expected = [int(x) for x in expected_seeds]
    expected_unique = sorted(set(expected))
    columns = list(group_cols) + [
        'n_seeds_observed',
        'observed_seeds',
        'n_seeds_expected',
        'expected_seeds',
        'n_missing_seeds',
        'missing_seeds',
        'incomplete_seed_coverage',
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    rows: List[Dict[str, Any]] = []
    for group_key, sub in df.groupby(list(group_cols), dropna=False, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        observed = [int(x) for x in pd.to_numeric(sub['seed'], errors='coerce').dropna().astype(int).tolist()]
        observed_unique = sorted(set(observed))
        missing = [seed for seed in expected_unique if seed not in observed_unique]
        row = {col: val for col, val in zip(group_cols, group_key)}
        row.update(
            {
                'n_seeds_observed': int(len(observed_unique)),
                'observed_seeds': '|'.join(str(x) for x in observed_unique),
                'n_seeds_expected': int(len(expected_unique)),
                'expected_seeds': '|'.join(str(x) for x in expected_unique),
                'n_missing_seeds': int(len(missing)),
                'missing_seeds': '|'.join(str(x) for x in missing),
                'incomplete_seed_coverage': bool(expected_unique and observed_unique != expected_unique),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def validate_seed_coverage(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    expected_seeds: Sequence[int],
    stage_label: str,
) -> None:
    expected = [int(x) for x in expected_seeds]
    if not expected:
        return
    problems: List[str] = []
    for group_key, sub in df.groupby(list(group_cols), dropna=False, sort=False):
        observed = [int(x) for x in sorted(sub['seed'].dropna().astype(int).tolist())]
        observed_unique = sorted(set(observed))
        if observed != expected or observed_unique != expected:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            key_str = ', '.join(f'{col}={val}' for col, val in zip(group_cols, group_key))
            problems.append(
                f'{key_str} :: observed seeds={observed} unique={observed_unique} expected={expected}'
            )
    if problems:
        preview = '; '.join(problems[:5])
        if len(problems) > 5:
            preview += f' ... ({len(problems)} groups total)'
        raise SystemExit(f'Incomplete or inconsistent seed coverage for {stage_label}: {preview}')


def select_representative_run(df: pd.DataFrame, winner_summary: pd.Series, *, match_cols: Sequence[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in match_cols:
        mask &= df[col] == winner_summary[col]
    sub = df[mask].sort_values(['val_recall5', 'val_mrr100', 'seed'], ascending=[False, False, True]).reset_index(drop=True)
    if sub.empty:
        raise SystemExit('Could not find an individual run matching the winning summary row.')
    return sub.iloc[0]


def load_manifest_records(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f'Manifest not found: {p}')
    if p.suffix.lower() == '.csv':
        return pd.read_csv(p).to_dict(orient='records')
    payload = json.loads(p.read_text(encoding='utf-8'))
    if isinstance(payload, list):
        return [dict(x) for x in payload]
    for key in ['records', 'rows', 'top2', 'winners', 'model_selections']:
        xs = payload.get(key)
        if isinstance(xs, list):
            return [dict(x) for x in xs]
    raise SystemExit(f'Unsupported manifest structure in {p}')


def build_phase1_config(
    *,
    base_cfg: Mapping[str, Any],
    model_name: str,
    base_model_id: str,
    query_variant: str,
    recipe: str,
    cde_format: str,
    rerank_mode: str,
    hybrid_alpha: float,
    stage_tag: str,
    losses: Sequence[str],
    seeds: Sequence[int],
    lrs: Sequence[float] | None = None,
    batch_sizes: Sequence[int] | None = None,
    temperatures: Sequence[float] | None = None,
    epochs: Sequence[int] | None = None,
    eval_splits: Sequence[str] | None = None,
    placeholder_strategy: str | None = None,
) -> Dict[str, Any]:
    cfg = deepcopy(dict(base_cfg))
    cfg['model_name'] = str(model_name)
    cfg['base_model_id'] = str(base_model_id)
    cfg['query_variant'] = str(query_variant)
    cfg['recipe'] = str(recipe)
    cfg['cde_format'] = str(cde_format)
    cfg['rerank_mode'] = str(rerank_mode)
    cfg['hybrid_alpha'] = float(hybrid_alpha)
    cfg['artifacts_dir'] = 'artifacts'
    cfg['runs_dir'] = 'auto'
    cfg['stage_tag'] = str(stage_tag)

    if placeholder_strategy is not None:
        cfg['recipe_configs'] = apply_placeholder_strategy(
            cfg.get('recipe_configs') if isinstance(cfg.get('recipe_configs'), Mapping) else {},
            placeholder_strategy=str(placeholder_strategy),
        )

    train = deepcopy(dict(cfg.get('train') or {}))
    train['seeds'] = [int(x) for x in seeds]
    train['losses'] = [str(x) for x in losses]
    if lrs is not None:
        train['lrs'] = [float(x) for x in lrs]
    if batch_sizes is not None:
        train['batch_sizes'] = [int(x) for x in batch_sizes]
    if temperatures is not None:
        train['temperatures'] = [float(x) for x in temperatures]
    if epochs is not None:
        train['epochs'] = [int(x) for x in epochs]
    cfg['train'] = train

    if eval_splits is not None:
        ev = deepcopy(dict(cfg.get('eval') or {}))
        ev['eval_splits'] = [str(x) for x in eval_splits]
        cfg['eval'] = ev

    return {'finetune_phase1': cfg}


def task_runtime_dirs(*, stage_tag: str, base_model_id: str, task_id: str) -> tuple[str, str]:
    stage_slug = slug(str(stage_tag or 'finetune_phase1'))
    model_slug = slug(str(base_model_id))
    task_slug = slug(str(task_id))
    stage_root = Path('artifacts') / stage_slug / model_slug
    return str(stage_root / 'task_artifacts' / task_slug), str(stage_root / 'runs')


def write_task_manifest(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    required = ['task_id', 'stage', 'config_path', 'model_name', 'base_model_id']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f'Task manifest missing required columns: {missing}')

    if df.empty:
        raise SystemExit('Task manifest is empty; no tasks to write.')

    dup_task_ids = df[df['task_id'].astype(str).duplicated(keep=False)]
    if not dup_task_ids.empty:
        vals = sorted(set(dup_task_ids['task_id'].astype(str).tolist()))
        raise SystemExit(f'Duplicate task_id values would collide during execution: {vals}')

    dup_cfg = df[df['config_path'].astype(str).duplicated(keep=False)]
    if not dup_cfg.empty:
        vals = sorted(set(dup_cfg['config_path'].astype(str).tolist()))
        raise SystemExit(f'Duplicate config_path values would collide during execution: {vals}')

    for optional_col, label in [('task_artifacts_dir', 'task_artifacts_dir'), ('task_collision_key', 'task_collision_key')]:
        if optional_col in df.columns:
            dup = df[df[optional_col].astype(str).duplicated(keep=False)]
            if not dup.empty:
                vals = sorted(set(dup[optional_col].astype(str).tolist()))
                raise SystemExit(f'Duplicate {label} values would collide during execution: {vals}')

    df.to_csv(path, sep='\t', index=False)


__all__ = [
    'DEFAULT_PLACEHOLDER_STRATEGY',
    'DEFAULT_PV_PLACEHOLDER',
    'DEFAULT_SCREEN_BASELINE_LOSS',
    'DEFAULT_SCREEN_BATCH_SIZE',
    'DEFAULT_SCREEN_EPOCHS',
    'DEFAULT_SCREEN_LOSSES',
    'DEFAULT_SCREEN_LR',
    'DEFAULT_SCREEN_REP_STAGE_TAG',
    'DEFAULT_SCREEN_LOSS_STAGE_TAG',
    'DEFAULT_SCREEN_SEEDS',
    'DEFAULT_SCREEN_STAGE_TAG',
    'DEFAULT_SCREEN_TEMPERATURE',
    'DEFAULT_SHORT_NAME_PLACEHOLDER',
    'CONDITION_ORDER',
    'apply_placeholder_strategy',
    'build_phase1_config',
    'build_seed_coverage_table',
    'collect_phase1_runs',
    'extract_placeholder_strategy',
    'load_manifest_records',
    'normalize_placeholder_strategy',
    'resolved_ft1_block',
    'select_representative_run',
    'merge_top2_with_refinement_conditions',
    'select_best_refinement_conditions_by_model',
    'slug',
    'summarize_runs',
    'task_runtime_dirs',
    'validate_seed_coverage',
    'write_task_manifest',
    'write_yaml',
]
