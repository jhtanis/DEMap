from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from demap_repro.biencoder.phase1_screening import extract_placeholder_strategy, select_representative_run, slug
from demap_repro.biencoder.selection_rules import add_representation_complexity_columns, order_candidates_with_tie_margin


DEFAULT_SELECTION_SPLIT = 'val'
DEFAULT_SELECTION_METRIC = 'recall@5'
DEFAULT_SECONDARY_METRIC = 'mrr@100'
DEFAULT_TIE_MARGIN = 0.0

OFF_THE_SHELF_GROUP_COLS = [
    'model_name',
    'base_model_id',
    'query_variant',
    'recipe',
    'cde_format',
    'rerank_mode',
    'hybrid_alpha',
    'placeholder_strategy',
]

RUN_METADATA_COLS = [
    'run_dir',
    'run_id',
    'stage_tag',
    'model_name',
    'base_model_id',
    'query_variant',
    'recipe',
    'cde_format',
    'rerank_mode',
    'hybrid_alpha',
    'placeholder_strategy',
    'seed',
    'split',
]


def _coerce_numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return float(out)
    return out


def _metadata_from_run_config(run_dir: Path, rc: Mapping[str, Any]) -> Dict[str, Any]:
    rep = rc.get('representation') if isinstance(rc.get('representation'), Mapping) else {}
    train = rc.get('train') if isinstance(rc.get('train'), Mapping) else {}
    model_name = str(rc.get('model_name') or rc.get('base_model_id') or rc.get('base_model') or '')
    base_model_id = str(rc.get('base_model_id') or rc.get('base_model') or model_name)
    return {
        'run_dir': str(run_dir),
        'run_id': str(run_dir.name),
        'stage_tag': str(rc.get('stage_tag') or ''),
        'model_name': model_name,
        'base_model_id': base_model_id,
        'query_variant': str(rep.get('query_variant') or rc.get('query_variant') or ''),
        'recipe': str(rep.get('recipe') or rc.get('recipe') or ''),
        'cde_format': str(rep.get('cde_format') or rc.get('cde_format') or 'labeled'),
        'rerank_mode': str(rep.get('rerank_mode') or rc.get('rerank_mode') or 'R0'),
        'hybrid_alpha': float(rep.get('hybrid_alpha') or rc.get('hybrid_alpha') or 0.5),
        'placeholder_strategy': extract_placeholder_strategy(rc),
        'seed': int(rc.get('seed', train.get('seed', -1))),
    }


def collect_off_the_shelf_runs(stage_root: Path) -> tuple[pd.DataFrame, list[str]]:
    rows: List[Dict[str, Any]] = []
    metric_cols: set[str] = set()

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

        met_by_split = metrics.get('metrics_by_split') or {}
        if not isinstance(met_by_split, Mapping) or not met_by_split:
            continue

        base = _metadata_from_run_config(run_dir, rc)
        for split_name, split_metrics_raw in met_by_split.items():
            if not isinstance(split_metrics_raw, Mapping):
                continue
            row = dict(base)
            row['split'] = str(split_name)
            any_metric = False
            for metric_name, metric_value in split_metrics_raw.items():
                metric_num = _coerce_numeric(metric_value)
                if metric_num is None:
                    continue
                row[str(metric_name)] = metric_num
                metric_cols.add(str(metric_name))
                any_metric = True
            if any_metric:
                rows.append(row)

    df = pd.DataFrame(rows)
    ordered_metrics = sorted(metric_cols)
    if df.empty:
        return df, ordered_metrics
    for col in RUN_METADATA_COLS:
        if col not in df.columns:
            df[col] = pd.Series(dtype='object')
    for col in ordered_metrics:
        if col not in df.columns:
            df[col] = pd.Series(dtype=float)
    df = df[RUN_METADATA_COLS + ordered_metrics].copy()
    return df, ordered_metrics


def summarize_off_the_shelf_runs(
    df: pd.DataFrame,
    *,
    metric_cols: Sequence[str],
    group_cols: Sequence[str] = OFF_THE_SHELF_GROUP_COLS,
) -> pd.DataFrame:
    if df.empty:
        raise SystemExit('No off-the-shelf runs found under the supplied stage root.')

    named_aggs: Dict[str, tuple[str, str]] = {
        'n_runs': ('run_dir', 'count'),
        'n_seeds': ('seed', 'nunique'),
    }
    for metric in metric_cols:
        named_aggs[f'{metric}_mean'] = (metric, 'mean')
        named_aggs[f'{metric}_std'] = (metric, 'std')
        named_aggs[f'{metric}_n'] = (metric, 'count')

    summ = (
        df.groupby(list(group_cols) + ['split'], dropna=False)
        .agg(**named_aggs)
        .reset_index()
    )
    for metric in metric_cols:
        std_col = f'{metric}_std'
        if std_col in summ.columns:
            summ[std_col] = summ[std_col].fillna(0.0)
    return summ


def build_selection_summary(
    summary_by_split: pd.DataFrame,
    *,
    selection_split: str,
    selection_metric: str,
    secondary_metric: str,
) -> pd.DataFrame:
    sel = summary_by_split[summary_by_split['split'] == str(selection_split)].copy().reset_index(drop=True)
    if sel.empty:
        raise SystemExit(f'No rows found for selection split: {selection_split!r}')

    required = [f'{selection_metric}_mean', f'{secondary_metric}_mean']
    missing = [col for col in required if col not in sel.columns]
    if missing:
        raise SystemExit(
            'Selection summary is missing required metric columns: ' + ', '.join(missing)
        )

    sel = sel.drop(columns=['split'])
    sel['selection_split'] = str(selection_split)
    sel['selection_metric'] = str(selection_metric)
    sel['secondary_metric'] = str(secondary_metric)
    sel['selection_recall5_mean'] = pd.to_numeric(sel[f'{selection_metric}_mean'], errors='coerce')
    sel['selection_recall5_std'] = pd.to_numeric(sel.get(f'{selection_metric}_std'), errors='coerce').fillna(0.0)
    sel['selection_recall5_n'] = pd.to_numeric(sel.get(f'{selection_metric}_n'), errors='coerce').fillna(0).astype(int)
    sel['selection_mrr100_mean'] = pd.to_numeric(sel[f'{secondary_metric}_mean'], errors='coerce')
    sel['selection_mrr100_std'] = pd.to_numeric(sel.get(f'{secondary_metric}_std'), errors='coerce').fillna(0.0)
    sel['selection_mrr100_n'] = pd.to_numeric(sel.get(f'{secondary_metric}_n'), errors='coerce').fillna(0).astype(int)
    return sel


def rank_off_the_shelf_by_model(selection_summary: pd.DataFrame, *, tie_margin: float = DEFAULT_TIE_MARGIN) -> pd.DataFrame:
    work = add_representation_complexity_columns(selection_summary)
    ranked = order_candidates_with_tie_margin(
        work,
        primary_col='selection_recall5_mean',
        secondary_col='selection_mrr100_mean',
        tie_margin=float(tie_margin),
        fallback_cols=[
            'selection_recall5_std',
            'selection_mrr100_std',
            'selection_question_complexity_rank',
            'selection_recipe_complexity_max_rank',
            'selection_recipe_complexity_n_atoms',
            'selection_recipe_complexity_sum_rank',
            'selection_recipe_complexity_profile',
            'query_variant',
            'recipe',
            'cde_format',
            'rerank_mode',
            'placeholder_strategy',
        ],
        fallback_ascending=[True, True, True, True, True, True, True, True, True, True, True, True],
        group_cols=['base_model_id'],
    )
    return ranked.reset_index(drop=True)


def representative_selection_runs(
    runs_df: pd.DataFrame,
    *,
    selection_split: str,
    selection_metric: str,
    secondary_metric: str,
) -> pd.DataFrame:
    sel_runs = runs_df[runs_df['split'] == str(selection_split)].copy().reset_index(drop=True)
    if sel_runs.empty:
        raise SystemExit(f'No individual runs found for selection split: {selection_split!r}')
    if selection_metric not in sel_runs.columns:
        raise SystemExit(f'Selection metric not present in individual-run table: {selection_metric!r}')
    if secondary_metric not in sel_runs.columns:
        raise SystemExit(f'Secondary metric not present in individual-run table: {secondary_metric!r}')
    sel_runs['val_recall5'] = pd.to_numeric(sel_runs[selection_metric], errors='coerce')
    sel_runs['val_mrr100'] = pd.to_numeric(sel_runs[secondary_metric], errors='coerce')
    return sel_runs


def choose_model_winners(
    ranked_selection: pd.DataFrame,
    *,
    selection_runs_df: pd.DataFrame,
    group_cols: Sequence[str] = OFF_THE_SHELF_GROUP_COLS,
) -> tuple[pd.DataFrame, list[Dict[str, Any]]]:
    winners = ranked_selection.groupby('base_model_id', dropna=False, sort=False).head(1).copy().reset_index(drop=True)
    payloads: list[Dict[str, Any]] = []
    rows: list[Dict[str, Any]] = []

    for _, winner_summary in winners.iterrows():
        rep = select_representative_run(selection_runs_df, winner_summary, match_cols=group_cols)
        row = winner_summary.to_dict()
        row['representative_run_dir'] = str(rep['run_dir'])
        row['representative_run_id'] = str(rep['run_id'])
        row['representative_seed'] = int(rep['seed'])
        row['representative_selection_recall5'] = float(rep['val_recall5'])
        row['representative_selection_mrr100'] = float(rep['val_mrr100'])
        rows.append(row)
        payloads.append(
            {
                'winner_summary': winner_summary.to_dict(),
                'representative_run': rep.to_dict(),
            }
        )

    return pd.DataFrame(rows), payloads


def build_winner_scores_by_split(
    summary_by_split: pd.DataFrame,
    winners_df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = OFF_THE_SHELF_GROUP_COLS,
) -> pd.DataFrame:
    winner_meta_cols = [
        *group_cols,
        'selection_rank',
        'selection_primary_best',
        'selection_primary_gap',
        'selection_within_tie_margin',
        'selection_tie_margin',
        'selection_split',
        'selection_metric',
        'secondary_metric',
        'selection_recall5_mean',
        'selection_recall5_std',
        'selection_recall5_n',
        'selection_mrr100_mean',
        'selection_mrr100_std',
        'selection_mrr100_n',
        'representative_run_dir',
        'representative_run_id',
        'representative_seed',
        'representative_selection_recall5',
        'representative_selection_mrr100',
    ]
    meta = winners_df[winner_meta_cols].copy()
    merged = summary_by_split.merge(meta, on=list(group_cols), how='inner')
    return merged.reset_index(drop=True)


def rank_best_model_by_split(
    winner_scores_by_split: pd.DataFrame,
    *,
    selection_metric: str,
    secondary_metric: str,
    tie_margin: float = DEFAULT_TIE_MARGIN,
) -> pd.DataFrame:
    primary_col = f'{selection_metric}_mean'
    secondary_col = f'{secondary_metric}_mean'
    missing = [col for col in [primary_col, secondary_col] if col not in winner_scores_by_split.columns]
    if missing:
        raise SystemExit('Winner scores are missing required ranking columns: ' + ', '.join(missing))

    work = add_representation_complexity_columns(winner_scores_by_split)
    ranked = order_candidates_with_tie_margin(
        work,
        primary_col=primary_col,
        secondary_col=secondary_col,
        tie_margin=float(tie_margin),
        fallback_cols=[
            f'{selection_metric}_std',
            f'{secondary_metric}_std',
            'selection_question_complexity_rank',
            'selection_recipe_complexity_max_rank',
            'selection_recipe_complexity_n_atoms',
            'selection_recipe_complexity_sum_rank',
            'selection_recipe_complexity_profile',
            'model_name',
            'query_variant',
            'recipe',
        ],
        fallback_ascending=[True, True, True, True, True, True, True, True, True, True],
        group_cols=['split'],
    )
    return ranked.reset_index(drop=True)


__all__ = [
    'DEFAULT_SECONDARY_METRIC',
    'DEFAULT_SELECTION_METRIC',
    'DEFAULT_SELECTION_SPLIT',
    'DEFAULT_TIE_MARGIN',
    'OFF_THE_SHELF_GROUP_COLS',
    'build_selection_summary',
    'build_winner_scores_by_split',
    'choose_model_winners',
    'collect_off_the_shelf_runs',
    'rank_best_model_by_split',
    'rank_off_the_shelf_by_model',
    'representative_selection_runs',
    'slug',
    'summarize_off_the_shelf_runs',
]
