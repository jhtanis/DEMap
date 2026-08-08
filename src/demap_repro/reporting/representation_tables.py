from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from demap_repro.biencoder.engine import baseline_grid
from demap_repro.biencoder.engine.st_loader import parse_st_model_spec
from demap_repro.biencoder.selection_rules import DEFAULT_TIE_MARGIN, order_candidates_with_tie_margin
from demap_repro.text.recipes import parse_recipe
from demap_repro.utils.config import load_config

HEADLINE_METRICS = ['recall@5', 'mrr@100', 'recall@10', 'top1_accuracy']
PRIMARY_METRIC = 'recall@5'
SECONDARY_METRIC = 'mrr@100'
TOP4_QUERY_VARIANT_ORDER = ['Q1', 'Q2', 'Q3', 'Q4']
CONDITION_ORDER = [
    ('labeled', 'placeholder'),
    ('raw', 'placeholder'),
    ('labeled', 'omit'),
    ('raw', 'omit'),
]
POOLING_ORDER = ['mean', 'cls']

QUERY_DISPLAY = {
    'Q1': 'RAW',
    'Q2': 'PREF_PV',
    'Q3': 'RAW_PV',
    'Q4': 'RAW_PV_PH',
}

CDE_ATOM_DISPLAY = {
    'v1': 'SN',
    'v2a': 'LN',
    'v2b': 'DEF',
    'v3': 'PQT',
    'v4': 'VD',
    'v5': 'PV',
    'v6': 'DEC',
}


def _expand_recipe_atoms(recipe: object) -> List[str]:
    parts = parse_recipe(str(recipe))
    atoms: List[str] = []
    for part in parts:
        if part == 'v2':
            atoms.extend(['v2a', 'v2b'])
        else:
            atoms.append(part)
    return atoms


def query_display_name(query_variant: object) -> str:
    return QUERY_DISPLAY.get(str(query_variant or '').strip().upper(), str(query_variant or ''))


def recipe_display_name(recipe: object) -> str:
    try:
        atoms = _expand_recipe_atoms(recipe)
    except Exception:
        return str(recipe)
    labels = [CDE_ATOM_DISPLAY.get(atom, atom) for atom in atoms]
    return '_'.join(labels)


def representation_display_name(query_variant: object, recipe: object) -> str:
    return f"{query_display_name(query_variant)} × {recipe_display_name(recipe)}"


def _load_baseline_cfg(path: str | Path) -> Dict[str, Any]:
    cfg_all = load_config(str(path))
    cfg = cfg_all.get('baseline_grid', cfg_all)
    if not isinstance(cfg, dict):
        raise TypeError(f'Config at {path} did not resolve to a dict.')
    return cfg


def _union_eval_splits(config_paths: Sequence[Path]) -> List[str]:
    out: List[str] = []
    for path in config_paths:
        cfg = _load_baseline_cfg(path)
        for split in list(cfg.get('eval_splits') or []):
            split_text = str(split)
            if split_text not in out:
                out.append(split_text)
    return out


def _pooling_variant(model_name: object) -> Optional[str]:
    text = str(model_name or '')
    spec = parse_st_model_spec(text)
    if 'sapbert-from-pubmedbert-fulltext' not in spec.base_model.lower():
        return None
    return 'cls' if spec.variant == 'cls' else 'mean'


def _model_label(model_name: object) -> str:
    text = str(model_name or '')
    spec = parse_st_model_spec(text)
    tail = spec.base_model.split('/')[-1] if '/' in spec.base_model else spec.base_model
    if 'sapbert-from-pubmedbert-fulltext' in spec.base_model.lower():
        return 'SapBERT [CLS]' if spec.variant == 'cls' else 'SapBERT [mean]'
    return tail


def _extract_placeholder_policy(record: Mapping[str, Any]) -> str:
    rep = record.get('representation') if isinstance(record.get('representation'), Mapping) else {}
    rep_recipe_cfg = rep.get('recipe_configs') if isinstance(rep.get('recipe_configs'), Mapping) else {}
    top_recipe_cfg = record.get('recipe_configs') if isinstance(record.get('recipe_configs'), Mapping) else {}
    candidate = (
        record.get('placeholder_policy')
        or record.get('placeholder_strategy')
        or rep_recipe_cfg.get('placeholder_policy')
        or top_recipe_cfg.get('placeholder_policy')
        or 'placeholder'
    )
    value = str(candidate or '').strip().lower()
    if not value or value in {'nan', 'none'}:
        value = 'placeholder'
    return value


def _collect_stage_split_metrics(stage_root: Path) -> pd.DataFrame:
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
        rep = rc.get('representation') if isinstance(rc.get('representation'), Mapping) else {}
        model_name = str(rc.get('model_name') or rc.get('base_model_id') or '')
        query_variant = str(rep.get('query_variant') or rc.get('query_variant') or '')
        recipe = str(rep.get('recipe') or rc.get('recipe') or '')
        cde_format = str(rep.get('cde_format') or rc.get('cde_format') or 'labeled')
        rerank_mode = str(rep.get('rerank_mode') or rc.get('rerank_mode') or 'none')
        placeholder_policy = _extract_placeholder_policy(rc)
        seed = int(rc.get('seed', -1))
        for split, vals in (metrics.get('metrics_by_split') or {}).items():
            if not isinstance(vals, Mapping):
                continue
            for metric in HEADLINE_METRICS:
                value = vals.get(metric)
                if value is None:
                    continue
                try:
                    value_f = float(value)
                except Exception:
                    continue
                if not np.isfinite(value_f):
                    continue
                rows.append(
                    {
                        'model_name': model_name,
                        'model_label': _model_label(model_name),
                        'pooling_variant': _pooling_variant(model_name),
                        'seed': seed,
                        'recipe': recipe,
                        'query_variant': query_variant,
                        'cde_format': str(baseline_grid.CDE_FORMAT_ALIASES.get(cde_format, cde_format)),
                        'rerank_mode': rerank_mode,
                        'rerank_tag': baseline_grid.rerank_tag(rerank_mode),
                        'placeholder_policy': placeholder_policy,
                        'cell_label': representation_display_name(query_variant, recipe),
                        'run_dir': str(run_dir),
                        'split': str(split),
                        'metric': metric,
                        'value': value_f,
                    }
                )
    return pd.DataFrame(rows)


def aggregate_over_seeds(df: pd.DataFrame, *, group_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[*group_cols, 'mean_value', 'std_value', 'n'])

    def _std(series: pd.Series) -> float:
        if len(series) <= 1:
            return float('nan')
        return float(series.std(ddof=1))

    return (
        df.groupby(list(group_cols), dropna=False)['value']
        .agg(mean_value='mean', std_value=_std, n='count')
        .reset_index()
    )


def collapse_cells_per_seed(df: pd.DataFrame, *, group_cols_without_seed_and_cell: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[*group_cols_without_seed_and_cell, 'seed', 'value'])
    seed_cols = list(group_cols_without_seed_and_cell) + ['seed']
    return df.groupby(seed_cols, dropna=False)['value'].mean().reset_index()


def flatten_metric_pivot(df: pd.DataFrame, *, index_cols: Sequence[str], metric_order: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(index_cols))
    metric_order = list(metric_order or sorted(df['metric'].dropna().unique().tolist()))
    pivot = df.pivot_table(index=list(index_cols), columns='metric', values=['mean_value', 'std_value', 'n'], aggfunc='first')
    ordered_arrays: List[tuple[str, str]] = []
    out_cols: List[str] = []
    for metric in metric_order:
        for stat in ['mean_value', 'std_value', 'n']:
            tup = (stat, metric)
            if tup in pivot.columns:
                ordered_arrays.append(tup)
                stat_name = {'mean_value': 'mean', 'std_value': 'std', 'n': 'n'}[stat]
                out_cols.append(f'{metric}_{stat_name}')
    if ordered_arrays:
        pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(ordered_arrays))
        pivot.columns = out_cols
    return pivot.reset_index()


def _step6_metric_frame(df: pd.DataFrame, *, metric: str, split: str) -> pd.DataFrame:
    sub = df[
        (df['metric'].astype(str) == metric)
        & (df['split'].astype(str) == split)
        & (df['cde_format'].astype(str) == 'labeled')
        & (df['rerank_tag'].astype(str) == 'R0')
    ].copy()
    if sub.empty:
        raise ValueError(
            f'No rows found for metric={metric!r}, split={split!r}, cde_format="labeled", rerank_tag="R0"'
        )
    keep = ['recipe', 'query_variant', 'mean_value', 'std_value', 'n']
    return sub[keep]


def _step6_secondary_tie_warning_rows(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['query_variant', 'n_recipes_tied_after_secondary', 'recipes_tied_after_secondary', 'primary_mean_best', 'secondary_mean_best', 'selection_recipe_after_fallback', 'warning']
    rows: List[Dict[str, Any]] = []
    for qv, sub in df.groupby('query_variant', dropna=False):
        near = sub[sub['selection_within_tie_margin']].copy()
        if near.empty:
            continue
        secondary = pd.to_numeric(near['secondary_mean'], errors='coerce')
        best_secondary = secondary.max(skipna=True)
        if pd.isna(best_secondary):
            continue
        tied = near[np.isclose(secondary.to_numpy(dtype=float), float(best_secondary), atol=1e-12, rtol=0.0)].copy()
        if len(tied) > 1:
            rows.append(
                {
                    'query_variant': qv,
                    'n_recipes_tied_after_secondary': int(len(tied)),
                    'recipes_tied_after_secondary': ' | '.join(tied.sort_values('recipe')['recipe'].astype(str).tolist()),
                    'primary_mean_best': float(pd.to_numeric(tied['primary_mean'], errors='coerce').max(skipna=True)),
                    'secondary_mean_best': float(best_secondary),
                    'selection_recipe_after_fallback': str(tied.sort_values(['selection_rank', 'recipe']).iloc[0]['recipe']),
                    'warning': 'Secondary tie remains within the primary tie margin; fallback is deterministic by recipe.',
                }
            )
    return pd.DataFrame(rows, columns=cols)


def compute_step6_data(*, step5_summary_dir: Path) -> Dict[str, Any]:
    cells_path = step5_summary_dir / 'cells_agg.csv'
    runs_path = step5_summary_dir / 'runs_table.csv'
    if not cells_path.exists():
        raise FileNotFoundError(f'Missing required heatmap summary file: {cells_path}')
    cells_agg_file = pd.read_csv(cells_path)
    runs_table = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame()

    required = {'recipe', 'query_variant', 'cde_format', 'rerank_tag', 'split', 'metric', 'mean_value', 'std_value', 'n'}
    missing = sorted(required - set(cells_agg_file.columns))
    if missing:
        raise ValueError(f'cells_agg.csv is missing expected columns: {missing}')

    group_cols = ['recipe', 'query_variant', 'cde_format', 'rerank_tag', 'split', 'metric']
    agg_source = 'cells_agg.csv'
    step6_cells_agg = cells_agg_file.copy()
    mismatch_rows = pd.DataFrame()
    if not runs_table.empty and {'value', *group_cols}.issubset(runs_table.columns):
        recomputed = (
            runs_table.groupby(group_cols, dropna=False)['value']
            .agg(mean_value='mean', std_value=lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0, n='size')
            .reset_index()
        )
        step6_cells_agg = recomputed.copy()
        agg_source = 'runs_table.csv (recomputed)'
        compare = recomputed.merge(
            cells_agg_file[group_cols + ['mean_value', 'std_value', 'n']],
            on=group_cols,
            how='left',
            suffixes=('_recomputed', '_file'),
        )
        compare['mean_diff_abs'] = (compare['mean_value_recomputed'] - compare['mean_value_file']).abs()
        compare['std_diff_abs'] = (compare['std_value_recomputed'] - compare['std_value_file']).abs()
        compare['n_diff'] = compare['n_recomputed'] - compare['n_file']
        mismatch_rows = compare[(compare['mean_diff_abs'] > 1e-12) | (compare['std_diff_abs'] > 1e-12) | (compare['n_diff'] != 0)].copy()

    primary = _step6_metric_frame(step6_cells_agg, metric=PRIMARY_METRIC, split='val').rename(
        columns={'mean_value': 'primary_mean', 'std_value': 'primary_std', 'n': 'primary_n'}
    )
    secondary = _step6_metric_frame(step6_cells_agg, metric=SECONDARY_METRIC, split='val').rename(
        columns={'mean_value': 'secondary_mean', 'std_value': 'secondary_std', 'n': 'secondary_n'}
    )
    rank_df = primary.merge(secondary, on=['recipe', 'query_variant'], how='left')

    aux = step6_cells_agg[
        (step6_cells_agg['metric'].astype(str) == PRIMARY_METRIC)
        & (step6_cells_agg['split'].astype(str) == 'test')
        & (step6_cells_agg['cde_format'].astype(str) == 'labeled')
        & (step6_cells_agg['rerank_tag'].astype(str) == 'R0')
    ][['recipe', 'query_variant', 'mean_value', 'std_value', 'n']].copy()
    if not aux.empty:
        aux = aux.rename(columns={'mean_value': 'aux_mean', 'std_value': 'aux_std', 'n': 'aux_n'})
        rank_df = rank_df.merge(aux, on=['recipe', 'query_variant'], how='left')
    else:
        rank_df['aux_mean'] = np.nan
        rank_df['aux_std'] = np.nan
        rank_df['aux_n'] = np.nan

    rank_df['cell_label'] = rank_df.apply(lambda row: representation_display_name(row['query_variant'], row['recipe']), axis=1)
    rank_df = order_candidates_with_tie_margin(
        rank_df,
        primary_col='primary_mean',
        secondary_col='secondary_mean',
        tertiary_col=None,
        tie_margin=DEFAULT_TIE_MARGIN,
        fallback_cols=['recipe'],
        fallback_ascending=[True],
        group_cols=['query_variant'],
    )
    rank_df['query_variant_rank'] = rank_df['selection_rank']
    tie_warnings = _step6_secondary_tie_warning_rows(rank_df)
    tie_qvs = set(tie_warnings['query_variant'].astype(str).tolist())
    rank_df['secondary_tie_requires_warning'] = rank_df['query_variant'].astype(str).isin(tie_qvs)

    chosen_rows: List[pd.Series] = []
    for qv in TOP4_QUERY_VARIANT_ORDER:
        sub = rank_df[rank_df['query_variant'].astype(str) == qv].copy()
        if sub.empty:
            raise ValueError(f'Missing ranked rows for {qv}')
        chosen_rows.append(sub.sort_values('selection_rank').iloc[0])
    selected_top4 = pd.DataFrame(chosen_rows).reset_index(drop=True)
    selected_top4['selection_rule'] = 'top_per_query_variant'
    selected_top4['secondary_tie_requires_warning'] = selected_top4['query_variant'].astype(str).isin(tie_qvs)
    selected_top4['plot_label'] = selected_top4['cell_label']
    selected_spec = ','.join(
        f'{row.query_variant}:{row.recipe}'
        for row in selected_top4[['query_variant', 'recipe']]
        .sort_values('query_variant', key=lambda s: s.map({q: i for i, q in enumerate(TOP4_QUERY_VARIANT_ORDER)}))
        .itertuples(index=False)
    )
    config_cmd = f'python scripts/paper_generate_configs.py --cells "{selected_spec}"'
    return {
        'agg_source': agg_source,
        'rank_table': rank_df,
        'selected_top4': selected_top4,
        'secondary_tie_warnings': tie_warnings,
        'mismatch_rows': mismatch_rows,
        'config_command': config_cmd,
    }


def compute_step8_data(*, step8_placeholder_stage_root: Path, step8_omit_stage_root: Path, refinement_config_paths: Sequence[Path]) -> Dict[str, Any]:
    placeholder_rows = _collect_stage_split_metrics(step8_placeholder_stage_root)
    omit_rows = _collect_stage_split_metrics(step8_omit_stage_root)
    all_rows = pd.concat([placeholder_rows, omit_rows], ignore_index=True) if not placeholder_rows.empty or not omit_rows.empty else pd.DataFrame()
    if all_rows.empty:
        raise ValueError('No refinement rows found in Step 8 outputs.')
    all_rows['stage_kind'] = 'refinement'

    agg_split = aggregate_over_seeds(
        all_rows,
        group_cols=['stage_kind', 'model_name', 'model_label', 'recipe', 'query_variant', 'cde_format', 'rerank_mode', 'rerank_tag', 'placeholder_policy', 'cell_label', 'split', 'metric'],
    )
    refine_split = agg_split[agg_split['metric'].isin(HEADLINE_METRICS)].copy()
    refine_split['condition_label'] = refine_split['cde_format'].astype(str) + ' + ' + refine_split['placeholder_policy'].fillna('NA').astype(str)

    refine_val_per_cell = flatten_metric_pivot(
        refine_split[refine_split['split'].astype(str) == 'val'],
        index_cols=['cell_label', 'query_variant', 'recipe', 'cde_format', 'placeholder_policy', 'condition_label'],
        metric_order=HEADLINE_METRICS,
    )
    condition_rank = {f'{fmt} + {pol}': i for i, (fmt, pol) in enumerate(CONDITION_ORDER)}
    if not refine_val_per_cell.empty:
        refine_val_per_cell['_cond_rank'] = refine_val_per_cell['condition_label'].map(condition_rank)
        refine_val_per_cell = refine_val_per_cell.sort_values(['query_variant', '_cond_rank']).drop(columns=['_cond_rank']).reset_index(drop=True)

    refine_runs_seed_condition = collapse_cells_per_seed(
        all_rows[all_rows['metric'].isin(HEADLINE_METRICS)].copy(),
        group_cols_without_seed_and_cell=['stage_kind', 'model_name', 'model_label', 'placeholder_policy', 'cde_format', 'split', 'metric'],
    )
    refine_condition_summary = aggregate_over_seeds(
        refine_runs_seed_condition,
        group_cols=['stage_kind', 'model_name', 'model_label', 'placeholder_policy', 'cde_format', 'split', 'metric'],
    )
    refine_condition_summary['condition_label'] = refine_condition_summary['cde_format'].astype(str) + ' + ' + refine_condition_summary['placeholder_policy'].fillna('NA').astype(str)

    refine_condition_val = flatten_metric_pivot(
        refine_condition_summary[refine_condition_summary['split'].astype(str) == 'val'],
        index_cols=['model_label', 'cde_format', 'placeholder_policy', 'condition_label'],
        metric_order=HEADLINE_METRICS,
    )
    if not refine_condition_val.empty:
        refine_condition_val['_cond_rank'] = refine_condition_val['condition_label'].map(condition_rank)
        refine_condition_val = refine_condition_val.sort_values(['recall@5_mean', 'mrr@100_mean', '_cond_rank'], ascending=[False, False, True]).drop(columns=['_cond_rank']).reset_index(drop=True)
    best_refinement_condition_val = refine_condition_val.head(1).copy()

    anchor = refine_split[
        (refine_split['cde_format'].astype(str) == 'labeled')
        & (refine_split['placeholder_policy'].astype(str) == 'placeholder')
    ][['cell_label', 'split', 'metric', 'mean_value']].rename(columns={'mean_value': 'anchor_mean_value'})
    refine_delta_summary = pd.DataFrame()
    refine_delta_val = pd.DataFrame()
    if not anchor.empty:
        refine_delta_vs_anchor = refine_split.merge(anchor, on=['cell_label', 'split', 'metric'], how='left')
        refine_delta_vs_anchor['delta_vs_anchor'] = refine_delta_vs_anchor['mean_value'] - refine_delta_vs_anchor['anchor_mean_value']
        refine_delta_summary = (
            refine_delta_vs_anchor.groupby(['cde_format', 'placeholder_policy', 'split', 'metric'], dropna=False)['delta_vs_anchor']
            .mean()
            .reset_index()
        )
        refine_delta_val = (
            refine_delta_summary[refine_delta_summary['split'].astype(str) == 'val']
            .pivot(index=['cde_format', 'placeholder_policy'], columns='metric', values='delta_vs_anchor')
            .reset_index()
        )
        refine_delta_val = refine_delta_val.rename(columns={c: f'{c}_delta_vs_anchor' for c in refine_delta_val.columns if c not in {'cde_format', 'placeholder_policy'}})

    best_keys = best_refinement_condition_val[['cde_format', 'placeholder_policy']].copy()
    refine_best_condition = refine_split.merge(best_keys, on=['cde_format', 'placeholder_policy'], how='inner') if not best_keys.empty else pd.DataFrame()
    split_order = [s for s in _union_eval_splits(refinement_config_paths) if s]
    report_split_order = [s for s in split_order if s != 'val' and s in refine_best_condition['split'].astype(str).tolist()] if not refine_best_condition.empty else []

    refine_best_condition_reports = flatten_metric_pivot(
        refine_best_condition[refine_best_condition['split'].isin(report_split_order)].rename(columns={'split': 'metric_split'}).assign(metric=lambda d: d['metric'].astype(str) + '__' + d['metric_split'].astype(str)),
        index_cols=['condition_label'],
        metric_order=[f'{PRIMARY_METRIC}__{s}' for s in report_split_order] + [f'{SECONDARY_METRIC}__{s}' for s in report_split_order],
    ) if not refine_best_condition.empty else pd.DataFrame()

    if not refine_best_condition.empty and 'test' in refine_best_condition['split'].astype(str).tolist():
        refine_best_condition_test_per_cell = flatten_metric_pivot(
            refine_best_condition[refine_best_condition['split'].astype(str) == 'test'],
            index_cols=['cell_label', 'query_variant', 'recipe', 'condition_label'],
            metric_order=HEADLINE_METRICS,
        )
    else:
        refine_best_condition_test_per_cell = pd.DataFrame()

    if not refine_delta_summary.empty and report_split_order:
        refine_delta_report = (
            refine_delta_summary[refine_delta_summary['split'].isin(report_split_order)]
            .pivot(index=['split', 'cde_format', 'placeholder_policy'], columns='metric', values='delta_vs_anchor')
            .reset_index()
        )
        refine_delta_report = refine_delta_report.rename(columns={c: f'{c}_delta_vs_anchor' for c in refine_delta_report.columns if c not in {'split', 'cde_format', 'placeholder_policy'}})
    else:
        refine_delta_report = pd.DataFrame()

    heat_val = refine_condition_summary[
        (refine_condition_summary['split'].astype(str) == 'val')
        & (refine_condition_summary['metric'].astype(str) == PRIMARY_METRIC)
    ].pivot(index='placeholder_policy', columns='cde_format', values='mean_value').reindex(index=['placeholder', 'omit'], columns=['labeled', 'raw'])

    plot_split = 'test' if 'test' in refine_condition_summary['split'].astype(str).tolist() else (report_split_order[0] if report_split_order else None)
    heat_report = None
    if plot_split is not None:
        heat_report = refine_condition_summary[
            (refine_condition_summary['split'].astype(str) == plot_split)
            & (refine_condition_summary['metric'].astype(str) == PRIMARY_METRIC)
        ].pivot(index='placeholder_policy', columns='cde_format', values='mean_value').reindex(index=['placeholder', 'omit'], columns=['labeled', 'raw'])

    return {
        'all_rows': all_rows,
        'agg_split': agg_split,
        'refine_val_per_cell': refine_val_per_cell,
        'refine_condition_summary': refine_condition_summary,
        'refine_condition_val': refine_condition_val,
        'best_refinement_condition_val': best_refinement_condition_val,
        'refine_delta_val': refine_delta_val,
        'refine_best_condition_reports': refine_best_condition_reports,
        'refine_best_condition_test_per_cell': refine_best_condition_test_per_cell,
        'refine_delta_report': refine_delta_report,
        'heat_validation': heat_val,
        'heat_report': heat_report,
        'report_plot_split': plot_split,
        'report_split_order': report_split_order,
    }


def compute_step9_data(*, step9_stage_root: Path, step9_config_path: Path) -> Dict[str, Any]:
    all_rows = _collect_stage_split_metrics(step9_stage_root)
    if all_rows.empty:
        raise ValueError('No SapBERT pooling rows found in Step 9 outputs.')
    all_rows['stage'] = 'sapbert_pooling'

    agg_split = aggregate_over_seeds(
        all_rows,
        group_cols=['stage', 'model_name', 'model_label', 'pooling_variant', 'recipe', 'query_variant', 'cde_format', 'rerank_mode', 'rerank_tag', 'placeholder_policy', 'cell_label', 'split', 'metric'],
    )
    pool_split = agg_split[agg_split['metric'].isin(HEADLINE_METRICS)].copy()

    pool_val_per_cell = flatten_metric_pivot(
        pool_split[pool_split['split'].astype(str) == 'val'].rename(columns={'pooling_variant': 'variant'}).assign(metric=lambda d: d['metric'].astype(str) + '__' + d['variant'].astype(str)),
        index_cols=['cell_label', 'query_variant', 'recipe'],
        metric_order=[
            f'{PRIMARY_METRIC}__mean', f'{PRIMARY_METRIC}__cls',
            f'{SECONDARY_METRIC}__mean', f'{SECONDARY_METRIC}__cls',
            'recall@10__mean', 'recall@10__cls',
            'top1_accuracy__mean', 'top1_accuracy__cls',
        ],
    )
    for metric in [PRIMARY_METRIC, SECONDARY_METRIC, 'recall@10', 'top1_accuracy']:
        mean_col = f'{metric}__mean_mean'
        cls_col = f'{metric}__cls_mean'
        if mean_col in pool_val_per_cell.columns and cls_col in pool_val_per_cell.columns:
            pool_val_per_cell[f'{metric}_mean_minus_cls'] = pool_val_per_cell[mean_col] - pool_val_per_cell[cls_col]

    pool_seed_summary = collapse_cells_per_seed(
        all_rows[all_rows['metric'].isin(HEADLINE_METRICS)].copy(),
        group_cols_without_seed_and_cell=['stage', 'model_name', 'model_label', 'pooling_variant', 'split', 'metric'],
    )
    pool_condition_summary = aggregate_over_seeds(
        pool_seed_summary,
        group_cols=['stage', 'model_name', 'model_label', 'pooling_variant', 'split', 'metric'],
    )
    pool_condition_val = flatten_metric_pivot(
        pool_condition_summary[pool_condition_summary['split'].astype(str) == 'val'],
        index_cols=['model_label', 'pooling_variant'],
        metric_order=HEADLINE_METRICS,
    )
    if not pool_condition_val.empty:
        order_map = {name: idx for idx, name in enumerate(POOLING_ORDER)}
        pool_condition_val['_variant_rank'] = pool_condition_val['pooling_variant'].map(order_map)
        pool_condition_val = pool_condition_val.sort_values(['recall@5_mean', 'mrr@100_mean', '_variant_rank'], ascending=[False, False, True]).drop(columns=['_variant_rank']).reset_index(drop=True)
    best_pooling_variant_val = pool_condition_val.head(1).copy()

    pool_delta_split = (
        pool_condition_summary[pool_condition_summary['metric'].isin([PRIMARY_METRIC, SECONDARY_METRIC])]
        .pivot_table(index=['split', 'metric'], columns='pooling_variant', values='mean_value', aggfunc='first')
        .reset_index()
    )
    if {'mean', 'cls'}.issubset(pool_delta_split.columns):
        pool_delta_split['mean_minus_cls'] = pool_delta_split['mean'] - pool_delta_split['cls']

    best_keys = best_pooling_variant_val[['pooling_variant']].copy()
    pool_best_variant = pool_split.merge(best_keys, on=['pooling_variant'], how='inner') if not best_keys.empty else pd.DataFrame()
    split_order = list(_load_baseline_cfg(step9_config_path).get('eval_splits') or [])
    report_split_order = [s for s in split_order if s != 'val' and s in pool_best_variant['split'].astype(str).tolist()] if not pool_best_variant.empty else []

    pool_best_variant_reports = flatten_metric_pivot(
        pool_best_variant[pool_best_variant['split'].isin(report_split_order)].rename(columns={'split': 'metric_split'}).assign(metric=lambda d: d['metric'].astype(str) + '__' + d['metric_split'].astype(str)),
        index_cols=['pooling_variant'],
        metric_order=[f'{PRIMARY_METRIC}__{s}' for s in report_split_order] + [f'{SECONDARY_METRIC}__{s}' for s in report_split_order],
    ) if not pool_best_variant.empty else pd.DataFrame()

    if not pool_best_variant.empty and 'test' in pool_best_variant['split'].astype(str).tolist():
        pool_best_variant_test_per_cell = flatten_metric_pivot(
            pool_best_variant[pool_best_variant['split'].astype(str) == 'test'],
            index_cols=['cell_label', 'query_variant', 'recipe', 'pooling_variant'],
            metric_order=HEADLINE_METRICS,
        )
    else:
        pool_best_variant_test_per_cell = pd.DataFrame()

    validation_plot_df = pool_val_per_cell[['cell_label', f'{PRIMARY_METRIC}__mean_mean', f'{PRIMARY_METRIC}__cls_mean']].copy() if not pool_val_per_cell.empty and f'{PRIMARY_METRIC}__mean_mean' in pool_val_per_cell.columns and f'{PRIMARY_METRIC}__cls_mean' in pool_val_per_cell.columns else pd.DataFrame()
    plot_split = 'test' if 'test' in pool_split['split'].astype(str).tolist() else (report_split_order[0] if report_split_order else None)
    report_plot_df = pd.DataFrame()
    if plot_split is not None:
        report_plot_df = flatten_metric_pivot(
            pool_split[pool_split['split'].astype(str) == plot_split].rename(columns={'pooling_variant': 'variant'}).assign(metric=lambda d: d['metric'].astype(str) + '__' + d['variant'].astype(str)),
            index_cols=['cell_label', 'query_variant', 'recipe'],
            metric_order=[
                f'{PRIMARY_METRIC}__mean', f'{PRIMARY_METRIC}__cls',
                f'{SECONDARY_METRIC}__mean', f'{SECONDARY_METRIC}__cls',
                'recall@10__mean', 'recall@10__cls',
                'top1_accuracy__mean', 'top1_accuracy__cls',
            ],
        )
        if report_plot_df.empty or f'{PRIMARY_METRIC}__mean_mean' not in report_plot_df.columns or f'{PRIMARY_METRIC}__cls_mean' not in report_plot_df.columns:
            report_plot_df = pd.DataFrame()

    return {
        'all_rows': all_rows,
        'agg_split': agg_split,
        'pool_val_per_cell': pool_val_per_cell,
        'pool_condition_summary': pool_condition_summary,
        'pool_condition_val': pool_condition_val,
        'best_pooling_variant_val': best_pooling_variant_val,
        'pool_delta_split': pool_delta_split,
        'pool_best_variant_reports': pool_best_variant_reports,
        'pool_best_variant_test_per_cell': pool_best_variant_test_per_cell,
        'validation_plot_df': validation_plot_df,
        'report_plot_df': report_plot_df,
        'report_plot_split': plot_split,
        'report_split_order': report_split_order,
    }


__all__ = [
    'CONDITION_ORDER',
    'HEADLINE_METRICS',
    'POOLING_ORDER',
    'PRIMARY_METRIC',
    'SECONDARY_METRIC',
    'TOP4_QUERY_VARIANT_ORDER',
    'aggregate_over_seeds',
    'collapse_cells_per_seed',
    'compute_step6_data',
    'compute_step8_data',
    'compute_step9_data',
    'flatten_metric_pivot',
    'query_display_name',
    'recipe_display_name',
    'representation_display_name',
]
