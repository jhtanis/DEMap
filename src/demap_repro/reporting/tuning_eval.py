from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from demap_repro.reporting.publication_figures import (
    loss_display_name,
    model_display_name,
    representation_display_name,
    split_display_name,
)
from demap_repro.biencoder.phase1_screening import build_seed_coverage_table, collect_phase1_runs, select_representative_run
from demap_repro.biencoder.phase2_collect import DEFAULT_PHASE2_SEEDS, PHASE2_GROUP_COLS, collect_phase2_runs
from demap_repro.biencoder.selection_rules import DEFAULT_TIE_MARGIN, order_phase_tuning_configs

PHASE1_STAGE_TAG = 'paper_step10_finetune_phase1'
PHASE2_STAGE_TAG = 'paper_step13_finetune_phase2_bf16'
DEFAULT_PHASE1_SEEDS = [0, 1]

PHASE1_GROUP_COLS = [
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
]

# Canonical reporting eval set — single source of truth in
# configs/evaluation/canonical_eval_datasets.yaml (test, cctg, oid_alt, cdash,
# gdc_combined, cimac_v2). Falls back to the legacy list if the registry is absent.
try:
    from demap_repro.evaluation.canonical_datasets import canonical_split_names as _canonical_split_names
    DEFAULT_REPORT_SPLITS = list(_canonical_split_names())
except Exception:
    DEFAULT_REPORT_SPLITS = [
        'test',
        'external_holdout_org',
        'external_holdout_standard',
        'external_holdout_refslice',
        'external_holdout_gdc_altnames',
        'external_holdout_gdc_questiontext',
    ]
DEFAULT_REPORT_METRICS = ['recall@5', 'mrr@100']
HOLDOUT_SPLITS = [x for x in DEFAULT_REPORT_SPLITS if x != 'test']

STRATEGY_DISPLAY = {
    '': 'None',
    'none': 'None',
    'hard_top25': 'Hard',
    'hard_1_25': 'Hard',
    'semihard_1_50': 'Semi-hard',
    'semihard': 'Semi-hard',
    'semihard_band': 'Semi-hard',
    'semihard_tight_10_50': 'Semi-hard tight',
    'semihard_tight': 'Semi-hard tight',
    'curr_50_200__25_100__1_25': 'Curriculum',
    'curr': 'Curriculum',
    'hardcurr_1_50__1_25__1_15': 'Hard curriculum',
    'hardcurr': 'Hard curriculum',
}


@dataclass(frozen=True)
class PhaseSpec:
    stage_tag: str
    phase_label: str
    group_cols: list[str]


PHASE_SPECS: dict[str, PhaseSpec] = {
    PHASE1_STAGE_TAG: PhaseSpec(
        stage_tag=PHASE1_STAGE_TAG,
        phase_label='Phase 1',
        group_cols=list(PHASE1_GROUP_COLS),
    ),
    PHASE2_STAGE_TAG: PhaseSpec(
        stage_tag=PHASE2_STAGE_TAG,
        phase_label='Phase 2',
        group_cols=list(PHASE2_GROUP_COLS),
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float('nan')


def _strategy_key(record: Mapping[str, Any]) -> str:
    mining_strategy = str(record.get('mining_strategy') or '').strip()
    if mining_strategy:
        return mining_strategy
    return str(record.get('strategy') or '').strip()


def strategy_display_name(value: object) -> str:
    key = str(value or '').strip().lower()
    if key in STRATEGY_DISPLAY:
        return STRATEGY_DISPLAY[key]
    return str(value or '')


def _metrics_from_payload(
    payload: Mapping[str, Any],
    *,
    metrics: Sequence[str],
) -> dict[str, float]:
    rows = payload.get('metrics_by_split') if isinstance(payload, Mapping) else {}
    out: dict[str, float] = {}
    if not isinstance(rows, Mapping):
        return out
    for split, vals in rows.items():
        if not isinstance(vals, Mapping):
            continue
        for metric in metrics:
            if metric in vals:
                out[f'{split}__{metric}'] = _safe_float(vals.get(metric))
    return out


def _split_metric_mean_col(split: str, metric: str) -> str:
    return f'{split}__{metric}__mean'


def _split_metric_std_col(split: str, metric: str) -> str:
    return f'{split}__{metric}__std'


def _mean_from_cols(df: pd.DataFrame, cols: Sequence[str], out_col: str) -> pd.DataFrame:
    out = df.copy()
    keep = [c for c in cols if c in out.columns]
    if keep:
        out[out_col] = out[keep].mean(axis=1, skipna=True)
    else:
        out[out_col] = float('nan')
    return out


def _collect_validation_runs(stage_root: Path, phase_tag: str) -> pd.DataFrame:
    if phase_tag == PHASE1_STAGE_TAG:
        return collect_phase1_runs(stage_root)
    if phase_tag == PHASE2_STAGE_TAG:
        return collect_phase2_runs(stage_root)
    raise ValueError(f'Unsupported phase tag: {phase_tag}')


def collect_phase_eval_runs(
    stage_root: str | Path,
    *,
    phase_tag: str,
    report_splits: Sequence[str] = DEFAULT_REPORT_SPLITS,
    report_metrics: Sequence[str] = DEFAULT_REPORT_METRICS,
) -> pd.DataFrame:
    """Collect run-level validation metadata from ``runs`` and report metrics from ``eval``.

    The join key is the run directory name (``run_id``), matching:
      ``.../runs/<run_id>/run_config.json``
      ``.../eval/<run_id>/metrics.json``
    """
    stage_root = Path(stage_root)
    spec = PHASE_SPECS[phase_tag]
    runs = _collect_validation_runs(stage_root, phase_tag=phase_tag).copy()
    if runs.empty:
        return runs

    rows: list[dict[str, Any]] = []
    for _, row in runs.iterrows():
        rec = row.to_dict()
        run_dir = Path(str(rec['run_dir']))
        run_id = run_dir.name
        model_slug = run_dir.parent.parent.name
        eval_dir = stage_root / model_slug / 'eval' / run_id
        eval_metrics_path = eval_dir / 'metrics.json'
        eval_payload = _read_json(eval_metrics_path) if eval_metrics_path.exists() else {}
        rec['phase'] = phase_tag
        rec['phase_label'] = spec.phase_label
        rec['run_id'] = run_id
        rec['model_slug'] = model_slug
        rec['eval_dir'] = str(eval_dir)
        rec['eval_metrics_path'] = str(eval_metrics_path)
        rec['has_eval_metrics'] = bool(eval_metrics_path.exists())
        rec.update(_metrics_from_payload(eval_payload, metrics=report_metrics))
        rows.append(rec)

    out = pd.DataFrame(rows)
    out['model'] = out['base_model_id'].map(model_display_name)
    out['representation'] = out.apply(lambda r: representation_display_name(r['query_variant'], r['recipe']), axis=1)
    out['loss_display'] = out['loss'].map(loss_display_name)
    out['strategy_key'] = out.apply(_strategy_key, axis=1)
    out['strategy_display'] = out['strategy_key'].map(strategy_display_name)

    holdout_r5_cols = [f'{split}__recall@5' for split in HOLDOUT_SPLITS if f'{split}__recall@5' in out.columns]
    holdout_mrr_cols = [f'{split}__mrr@100' for split in HOLDOUT_SPLITS if f'{split}__mrr@100' in out.columns]
    out = _mean_from_cols(out, holdout_r5_cols, 'holdout_mean_recall5')
    out = _mean_from_cols(out, holdout_mrr_cols, 'holdout_mean_mrr100')
    if 'test__recall@5' not in out.columns:
        out['test__recall@5'] = float('nan')
    if 'test__mrr@100' not in out.columns:
        out['test__mrr@100'] = float('nan')
    return out


def summarize_phase_eval_runs(
    runs_df: pd.DataFrame,
    *,
    phase_tag: str,
    report_splits: Sequence[str] = DEFAULT_REPORT_SPLITS,
    report_metrics: Sequence[str] = DEFAULT_REPORT_METRICS,
    tie_margin: float = DEFAULT_TIE_MARGIN,
    expected_seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    spec = PHASE_SPECS[phase_tag]
    if runs_df.empty:
        return runs_df.copy()

    agg_kwargs: dict[str, tuple[str, str]] = {
        'n_runs': ('run_id', 'count'),
        'n_seeds': ('seed', 'nunique'),
        'n_eval_runs': ('has_eval_metrics', 'sum'),
        'val_recall5_mean': ('val_recall5', 'mean'),
        'val_recall5_std': ('val_recall5', 'std'),
        'val_mrr100_mean': ('val_mrr100', 'mean'),
        'val_mrr100_std': ('val_mrr100', 'std'),
        'run_ids': ('run_id', lambda s: '|'.join(sorted(str(x) for x in s))),
    }
    for split in report_splits:
        for metric in report_metrics:
            col = f'{split}__{metric}'
            if col not in runs_df.columns:
                continue
            agg_kwargs[_split_metric_mean_col(split, metric)] = (col, 'mean')
            agg_kwargs[_split_metric_std_col(split, metric)] = (col, 'std')

    summary = runs_df.groupby(spec.group_cols, dropna=False).agg(**agg_kwargs).reset_index()
    coverage = build_seed_coverage_table(
        runs_df,
        group_cols=spec.group_cols,
        expected_seeds=list(expected_seeds or []),
    )
    if not coverage.empty:
        summary = summary.merge(coverage, on=spec.group_cols, how='left')
    for col in [
        'val_recall5_std',
        'val_mrr100_std',
        *[c for c in summary.columns if c.endswith('__std')],
    ]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors='coerce').fillna(0.0)

    if 'n_seeds_observed' not in summary.columns:
        summary['n_seeds_observed'] = pd.to_numeric(summary['n_seeds'], errors='coerce').fillna(0).astype(int)
    else:
        summary['n_seeds_observed'] = pd.to_numeric(summary['n_seeds_observed'], errors='coerce').fillna(0).astype(int)
    if 'observed_seeds' not in summary.columns:
        summary['observed_seeds'] = ''
    if 'expected_seeds' not in summary.columns:
        summary['expected_seeds'] = ''
    if 'missing_seeds' not in summary.columns:
        summary['missing_seeds'] = ''
    if 'n_missing_seeds' not in summary.columns:
        summary['n_missing_seeds'] = 0
    if 'incomplete_seed_coverage' not in summary.columns:
        summary['incomplete_seed_coverage'] = False
    summary['has_any_eval_metrics'] = pd.to_numeric(summary['n_eval_runs'], errors='coerce').fillna(0).gt(0)

    summary['phase'] = phase_tag
    summary['phase_label'] = spec.phase_label
    summary['phase_order'] = 1 if phase_tag == PHASE1_STAGE_TAG else 2
    summary['model'] = summary['base_model_id'].map(model_display_name)
    summary['representation'] = summary.apply(lambda r: representation_display_name(r['query_variant'], r['recipe']), axis=1)
    summary['loss_display'] = summary['loss'].map(loss_display_name)
    if 'strategy' not in summary.columns:
        summary['strategy'] = ''
    if 'mining_strategy' not in summary.columns:
        summary['mining_strategy'] = summary['strategy'].map(str)
    if 'nneg' not in summary.columns:
        summary['nneg'] = 0
    summary['strategy_key'] = summary.apply(_strategy_key, axis=1)
    summary['strategy_display'] = summary['strategy_key'].map(strategy_display_name)

    summary = _mean_from_cols(
        summary,
        [_split_metric_mean_col(split, 'recall@5') for split in HOLDOUT_SPLITS],
        'holdout_mean_recall5_mean',
    )
    summary = _mean_from_cols(
        summary,
        [_split_metric_std_col(split, 'recall@5') for split in HOLDOUT_SPLITS],
        'holdout_mean_recall5_std',
    )
    summary = _mean_from_cols(
        summary,
        [_split_metric_mean_col(split, 'mrr@100') for split in HOLDOUT_SPLITS],
        'holdout_mean_mrr100_mean',
    )
    summary = _mean_from_cols(
        summary,
        [_split_metric_std_col(split, 'mrr@100') for split in HOLDOUT_SPLITS],
        'holdout_mean_mrr100_std',
    )

    ranked = order_phase_tuning_configs(
        summary,
        primary_col='val_recall5_mean',
        secondary_col='val_mrr100_mean',
        primary_std_col='val_recall5_std',
        secondary_std_col='val_mrr100_std',
        tie_margin=float(tie_margin),
        fallback_cols=spec.group_cols,
        fallback_ascending=[True] * len(spec.group_cols),
        group_cols=['base_model_id'],
    ).reset_index(drop=True)
    return ranked


def attach_representative_runs(
    ranked_summary: pd.DataFrame,
    runs_df: pd.DataFrame,
    *,
    phase_tag: str,
) -> pd.DataFrame:
    spec = PHASE_SPECS[phase_tag]
    if ranked_summary.empty:
        return ranked_summary.copy()
    rows: list[dict[str, Any]] = []
    for _, rec in ranked_summary.iterrows():
        out = rec.to_dict()
        rep = select_representative_run(runs_df, rec, match_cols=spec.group_cols)
        out['representative_run_id'] = str(rep['run_id'])
        out['representative_seed'] = int(rep['seed']) if pd.notna(rep.get('seed')) else np.nan
        out['representative_run_dir'] = str(rep['run_dir'])
        out['representative_eval_dir'] = str(rep['eval_dir'])
        out['representative_has_eval_metrics'] = bool(rep.get('has_eval_metrics'))
        rows.append(out)
    return pd.DataFrame(rows)


def _rows_with_eval_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if 'has_any_eval_metrics' in df.columns:
        mask = df['has_any_eval_metrics'].fillna(False).astype(bool)
        return df[mask].reset_index(drop=True)
    if 'n_eval_runs' in df.columns:
        mask = pd.to_numeric(df['n_eval_runs'], errors='coerce').fillna(0).gt(0)
        return df[mask].reset_index(drop=True)
    return df.copy()


def build_phase_missingness_summary(
    ranked_summary: pd.DataFrame,
    *,
    phase_tag: str,
) -> pd.DataFrame:
    if ranked_summary.empty or 'incomplete_seed_coverage' not in ranked_summary.columns:
        return pd.DataFrame()
    work = ranked_summary[ranked_summary['incomplete_seed_coverage'].fillna(False)].copy()
    if work.empty:
        return pd.DataFrame()

    group_cols = ['model']
    if phase_tag == PHASE2_STAGE_TAG and 'strategy_display' in work.columns:
        group_cols = ['model', 'strategy_display']

    def _join_unique(values: pd.Series) -> str:
        keep = sorted({str(x) for x in values if str(x)})
        return '|'.join(keep)

    summary = (
        work.groupby(group_cols, dropna=False)
        .agg(
            n_incomplete_configs=('incomplete_seed_coverage', 'sum'),
            n_missing_seed_slots=('n_missing_seeds', 'sum'),
            observed_seed_patterns=('observed_seeds', _join_unique),
            missing_seed_patterns=('missing_seeds', _join_unique),
        )
        .reset_index()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    summary['phase'] = PHASE_SPECS[phase_tag].phase_label
    return summary


CROSS_PHASE_FALLBACK_COLS = [
    'phase_order',
    'phase',
    'model_name',
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
    'strategy',
    'mining_strategy',
    'nneg',
]


def select_best_across_phases(
    phase_winners: pd.DataFrame,
    *,
    tie_margin: float = DEFAULT_TIE_MARGIN,
) -> pd.DataFrame:
    if phase_winners.empty:
        return phase_winners.copy()
    work = phase_winners.copy()
    defaults: dict[str, Any] = {
        'phase_order': 99,
        'phase': '',
        'model_name': '',
        'query_variant': '',
        'recipe': '',
        'cde_format': '',
        'rerank_mode': '',
        'hybrid_alpha': float('nan'),
        'placeholder_strategy': '',
        'loss': '',
        'lr': float('nan'),
        'batch_size': float('nan'),
        'temperature': float('nan'),
        'epochs': float('nan'),
        'strategy': '',
        'mining_strategy': '',
        'nneg': float('nan'),
    }
    for col, default in defaults.items():
        if col not in work.columns:
            work[col] = default
        work[col] = work[col].fillna(default)

    ranked = order_phase_tuning_configs(
        work,
        primary_col='val_recall5_mean',
        secondary_col='val_mrr100_mean',
        primary_std_col='val_recall5_std',
        secondary_std_col='val_mrr100_std',
        tie_margin=float(tie_margin),
        fallback_cols=CROSS_PHASE_FALLBACK_COLS,
        fallback_ascending=[True] * len(CROSS_PHASE_FALLBACK_COLS),
        group_cols=['base_model_id'],
    ).reset_index(drop=True)
    return ranked.groupby('base_model_id', dropna=False, sort=False).head(1).reset_index(drop=True)


def _sort_models(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'model' not in df.columns:
        return df.copy()
    return df.sort_values(['model']).reset_index(drop=True)


def build_eval_metric_table(
    winners_df: pd.DataFrame,
    *,
    metric: str = 'recall@5',
    report_splits: Sequence[str] = DEFAULT_REPORT_SPLITS,
    include_phase: bool = True,
    include_strategy: bool = True,
) -> pd.DataFrame:
    if winners_df.empty:
        cols = ['model']
        if include_phase:
            cols.append('phase')
        if include_strategy:
            cols.append('strategy')
        cols.extend(split_display_name(split) for split in report_splits)
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, Any]] = []
    for _, rec in winners_df.iterrows():
        row: dict[str, Any] = {'model': rec['model']}
        if include_phase:
            row['phase'] = rec['phase_label']
        if include_strategy:
            row['strategy'] = rec['strategy_display']
        for split in report_splits:
            row[split_display_name(split)] = rec.get(_split_metric_mean_col(split, metric), float('nan'))
        rows.append(row)
    return _sort_models(pd.DataFrame(rows))


def build_selection_summary_long(
    *,
    phase1_winners: pd.DataFrame,
    phase2_winners: pd.DataFrame,
    best_overall_by_model: pd.DataFrame,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    keep_cols = [
        'model',
        'base_model_id',
        'selection_slot',
        'phase_label',
        'representation',
        'loss_display',
        'strategy_display',
        'lr',
        'batch_size',
        'temperature',
        'epochs',
        'val_recall5_mean',
        'val_mrr100_mean',
        'n_seeds_observed',
        'observed_seeds',
        'incomplete_seed_coverage',
        'has_any_eval_metrics',
        'representative_run_id',
        'run_ids',
        'test__recall@5__mean',
        'holdout_mean_recall5_mean',
    ]
    for slot, df in [
        ('best_phase1', phase1_winners),
        ('best_phase2', phase2_winners),
        ('best_overall', best_overall_by_model),
    ]:
        if df.empty:
            continue
        sub = df.copy()
        sub['selection_slot'] = slot
        pieces.append(sub[keep_cols].copy())
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, axis=0, ignore_index=True)
    out = out.rename(
        columns={
            'phase_label': 'phase',
            'loss_display': 'loss',
            'strategy_display': 'strategy',
            'val_recall5_mean': 'validation_recall5',
            'val_mrr100_mean': 'validation_mrr100',
            'test__recall@5__mean': 'test_recall5',
            'has_any_eval_metrics': 'has_eval_metrics',
        }
    )
    return out.sort_values(['model', 'selection_slot']).reset_index(drop=True)


def build_selection_companion_by_model(selection_long: pd.DataFrame) -> pd.DataFrame:
    if selection_long.empty:
        return pd.DataFrame()
    keep_cols = [
        'phase',
        'representation',
        'loss',
        'strategy',
        'validation_recall5',
        'validation_mrr100',
        'n_seeds_observed',
        'observed_seeds',
        'incomplete_seed_coverage',
        'has_eval_metrics',
        'test_recall5',
        'holdout_mean_recall5_mean',
        'representative_run_id',
    ]
    wide = selection_long.set_index(['model', 'selection_slot'])[keep_cols].unstack('selection_slot')
    wide.columns = [f'{slot}__{col}' for col, slot in wide.columns]
    return wide.reset_index().sort_values('model').reset_index(drop=True)


def build_tuning_evaluation_bundle(
    repo_root: str | Path,
    *,
    phase1_stage_tag: str = PHASE1_STAGE_TAG,
    phase2_stage_tag: str = PHASE2_STAGE_TAG,
    report_splits: Sequence[str] = DEFAULT_REPORT_SPLITS,
    report_metrics: Sequence[str] = DEFAULT_REPORT_METRICS,
    tie_margin: float = DEFAULT_TIE_MARGIN,
    available_results: bool = False,
    phase1_expected_seeds: Sequence[int] = tuple(DEFAULT_PHASE1_SEEDS),
    phase2_expected_seeds: Sequence[int] = tuple(DEFAULT_PHASE2_SEEDS),
    drop_missing_eval_from_final_tables: bool = True,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    phase_roots = {
        phase1_stage_tag: repo_root / 'artifacts' / phase1_stage_tag,
        phase2_stage_tag: repo_root / 'artifacts' / phase2_stage_tag,
    }
    expected_seeds_by_phase = {
        phase1_stage_tag: list(phase1_expected_seeds),
        phase2_stage_tag: list(phase2_expected_seeds),
    }

    phase_results: dict[str, dict[str, pd.DataFrame]] = {}
    phase_winner_frames: list[pd.DataFrame] = []
    for phase_tag, stage_root in phase_roots.items():
        runs = collect_phase_eval_runs(
            stage_root,
            phase_tag=phase_tag,
            report_splits=report_splits,
            report_metrics=report_metrics,
        )
        ranked = summarize_phase_eval_runs(
            runs,
            phase_tag=phase_tag,
            report_splits=report_splits,
            report_metrics=report_metrics,
            tie_margin=float(tie_margin),
            expected_seeds=expected_seeds_by_phase.get(phase_tag, []),
        )
        winners = ranked.groupby('base_model_id', dropna=False, sort=False).head(1).reset_index(drop=True)
        winners = attach_representative_runs(winners, runs, phase_tag=phase_tag)
        winners_for_eval = _rows_with_eval_metrics(winners) if drop_missing_eval_from_final_tables else winners.copy()
        phase_winner_frames.append(winners)
        phase_results[phase_tag] = {
            'runs': runs,
            'ranked_configs': ranked,
            'winners_by_model': winners,
            'winners_with_eval_by_model': winners_for_eval,
            'missingness_summary': build_phase_missingness_summary(ranked, phase_tag=phase_tag),
            'winners_eval_recall5': build_eval_metric_table(winners_for_eval, metric='recall@5', report_splits=report_splits),
            'winners_eval_mrr100': build_eval_metric_table(winners_for_eval, metric='mrr@100', report_splits=report_splits),
        }

    all_phase_winners = pd.concat(phase_winner_frames, axis=0, ignore_index=True) if phase_winner_frames else pd.DataFrame()
    best_overall_by_model = select_best_across_phases(all_phase_winners, tie_margin=float(tie_margin))
    best_overall_for_eval = _rows_with_eval_metrics(best_overall_by_model) if drop_missing_eval_from_final_tables else best_overall_by_model.copy()
    global_best = pd.DataFrame()
    if not all_phase_winners.empty:
        global_ranked = order_phase_tuning_configs(
            all_phase_winners.assign(_global_group='all'),
            primary_col='val_recall5_mean',
            secondary_col='val_mrr100_mean',
            primary_std_col='val_recall5_std',
            secondary_std_col='val_mrr100_std',
            tie_margin=float(tie_margin),
            fallback_cols=CROSS_PHASE_FALLBACK_COLS,
            fallback_ascending=[True] * len(CROSS_PHASE_FALLBACK_COLS),
            group_cols=['_global_group'],
        ).reset_index(drop=True)
        global_best = global_ranked.head(1).drop(columns=['_global_group'], errors='ignore')
    global_best_for_eval = _rows_with_eval_metrics(global_best) if drop_missing_eval_from_final_tables else global_best.copy()

    selection_summary_long = build_selection_summary_long(
        phase1_winners=phase_results.get(phase1_stage_tag, {}).get('winners_by_model', pd.DataFrame()),
        phase2_winners=phase_results.get(phase2_stage_tag, {}).get('winners_by_model', pd.DataFrame()),
        best_overall_by_model=best_overall_by_model,
    )
    selection_companion = build_selection_companion_by_model(selection_summary_long)

    return {
        'available_results': bool(available_results),
        'drop_missing_eval_from_final_tables': bool(drop_missing_eval_from_final_tables),
        'phase_results': phase_results,
        'all_phase_winners_by_model': _sort_models(all_phase_winners),
        'best_overall_by_model': _sort_models(best_overall_by_model),
        'best_overall_eval_recall5': build_eval_metric_table(best_overall_for_eval, metric='recall@5', report_splits=report_splits),
        'best_overall_eval_mrr100': build_eval_metric_table(best_overall_for_eval, metric='mrr@100', report_splits=report_splits),
        'selection_summary_long': selection_summary_long,
        'selection_companion_by_model': selection_companion,
        'global_best_overall': global_best,
        'global_best_eval_recall5': build_eval_metric_table(global_best_for_eval, metric='recall@5', report_splits=report_splits),
        'global_best_eval_mrr100': build_eval_metric_table(global_best_for_eval, metric='mrr@100', report_splits=report_splits),
    }


def plot_test_vs_holdout_scatter(
    winners_df: pd.DataFrame,
    *,
    metric: str = 'recall@5',
    annotate: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 5.0))
    if winners_df.empty:
        ax.set_title('No winners available')
        return ax

    if metric == 'recall@5':
        x_col = 'test__recall@5__mean'
        y_col = 'holdout_mean_recall5_mean'
        xlabel = 'Test Recall@5'
        ylabel = 'Mean holdout Recall@5'
    else:
        x_col = 'test__mrr@100__mean'
        y_col = 'holdout_mean_mrr100_mean'
        xlabel = 'Test MRR@100'
        ylabel = 'Mean holdout MRR@100'

    markers = {'Phase 1': 'o', 'Phase 2': 's'}
    for phase_label, sub in winners_df.groupby('phase_label', dropna=False, sort=False):
        xs = pd.to_numeric(sub[x_col], errors='coerce')
        ys = pd.to_numeric(sub[y_col], errors='coerce')
        ax.scatter(xs, ys, marker=markers.get(str(phase_label), 'o'), label=str(phase_label))
        if annotate:
            for _, rec in sub.iterrows():
                x = rec.get(x_col)
                y = rec.get(y_col)
                if pd.isna(x) or pd.isna(y):
                    continue
                label = str(rec.get('model') or '')
                if str(rec.get('phase_label') or '') == 'Phase 2':
                    strat = str(rec.get('strategy_display') or '')
                    if strat and strat != 'None':
                        label = f'{label} ({strat})'
                ax.annotate(label, (float(x), float(y)), xytext=(4, 4), textcoords='offset points', fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('Test vs mean holdout performance of selected tuning winners')
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_best_overall_holdout_bar(
    best_overall_df: pd.DataFrame,
    *,
    metric: str = 'recall@5',
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8.2, 4.6))
    if best_overall_df.empty:
        ax.set_title('No best-overall rows available')
        return ax
    if metric == 'recall@5':
        value_col = 'holdout_mean_recall5_mean'
        ylabel = 'Mean holdout Recall@5'
    else:
        value_col = 'holdout_mean_mrr100_mean'
        ylabel = 'Mean holdout MRR@100'
    work = best_overall_df[['model', 'phase_label', 'strategy_display', value_col]].copy()
    work = work.sort_values(value_col, ascending=False).reset_index(drop=True)
    xs = np.arange(len(work))
    bars = ax.bar(xs, pd.to_numeric(work[value_col], errors='coerce').astype(float).tolist())
    ax.set_xticks(xs)
    labels = []
    for _, rec in work.iterrows():
        label = str(rec['model'])
        if str(rec['phase_label']) == 'Phase 2' and str(rec['strategy_display']) not in {'', 'None'}:
            label = f"{label}\n{rec['strategy_display']}"
        labels.append(label)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title('Best overall config by model: mean holdout performance')
    for idx, bar in enumerate(bars):
        val = work.iloc[idx][value_col]
        if pd.notna(val):
            ax.text(bar.get_x() + bar.get_width() / 2.0, float(val) + 0.004, f'{float(val):0.3f}', ha='center', va='bottom', fontsize=8)
    return ax


__all__ = [
    'DEFAULT_PHASE1_SEEDS',
    'DEFAULT_REPORT_METRICS',
    'DEFAULT_REPORT_SPLITS',
    'HOLDOUT_SPLITS',
    'PHASE1_GROUP_COLS',
    'PHASE1_STAGE_TAG',
    'PHASE2_STAGE_TAG',
    'STRATEGY_DISPLAY',
    'attach_representative_runs',
    'build_eval_metric_table',
    'build_phase_missingness_summary',
    'build_selection_companion_by_model',
    'build_selection_summary_long',
    'build_tuning_evaluation_bundle',
    'collect_phase_eval_runs',
    'plot_best_overall_holdout_bar',
    'plot_test_vs_holdout_scatter',
    'select_best_across_phases',
    'strategy_display_name',
    'summarize_phase_eval_runs',
]
