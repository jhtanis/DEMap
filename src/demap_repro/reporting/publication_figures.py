from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from demap_repro.biencoder.constants import PAPER_CORE_MODEL_IDS, PAPER_FINALIST_REPRESENTATIONS
from demap_repro.biencoder.phase1_screening import collect_phase1_runs
from demap_repro.reporting.representation_tables import compute_step6_data, compute_step8_data, compute_step9_data
from demap_repro.biencoder.phase2_collect import PHASE2_GROUP_COLS
from demap_repro.biencoder.selection_rules import ATOMIC_RECIPE_COMPLEXITY_RANK
from demap_repro.text.recipes import parse_recipe

QUERY_DISPLAY = {
    'Q1': 'RAW',
    'Q2': 'PREF_PV',
    'Q3': 'RAW_PV',
    'Q4': 'RAW_PV_PH',
}

QUERY_PAPER_ORDER = ['Q1', 'Q3', 'Q4', 'Q2']

CDE_ATOM_DISPLAY = {
    'v1': 'SN',
    'v2a': 'LN',
    'v2b': 'DEF',
    'v3': 'PQT',
    'v4': 'VD',
    'v5': 'PV',
    'v6': 'DEC',
}

MODEL_DISPLAY = {
    'sentence-transformers/all-MiniLM-L6-v2': 'all-MiniLM-L6-v2',
    'sentence-transformers/all-mpnet-base-v2': 'all-mpnet-base-v2',
    'pritamdeka/S-PubMedBert-MS-MARCO': 'S-PubMedBert-MS-MARCO',
    'cambridgeltl/SapBERT-from-PubMedBERT-fulltext': 'SapBERT',
    'kamalkraj/BioSimCSE-BioLinkBERT-BASE': 'BioSimCSE-BioLinkBERT',
    'neuml/pubmedbert-base-embeddings': 'NeuML-PubMedBERT',
    'NeuML/pubmedbert-base-embeddings': 'NeuML-PubMedBERT',
}

MODEL_PAPER_ORDER = list(PAPER_CORE_MODEL_IDS)

LOSS_DISPLAY = {
    'mnrl': 'MNRL',
    'symmetric_mnrl': 'Symmetric_MNRL',
    'cached_mnrl': 'Cached_MNRL',
}

SPLIT_DISPLAY = {
    # Canonical reporting datasets (see configs/evaluation/canonical_eval_datasets.yaml).
    'test': 'Test',
    'cctg': 'CCTG',
    'oid_alt': 'OID ALT',
    'cdash': 'CDASH',
    'gdc_combined': 'GDC',
    'cimac_v2': 'CIMAC v2',
    # Legacy split display names (kept for backward-compat with older run outputs).
    'external_holdout_org': 'Holdout Org',
    'external_holdout_standard': 'Holdout ALT Format',
    'external_holdout_refslice': 'Holdout REF Format',
    'external_holdout_gdc_altnames': 'Holdout GDC ALT',
    'external_holdout_gdc_questiontext': 'Holdout GDC REF',
}
# Overlay canonical display names from the registry (authoritative if present).
try:
    from demap_repro.evaluation.canonical_datasets import canonical_display_map as _canonical_display_map
    SPLIT_DISPLAY.update(_canonical_display_map())
except Exception:
    pass

FINALIST_REPRESENTATIONS = list(PAPER_FINALIST_REPRESENTATIONS)

STEP10B_LOSS_ORDER = ['mnrl', 'symmetric_mnrl']
HEATMAP_CMAP = 'YlOrRd'
PLOT_DPI = 300


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            'figure.dpi': 150,
            'savefig.dpi': PLOT_DPI,
            'font.size': 10,
            'axes.titlesize': 11,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
        }
    )


_set_plot_style()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def representation_display_name(query_variant: object, recipe: object, *, multiline: bool = False) -> str:
    joiner = '\n× ' if multiline else ' × '
    return f"{query_display_name(query_variant)}{joiner}{recipe_display_name(recipe)}"


def loss_display_name(loss_name: object) -> str:
    loss = str(loss_name or '').strip().lower()
    return LOSS_DISPLAY.get(loss, str(loss_name or ''))


def model_display_name(model_id: object) -> str:
    key = str(model_id or '')
    if key in MODEL_DISPLAY:
        return MODEL_DISPLAY[key]
    tail = key.split('/')[-1] if '/' in key else key
    if tail in MODEL_DISPLAY.values():
        return tail
    if 'SapBERT' in tail:
        return 'SapBERT'
    if 'BioSimCSE-BioLinkBERT' in tail:
        return 'BioSimCSE-BioLinkBERT'
    if 'pubmedbert-base-embeddings' in tail or 'neuml/' in key.lower():
        return 'NeuML-PubMedBERT'
    return tail or key


def split_display_name(split_name: object) -> str:
    return SPLIT_DISPLAY.get(str(split_name or ''), str(split_name or ''))


def _model_order_key(model_id: object) -> Tuple[int, str]:
    key = str(model_id or '')
    if key in MODEL_PAPER_ORDER:
        return (MODEL_PAPER_ORDER.index(key), key)
    disp = model_display_name(key)
    for idx, mid in enumerate(MODEL_PAPER_ORDER):
        if model_display_name(mid) == disp:
            return (idx, key)
    return (len(MODEL_PAPER_ORDER), disp)


def recipe_plot_order_key(recipe: object) -> Tuple[int, int, int, Tuple[int, ...], str]:
    try:
        atoms = _expand_recipe_atoms(recipe)
    except Exception:
        text = str(recipe)
        return (99, 99, 99, (99,), text)
    ranks = tuple(int(ATOMIC_RECIPE_COMPLEXITY_RANK.get(atom, 99)) for atom in atoms)
    return (
        len(atoms),
        max(ranks) if ranks else 99,
        sum(ranks),
        ranks,
        str(recipe),
    )


def _format_metric(value: object, *, decimals: int = 3) -> str:
    if value is None:
        return ''
    try:
        f = float(value)
    except Exception:
        return str(value)
    if math.isnan(f):
        return ''
    return f"{f:.{int(decimals)}f}"


def _format_lr(value: object) -> str:
    try:
        x = float(value)
    except Exception:
        return str(value)
    text = f'{x:.0e}'
    return text.replace('e-0', 'e-').replace('e+0', 'e+')


def _format_temperature(value: object) -> str:
    try:
        x = float(value)
    except Exception:
        return str(value)
    text = f'{x:.2f}'
    return text.rstrip('0').rstrip('.')


def _format_generic_value(parameter: str, value: object) -> str:
    if parameter == 'Learning rate':
        return _format_lr(value)
    if parameter == 'Temperature':
        return _format_temperature(value)
    if parameter in {'Batch size', 'Epochs'}:
        try:
            return str(int(value))
        except Exception:
            return str(value)
    return str(value)


def _save_figure(fig: plt.Figure, out_base: Path) -> Dict[str, Path]:
    png_path = out_base.with_suffix('.png')
    pdf_path = out_base.with_suffix('.pdf')
    fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    return {'png': png_path, 'pdf': pdf_path}


def _round_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        lower = str(col).lower()
        if any(tok in lower for tok in ['recall', 'mrr', 'mean', 'std']):
            out[col] = out[col].map(lambda x: _format_metric(x, decimals=3))
        elif 'learning rate' in lower or lower == 'lr':
            out[col] = out[col].map(_format_lr)
        elif 'temperature' in lower:
            out[col] = out[col].map(_format_temperature)
    return out


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    disp = _round_for_display(df)
    headers = [str(c) for c in disp.columns]
    rows = [[str(v) for v in row] for row in disp.fillna('').itertuples(index=False, name=None)]
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    def fmt_row(cells: Sequence[str]) -> str:
        return '| ' + ' | '.join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells)) + ' |'
    sep = '| ' + ' | '.join('-' * widths[idx] for idx in range(len(widths))) + ' |'
    lines = [fmt_row(headers), sep]
    lines.extend(fmt_row(row) for row in rows)
    return '\n'.join(lines) + '\n'


def _write_table_exports(
    df: pd.DataFrame,
    csv_path: Path,
    *,
    md_path: Optional[Path] = None,
    latex_path: Optional[Path] = None,
) -> Dict[str, Path]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    out: Dict[str, Path] = {'csv': csv_path}
    if md_path is not None:
        md_path.write_text(_dataframe_to_markdown(df), encoding='utf-8')
        out['md'] = md_path
    if latex_path is not None:
        disp = _round_for_display(df)
        latex_path.write_text(disp.to_latex(index=False, escape=False), encoding='utf-8')
        out['tex'] = latex_path
    return out


def _annotate_heatmap(ax: plt.Axes, matrix: pd.DataFrame, *, decimals: int = 3, vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    arr = matrix.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    threshold: Optional[float] = None
    if finite.size:
        lo = float(np.nanmin(finite) if vmin is None else vmin)
        hi = float(np.nanmax(finite) if vmax is None else vmax)
        threshold = (lo + hi) / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if pd.isna(val):
                continue
            color = 'white' if threshold is not None and float(val) >= threshold else 'black'
            ax.text(j, i, _format_metric(val, decimals=decimals), ha='center', va='center', color=color)


def _plot_heatmap(
    matrix: pd.DataFrame,
    *,
    out_base: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    annotate: bool = True,
    xtick_rotation: int = 45,
    cmap: str = HEATMAP_CMAP,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Dict[str, Path]:
    if figsize is None:
        figsize = (max(6.0, 0.85 * matrix.shape[1]), max(3.5, 0.75 * matrix.shape[0]))
    fig, ax = plt.subplots(figsize=figsize)
    arr = matrix.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels([str(x) for x in matrix.columns], rotation=xtick_rotation, ha='right')
    ax.set_yticklabels([str(x) for x in matrix.index])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)
    if annotate:
        _annotate_heatmap(ax, matrix, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Recall@5', rotation=90, va='center')
    fig.tight_layout()
    return _save_figure(fig, out_base)


def _hparam_sort_key(parameter: str, value: object) -> Tuple[float, str]:
    try:
        return (float(value), str(value))
    except Exception:
        return (float('inf'), str(value))


def _phase_winner_counts(winners_df: pd.DataFrame) -> pd.DataFrame:
    params = [
        ('Learning rate', 'lr'),
        ('Batch size', 'batch_size'),
        ('Temperature', 'temperature'),
        ('Epochs', 'epochs'),
    ]
    rows: List[Dict[str, Any]] = []
    for parameter, column in params:
        if column not in winners_df.columns:
            continue
        for value, sub in winners_df.groupby(column, dropna=False):
            models = ', '.join(model_display_name(x) for x in sorted(sub['base_model_id'].astype(str).tolist(), key=_model_order_key))
            rows.append(
                {
                    'parameter': parameter,
                    'value': value,
                    'value_label': _format_generic_value(parameter, value),
                    'count': int(sub['base_model_id'].nunique(dropna=False)),
                    'models': models,
                }
            )
    counts = pd.DataFrame(rows)
    if counts.empty:
        return counts
    pieces: List[pd.DataFrame] = []
    for parameter, sub in counts.groupby('parameter', dropna=False, sort=False):
        pieces.append(sub.sort_values('value', key=lambda s: s.map(lambda x: _hparam_sort_key(str(parameter), x))))
    return pd.concat(pieces, axis=0, ignore_index=True)


def _plot_hparam_count_panels(counts: pd.DataFrame, *, out_base: Path, title_prefix: str) -> Dict[str, Path]:
    parameters = ['Learning rate', 'Batch size', 'Temperature', 'Epochs']
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.6), sharey=False)
    for ax, parameter in zip(axes, parameters):
        sub = counts[counts['parameter'] == parameter].copy()
        sub = sub.sort_values('value', key=lambda s: s.map(lambda x: _hparam_sort_key(parameter, x))).reset_index(drop=True)
        if sub.empty:
            ax.set_visible(False)
            continue
        xs = np.arange(len(sub))
        bars = ax.bar(xs, sub['count'].astype(int).tolist())
        ax.set_xticks(xs)
        ax.set_xticklabels(sub['value_label'].tolist(), rotation=45, ha='right')
        ax.set_title(parameter)
        ax.set_ylabel('Count')
        for idx, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.03, str(int(sub.iloc[idx]['count'])), ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, max(1.0, float(sub['count'].max()) + 1.0))
    fig.suptitle(title_prefix)
    fig.tight_layout()
    return _save_figure(fig, out_base)


def _plot_best_by_model(best_df: pd.DataFrame, *, out_base: Path) -> Dict[str, Path]:
    work = best_df.sort_values('validation_recall5', ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.0, 4.4))
    xs = np.arange(len(work))
    bars = ax.bar(xs, work['validation_recall5'].astype(float).tolist())
    ax.set_xticks(xs)
    ax.set_xticklabels(work['model'].tolist(), rotation=30, ha='right')
    ax.set_ylabel('Validation Recall@5')
    ax.set_title('Best Phase 1 validation Recall@5 by model')
    ax.set_ylim(0, max(0.05, float(work['validation_recall5'].max()) + 0.03))
    fig.tight_layout()
    return _save_figure(fig, out_base)


def _plot_count_bar(counts: pd.DataFrame, *, out_base: Path, title: str, xlabel: str) -> Dict[str, Path]:
    work = counts.copy().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    xs = np.arange(len(work))
    bars = ax.bar(xs, work['count'].astype(int).tolist())
    ax.set_xticks(xs)
    ax.set_xticklabels(work['label'].tolist(), rotation=30, ha='right')
    ax.set_ylabel('Count')
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for idx, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.03, str(int(work.iloc[idx]['count'])), ha='center', va='bottom')
    ax.set_ylim(0, max(1.0, float(work['count'].max()) + 1.0))
    fig.tight_layout()
    return _save_figure(fig, out_base)


def _find_heatmap_csv(summary_dir: Path, metric: str, split: str = 'val', cde_format: str = 'labeled', rerank_tag: str = 'R0') -> Path:
    exact = summary_dir / f'heatmap__{metric}__{split}__{cde_format}__{rerank_tag}__mean.csv'
    if exact.exists():
        return exact
    matches = sorted(summary_dir.glob(f'heatmap__*{metric}__{split}__{cde_format}__{rerank_tag}__mean.csv'))
    if matches:
        return matches[0]
    raise FileNotFoundError(f'Could not locate Step 5 heatmap CSV for metric={metric}, split={split} under {summary_dir}')


def _load_step5_matrix(summary_dir: Path) -> pd.DataFrame:
    csv_path = _find_heatmap_csv(summary_dir, 'recall@5', split='val', cde_format='labeled', rerank_tag='R0')
    matrix = pd.read_csv(csv_path, index_col=0)
    matrix.index = [str(x) for x in matrix.index]
    matrix.columns = [str(x) for x in matrix.columns]
    return matrix


def _load_step10b_summary(summary_dir: Path) -> pd.DataFrame:
    summary_csv = summary_dir / 'joint_screen_summary_by_model.csv'
    runs_csv = summary_dir / 'joint_screen_runs_validation.csv'
    if summary_csv.exists():
        return pd.read_csv(summary_csv)
    if runs_csv.exists():
        runs = pd.read_csv(runs_csv)
        summary = (
            runs.groupby(['model_name', 'base_model_id', 'query_variant', 'recipe', 'cde_format', 'rerank_mode', 'hybrid_alpha', 'placeholder_strategy', 'loss'], dropna=False)
            .agg(
                n_runs=('run_dir', 'count'),
                val_recall5_mean=('val_recall5', 'mean'),
                val_recall5_std=('val_recall5', 'std'),
                val_mrr100_mean=('val_mrr100', 'mean'),
                val_mrr100_std=('val_mrr100', 'std'),
            )
            .reset_index()
        )
        for col in ['val_recall5_std', 'val_mrr100_std']:
            summary[col] = summary[col].fillna(0.0)
        return summary
    raise FileNotFoundError(f'Could not locate joint screening summary inputs under {summary_dir}')


def _load_top2_manifest(top2_manifest_csv: Path) -> pd.DataFrame:
    if not top2_manifest_csv.exists():
        return pd.DataFrame()
    return pd.read_csv(top2_manifest_csv)


def _long_from_step5_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    tidy = matrix.reset_index().rename(columns={'index': 'query_variant'}).melt(id_vars='query_variant', var_name='recipe', value_name='validation_recall5')
    tidy['query'] = tidy['query_variant'].map(query_display_name)
    tidy['recipe_label'] = tidy['recipe'].map(recipe_display_name)
    tidy['representation'] = tidy.apply(lambda row: representation_display_name(row['query_variant'], row['recipe']), axis=1)
    return tidy


def _resolve_phase2_metrics_path(stage_root: Path, run_dir: Path) -> Path:
    model_slug = run_dir.parent.parent.name
    run_id = run_dir.name
    eval_metrics_path = stage_root / model_slug / 'eval' / run_id / 'metrics.json'
    if eval_metrics_path.exists():
        return eval_metrics_path
    return run_dir / 'metrics.json'


def _collect_phase2_split_metrics(stage_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rc_path in sorted(stage_root.glob('*/runs/*/run_config.json')):
        run_dir = rc_path.parent
        metrics_path = _resolve_phase2_metrics_path(stage_root, run_dir)
        if not metrics_path.exists():
            continue
        try:
            rc = json.loads(rc_path.read_text(encoding='utf-8'))
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        except Exception:
            continue
        rep = rc.get('representation') or {}
        train = rc.get('train') or {}
        hardneg = rc.get('hardneg') or {}
        model_name = str(rc.get('model_name') or rc.get('base_model_id') or rc.get('base_model') or '')
        base_model_id = str(rc.get('base_model_id') or rc.get('base_model') or model_name)
        strategy = str(hardneg.get('strategy') or '')
        hard_band = str(hardneg.get('hard_band') or '')
        semihard_band = str(hardneg.get('effective_semihard_band') or hardneg.get('semihard_band') or '')
        semihard_tight_band = str(hardneg.get('semihard_tight_band') or '')
        curriculum_bands = str(hardneg.get('curriculum_bands') or '')
        hardcurr_bands = str(hardneg.get('effective_curriculum_bands') or hardneg.get('hardcurr_bands') or '')
        mining_strategy = str(rc.get('mining_strategy') or hardneg.get('mining_strategy') or '')
        if not mining_strategy:
            if strategy == 'none':
                mining_strategy = 'none'
            elif strategy == 'hard' and hard_band == '1-25':
                mining_strategy = 'hard_top25'
            elif strategy == 'semihard' and semihard_band == '25-100':
                mining_strategy = 'semihard_1_50'
            elif strategy == 'hardcurr' and hardcurr_bands == '1-50__1-25__1-15':
                mining_strategy = 'hardcurr_1_50__1_25__1_15'
            else:
                mining_strategy = strategy
        for split, vals in (metrics.get('metrics_by_split') or {}).items():
            rows.append(
                {
                    'run_dir': str(run_dir),
                    'model_name': model_name,
                    'base_model_id': base_model_id,
                    'query_variant': str(rep.get('query_variant') or rc.get('query_variant') or ''),
                    'recipe': str(rep.get('recipe') or rc.get('recipe') or ''),
                    'cde_format': str(rep.get('cde_format') or rc.get('cde_format') or ''),
                    'rerank_mode': str(rep.get('rerank_mode') or rc.get('rerank_mode') or 'R0'),
                    'hybrid_alpha': float(rep.get('hybrid_alpha') or rc.get('hybrid_alpha') or 0.5),
                    'placeholder_strategy': str(rc.get('placeholder_strategy') or (rc.get('recipe_configs') or {}).get('placeholder_policy') or (rep.get('recipe_configs') or {}).get('placeholder_policy') or ''),
                    'seed': int(train.get('seed', -1)),
                    'loss': str(train.get('loss') or ''),
                    'lr': float(train.get('lr') or float('nan')),
                    'batch_size': int(train.get('batch_size', -1)),
                    'temperature': float(train.get('temperature') or float('nan')),
                    'epochs': int(train.get('epochs', -1)),
                    'mining_strategy': mining_strategy,
                    'strategy': strategy,
                    'nneg': int(hardneg.get('effective_nneg', hardneg.get('nneg', 0))),
                    'hard_band': hard_band,
                    'semihard_band': semihard_band,
                    'semihard_tight_band': semihard_tight_band,
                    'curriculum_bands': curriculum_bands,
                    'hardcurr_bands': hardcurr_bands,
                    'split': str(split),
                    'recall@5': float((vals or {}).get('recall@5', float('nan'))),
                    'mrr@100': float((vals or {}).get('mrr@100', float('nan'))),
                }
            )
    return pd.DataFrame(rows)


def _normalize_text_match_value(value: object) -> str:
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass
    text = str(value).strip()
    return '' if text.lower() == 'nan' else text


def _match_mask(df: pd.DataFrame, row: Mapping[str, Any], cols: Sequence[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in cols:
        if col not in df.columns or col not in row:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                target = float(row[col])
                current = pd.to_numeric(df[col], errors='coerce')
                mask &= np.isclose(current, target, equal_nan=True)
                continue
            except Exception:
                pass
        target_text = _normalize_text_match_value(row[col])
        current_text = df[col].map(_normalize_text_match_value)
        mask &= current_text == target_text
    return mask


@dataclass
class PublicationFigureBuilder:
    repo_root: Path | str = Path('.')
    out_root: Path | str = Path('artifacts/summaries/paper_publication_figures')
    step5_summary_dir: Optional[Path | str] = None
    step7_stage_root: Optional[Path | str] = None
    step8_placeholder_stage_root: Optional[Path | str] = None
    step8_omit_stage_root: Optional[Path | str] = None
    step8_placeholder_config_path: Optional[Path | str] = None
    step8_omit_config_path: Optional[Path | str] = None
    step9_stage_root: Optional[Path | str] = None
    step9_config_path: Optional[Path | str] = None
    step10b_summary_dir: Optional[Path | str] = None
    step10b_top2_manifest_csv: Optional[Path | str] = None
    step10c_winners_csv: Optional[Path | str] = None
    step13_winners_csv: Optional[Path | str] = None
    step13_stage_root: Optional[Path | str] = None
    available_results: bool = False
    with_latex: bool = False
    with_step7_companion_table: bool = True
    with_final_mrr_table: bool = False

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root)
        self.out_root = Path(self.out_root)
        self.step5_summary_dir = Path(self.step5_summary_dir) if self.step5_summary_dir is not None else self.repo_root / 'artifacts' / 'summaries' / 'paper_step5_heatmap'
        self.step7_stage_root = Path(self.step7_stage_root) if self.step7_stage_root is not None else self.repo_root / 'artifacts' / 'paper_step7_finalists'
        self.step8_placeholder_stage_root = Path(self.step8_placeholder_stage_root) if self.step8_placeholder_stage_root is not None else self.repo_root / 'artifacts' / 'paper_step8_refine_placeholder'
        self.step8_omit_stage_root = Path(self.step8_omit_stage_root) if self.step8_omit_stage_root is not None else self.repo_root / 'artifacts' / 'paper_step8_refine_omit'
        self.step8_placeholder_config_path = Path(self.step8_placeholder_config_path) if self.step8_placeholder_config_path is not None else self.repo_root / 'configs' / 'experiments' / 'paper_textrep_refinement_placeholder.yaml'
        self.step8_omit_config_path = Path(self.step8_omit_config_path) if self.step8_omit_config_path is not None else self.repo_root / 'configs' / 'experiments' / 'paper_textrep_refinement_omit.yaml'
        self.step9_stage_root = Path(self.step9_stage_root) if self.step9_stage_root is not None else self.repo_root / 'artifacts' / 'paper_step9_sapbert_pooling'
        self.step9_config_path = Path(self.step9_config_path) if self.step9_config_path is not None else self.repo_root / 'configs' / 'experiments' / 'paper_sapbert_pooling_ablation.yaml'
        self.step10b_summary_dir = Path(self.step10b_summary_dir) if self.step10b_summary_dir is not None else self.repo_root / 'artifacts' / 'summaries' / 'paper_step11_4_screen_joint'
        self.step10b_top2_manifest_csv = Path(self.step10b_top2_manifest_csv) if self.step10b_top2_manifest_csv is not None else self.repo_root / 'artifacts' / 'summaries' / 'paper_step11_3_top2_finalists' / 'top2_representations_by_model.csv'
        phase1_winners_name = 'phase1_model_winners_available.csv' if self.available_results else 'phase1_model_winners.csv'
        phase2_winners_name = 'phase2_model_winners_available.csv' if self.available_results else 'phase2_model_winners.csv'
        self.step10c_winners_csv = Path(self.step10c_winners_csv) if self.step10c_winners_csv is not None else self.repo_root / 'artifacts' / 'summaries' / 'paper_step11_phase1_winners_by_model' / phase1_winners_name
        self.step13_winners_csv = Path(self.step13_winners_csv) if self.step13_winners_csv is not None else self.repo_root / 'artifacts' / 'summaries' / 'paper_step13_phase2_winners_by_model' / phase2_winners_name
        self.step13_stage_root = Path(self.step13_stage_root) if self.step13_stage_root is not None else self.repo_root / 'artifacts' / 'paper_step13_finetune_phase2_bf16'

    def build_step5(self) -> Dict[str, Path]:
        out_dir = _ensure_dir(self.out_root / 'step5')
        matrix = _load_step5_matrix(Path(self.step5_summary_dir))
        query_order = [q for q in QUERY_PAPER_ORDER if q in matrix.index]
        recipe_order = sorted([str(x) for x in matrix.columns], key=recipe_plot_order_key)
        matrix = matrix.loc[query_order, recipe_order]
        display_matrix = matrix.copy()
        display_matrix.index = [query_display_name(x) for x in display_matrix.index]
        display_matrix.columns = [recipe_display_name(x) for x in display_matrix.columns]
        csv_path = out_dir / 'step5_heatmap_recall5.csv'
        display_matrix.to_csv(csv_path)
        outputs = {'csv': csv_path}
        outputs.update(
            _plot_heatmap(
                display_matrix,
                out_base=out_dir / 'step5_heatmap_recall5',
                title='Step 5 SapBERT (mean pooling) validation Recall@5 heatmap',
                xlabel='CDE representation',
                ylabel='Query representation',
                annotate=True,
                xtick_rotation=35,
            )
        )
        ranked = _long_from_step5_matrix(matrix).sort_values('validation_recall5', ascending=False, na_position='last').reset_index(drop=True)
        ranked.to_csv(out_dir / 'step5_heatmap_recall5_ranked_cells.csv', index=False)
        outputs['ranked_csv'] = out_dir / 'step5_heatmap_recall5_ranked_cells.csv'
        return outputs

    def build_step6(self) -> Dict[str, Path]:
        out_dir = _ensure_dir(self.out_root / 'step6')
        data = compute_step6_data(step5_summary_dir=Path(self.step5_summary_dir))
        outputs: Dict[str, Path] = {}
        rank_exports = _write_table_exports(
            data['rank_table'],
            out_dir / 'step6_heatmap_rank_table_val.csv',
            md_path=out_dir / 'step6_heatmap_rank_table_val.md',
            latex_path=(out_dir / 'step6_heatmap_rank_table_val.tex') if self.with_latex else None,
        )
        outputs.update({f'rank_table_{k}': v for k, v in rank_exports.items()})
        selected_exports = _write_table_exports(
            data['selected_top4'],
            out_dir / 'step6_selected_top4_val.csv',
            md_path=out_dir / 'step6_selected_top4_val.md',
            latex_path=(out_dir / 'step6_selected_top4_val.tex') if self.with_latex else None,
        )
        outputs.update({f'selected_top4_{k}': v for k, v in selected_exports.items()})
        if not data['secondary_tie_warnings'].empty:
            tie_csv = out_dir / 'step6_secondary_tie_warnings.csv'
            data['secondary_tie_warnings'].to_csv(tie_csv, index=False)
            outputs['secondary_tie_warnings_csv'] = tie_csv
        if not data['mismatch_rows'].empty:
            mismatch_csv = out_dir / 'step6_heatmap_recomputed_vs_file_mismatches.csv'
            data['mismatch_rows'].to_csv(mismatch_csv, index=False)
            outputs['mismatch_csv'] = mismatch_csv
        cmd_path = out_dir / 'paper_generate_configs_command.txt'
        cmd_path.write_text(str(data['config_command']).strip() + '\n', encoding='utf-8')
        outputs['config_command_txt'] = cmd_path

        plot_df = data['selected_top4'].sort_values(
            'query_variant', key=lambda s: s.map({q: i for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4'])})
        ).reset_index(drop=True)
        screening_fig, screening_ax = plt.subplots(figsize=(8.2, 4.0))
        xs = np.arange(len(plot_df))
        bars = screening_ax.bar(xs, plot_df['primary_mean'].astype(float).tolist())
        screening_ax.set_xticks(xs)
        screening_ax.set_xticklabels(plot_df['plot_label'].astype(str).tolist(), rotation=25, ha='right')
        screening_ax.set_ylabel('Validation Recall@5')
        screening_ax.set_title('Step 6 screening backbone selection Recall@5')
        screening_ax.set_ylim(0, max(0.05, float(plot_df['primary_mean'].max()) + 0.05))
        for idx, bar in enumerate(bars):
            val = float(plot_df.iloc[idx]['primary_mean'])
            screening_ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.005,
                _format_metric(val),
                ha='center',
                va='bottom',
                fontsize=9,
            )
        screening_fig.tight_layout()
        outputs.update(
            {f'screening_{k}': v for k, v in _save_figure(screening_fig, out_dir / 'step6_selected_top4_screening_validation_recall5').items()}
        )

        finalists_runs = collect_phase1_runs(Path(self.step7_stage_root)) if Path(self.step7_stage_root).exists() else pd.DataFrame()
        selected_pairs = data['selected_top4'][['query_variant', 'recipe']].drop_duplicates().copy()
        selected_pairs['representation'] = selected_pairs.apply(
            lambda row: representation_display_name(row['query_variant'], row['recipe']), axis=1
        )
        selected_order = selected_pairs.sort_values(
            'query_variant', key=lambda s: s.map({q: i for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4'])})
        )['representation'].astype(str).tolist()
        filtered = finalists_runs.merge(selected_pairs[['query_variant', 'recipe']], on=['query_variant', 'recipe'], how='inner') if not finalists_runs.empty else pd.DataFrame()
        if not filtered.empty:
            summary = (
                filtered.groupby(['base_model_id', 'query_variant', 'recipe'], dropna=False)
                .agg(
                    validation_recall5=('val_recall5', 'mean'),
                    validation_mrr100=('val_mrr100', 'mean'),
                    n_runs=('run_dir', 'count'),
                )
                .reset_index()
            )
            summary['model'] = summary['base_model_id'].map(model_display_name)
            summary['representation'] = summary.apply(
                lambda row: representation_display_name(row['query_variant'], row['recipe']), axis=1
            )
            model_ids = sorted(summary['base_model_id'].drop_duplicates().astype(str).tolist(), key=_model_order_key)
            col_order = [model_display_name(mid) for mid in model_ids]
            matrix = summary.pivot(index='representation', columns='model', values='validation_recall5')
            matrix = matrix.reindex(index=selected_order, columns=col_order)
            heatmap_csv = out_dir / 'step6_selected_top4_validation_recall5.csv'
            matrix.to_csv(heatmap_csv)
            outputs['heatmap_csv'] = heatmap_csv
            outputs.update(
                _plot_heatmap(
                    matrix,
                    out_base=out_dir / 'step6_selected_top4_validation_recall5',
                    title='Step 6 all-model validation Recall@5 on the selected top-4 representations',
                    xlabel='Model',
                    ylabel='Selected representation',
                    annotate=True,
                    xtick_rotation=25,
                    figsize=(max(7.0, 1.1 * matrix.shape[1]), max(3.8, 0.9 * matrix.shape[0])),
                )
            )
            companion = summary[
                ['model', 'base_model_id', 'representation', 'query_variant', 'recipe', 'validation_recall5', 'validation_mrr100', 'n_runs']
            ].copy()
            companion = companion.sort_values(['representation', 'model']).reset_index(drop=True)
            eval_exports = _write_table_exports(
                companion,
                out_dir / 'step6_selected_top4_all_models_validation.csv',
                md_path=out_dir / 'step6_selected_top4_all_models_validation.md',
                latex_path=(out_dir / 'step6_selected_top4_all_models_validation.tex') if self.with_latex else None,
            )
            outputs.update({f'all_models_{k}': v for k, v in eval_exports.items()})
        else:
            outputs.update(_save_figure(screening_fig, out_dir / 'step6_selected_top4_validation_recall5'))
        return outputs

    def build_step7(self) -> Dict[str, Path]:
        out_dir = _ensure_dir(self.out_root / 'step7')
        runs = collect_phase1_runs(Path(self.step7_stage_root))
        if runs.empty:
            raise FileNotFoundError(f'No Step 7 finalist runs found under {self.step7_stage_root}')
        winners = pd.DataFrame(FINALIST_REPRESENTATIONS, columns=['query_variant', 'recipe'])
        filtered = runs.merge(winners, on=['query_variant', 'recipe'], how='inner')
        if filtered.empty:
            raise ValueError('Step 7 runs did not contain any of the expected finalist representations.')
        present = {(str(r['query_variant']), str(r['recipe'])) for _, r in filtered[['query_variant', 'recipe']].drop_duplicates().iterrows()}
        missing = [pair for pair in FINALIST_REPRESENTATIONS if pair not in present]
        if missing:
            warnings.warn(f'Step 7 finalist runs are missing expected representations: {missing}')
        summary = (
            filtered.groupby(['base_model_id', 'query_variant', 'recipe'], dropna=False)
            .agg(
                validation_recall5=('val_recall5', 'mean'),
                validation_mrr100=('val_mrr100', 'mean'),
                n_runs=('run_dir', 'count'),
            )
            .reset_index()
        )
        summary['representation'] = summary.apply(lambda row: representation_display_name(row['query_variant'], row['recipe']), axis=1)
        row_order_pairs = [pair for pair in FINALIST_REPRESENTATIONS if pair in present]
        row_order = [representation_display_name(q, r) for q, r in row_order_pairs]
        model_ids = sorted(summary['base_model_id'].drop_duplicates().astype(str).tolist(), key=_model_order_key)
        summary['model'] = summary['base_model_id'].map(model_display_name)
        col_order = [model_display_name(mid) for mid in model_ids]
        matrix = summary.pivot(index='representation', columns='model', values='validation_recall5')
        matrix = matrix.reindex(index=row_order, columns=col_order)
        csv_path = out_dir / 'step7_finalists_heatmap_recall5.csv'
        matrix.to_csv(csv_path)
        outputs = {'csv': csv_path}
        outputs.update(
            _plot_heatmap(
                matrix,
                out_base=out_dir / 'step7_finalists_heatmap_recall5',
                title='Step 7 finalists validation Recall@5 heatmap',
                xlabel='Model',
                ylabel='Finalist representation',
                annotate=True,
                xtick_rotation=25,
                figsize=(max(7.0, 1.1 * matrix.shape[1]), max(3.8, 0.9 * matrix.shape[0])),
            )
        )
        if self.with_step7_companion_table:
            companion = summary[['model', 'representation', 'validation_recall5', 'validation_mrr100', 'n_runs']].copy()
            companion = companion.sort_values(['representation', 'model']).reset_index(drop=True)
            companion.to_csv(out_dir / 'step7_finalists_companion_metrics.csv', index=False)
            outputs['companion_csv'] = out_dir / 'step7_finalists_companion_metrics.csv'
        return outputs

    def build_step8(self) -> Dict[str, Path]:
        out_dir = _ensure_dir(self.out_root / 'step8')
        data = compute_step8_data(
            step8_placeholder_stage_root=Path(self.step8_placeholder_stage_root),
            step8_omit_stage_root=Path(self.step8_omit_stage_root),
            refinement_config_paths=[Path(self.step8_placeholder_config_path), Path(self.step8_omit_config_path)],
        )
        outputs: Dict[str, Path] = {}
        table_specs = [
            ('refine_val_per_cell', 'step8_refinement_validation_per_cell', True),
            ('refine_condition_val', 'step8_refinement_validation_condition_summary', True),
            ('best_refinement_condition_val', 'step8_refinement_best_condition_validation', True),
            ('refine_delta_val', 'step8_refinement_validation_delta_vs_anchor', False),
            ('refine_best_condition_reports', 'step8_refinement_report_best_validation_condition', True),
            ('refine_best_condition_test_per_cell', 'step8_refinement_test_best_validation_condition_per_cell', True),
            ('refine_delta_report', 'step8_refinement_report_delta_vs_anchor', False),
        ]
        for key, stem, use_latex in table_specs:
            df = data[key]
            if df is None or getattr(df, 'empty', True):
                continue
            exports = _write_table_exports(
                df,
                out_dir / f'{stem}.csv',
                md_path=out_dir / f'{stem}.md',
                latex_path=(out_dir / f'{stem}.tex') if self.with_latex and use_latex else None,
            )
            outputs.update({f'{stem}_{k}': v for k, v in exports.items()})
        val_heat = data['heat_validation'].copy()
        val_heat.index = ['placeholder' if str(x) == 'placeholder' else str(x) for x in val_heat.index]
        val_heat.columns = ['labeled' if str(x) == 'labeled' else str(x) for x in val_heat.columns]
        val_heat.to_csv(out_dir / 'step8_refinement_validation_recall5_heatmap.csv')
        outputs['validation_heatmap_csv'] = out_dir / 'step8_refinement_validation_recall5_heatmap.csv'
        outputs.update(
            _plot_heatmap(
                val_heat,
                out_base=out_dir / 'step8_refinement_validation_recall5_heatmap',
                title='Step 8 refinement validation Recall@5 heatmap',
                xlabel='CDE format',
                ylabel='Placeholder policy',
                annotate=True,
                xtick_rotation=0,
                figsize=(5.0, 4.0),
            )
        )
        if data['heat_report'] is not None:
            split_name = str(data['report_plot_split'])
            report_heat = data['heat_report'].copy()
            report_heat.to_csv(out_dir / f'step8_refinement_{split_name}_recall5_heatmap.csv')
            outputs[f'{split_name}_heatmap_csv'] = out_dir / f'step8_refinement_{split_name}_recall5_heatmap.csv'
            outputs.update(
                {f'{split_name}_{k}': v for k, v in _plot_heatmap(
                    report_heat,
                    out_base=out_dir / f'step8_refinement_{split_name}_recall5_heatmap',
                    title=f'Step 8 refinement {split_name} Recall@5 heatmap',
                    xlabel='CDE format',
                    ylabel='Placeholder policy',
                    annotate=True,
                    xtick_rotation=0,
                    figsize=(5.0, 4.0),
                ).items()}
            )
        return outputs

    def build_step9(self) -> Dict[str, Path]:
        out_dir = _ensure_dir(self.out_root / 'step9')
        data = compute_step9_data(
            step9_stage_root=Path(self.step9_stage_root),
            step9_config_path=Path(self.step9_config_path),
        )
        outputs: Dict[str, Path] = {}
        table_specs = [
            ('pool_val_per_cell', 'step9_pooling_validation_per_cell', True),
            ('pool_condition_val', 'step9_pooling_validation_condition_summary', True),
            ('best_pooling_variant_val', 'step9_pooling_best_variant_validation', True),
            ('pool_delta_split', 'step9_pooling_split_deltas', False),
            ('pool_best_variant_reports', 'step9_pooling_report_best_validation_variant', True),
            ('pool_best_variant_test_per_cell', 'step9_pooling_test_best_validation_variant_per_cell', True),
        ]
        for key, stem, use_latex in table_specs:
            df = data[key]
            if df is None or getattr(df, 'empty', True):
                continue
            exports = _write_table_exports(
                df,
                out_dir / f'{stem}.csv',
                md_path=out_dir / f'{stem}.md',
                latex_path=(out_dir / f'{stem}.tex') if self.with_latex and use_latex else None,
            )
            outputs.update({f'{stem}_{k}': v for k, v in exports.items()})
        if not data['validation_plot_df'].empty:
            plot_df = data['validation_plot_df'].copy()
            plot_df.to_csv(out_dir / 'step9_pooling_validation_recall5.csv', index=False)
            outputs['validation_plot_csv'] = out_dir / 'step9_pooling_validation_recall5.csv'
            labels = plot_df['cell_label'].astype(str).tolist()
            x = np.arange(len(labels))
            width = 0.35
            fig, ax = plt.subplots(figsize=(9.0, 4.5))
            ax.bar(x - width / 2, plot_df['recall@5__mean_mean'].astype(float).tolist(), width=width, label='mean')
            ax.bar(x + width / 2, plot_df['recall@5__cls_mean'].astype(float).tolist(), width=width, label='cls')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha='right')
            ax.set_ylabel('Recall@5')
            ax.set_title('Step 9 SapBERT pooling on validation')
            ax.legend()
            fig.tight_layout()
            outputs.update(_save_figure(fig, out_dir / 'step9_pooling_validation_recall5'))
        if not data['report_plot_df'].empty and data['report_plot_split'] is not None:
            split_name = str(data['report_plot_split'])
            plot_df = data['report_plot_df'].copy()
            plot_df.to_csv(out_dir / f'step9_pooling_{split_name}_recall5.csv', index=False)
            outputs[f'{split_name}_plot_csv'] = out_dir / f'step9_pooling_{split_name}_recall5.csv'
            labels = plot_df['cell_label'].astype(str).tolist()
            x = np.arange(len(labels))
            width = 0.35
            fig, ax = plt.subplots(figsize=(9.0, 4.5))
            ax.bar(x - width / 2, plot_df['recall@5__mean_mean'].astype(float).tolist(), width=width, label='mean')
            ax.bar(x + width / 2, plot_df['recall@5__cls_mean'].astype(float).tolist(), width=width, label='cls')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha='right')
            ax.set_ylabel('Recall@5')
            ax.set_title(f'Step 9 SapBERT pooling on {split_name}')
            ax.legend()
            fig.tight_layout()
            outputs.update({f'{split_name}_{k}': v for k, v in _save_figure(fig, out_dir / f'step9_pooling_{split_name}_recall5').items()})
        return outputs

    def build_step10b(self) -> Dict[str, Path]:
        out_dir = _ensure_dir(self.out_root / 'step10b')
        summary = _load_step10b_summary(Path(self.step10b_summary_dir))
        top2 = _load_top2_manifest(Path(self.step10b_top2_manifest_csv))
        summary = summary.copy()
        summary['model'] = summary['base_model_id'].map(model_display_name)
        summary['representation'] = summary.apply(lambda row: representation_display_name(row['query_variant'], row['recipe']), axis=1)
        model_ids = sorted(summary['base_model_id'].drop_duplicates().astype(str).tolist(), key=_model_order_key)
        tidy_rows: List[Dict[str, Any]] = []
        winner_rows: List[Dict[str, Any]] = []
        for model_id in model_ids:
            sub = summary[summary['base_model_id'] == model_id].copy()
            if sub.empty:
                continue
            if not top2.empty:
                top2_sub = top2[top2['base_model_id'].astype(str) == str(model_id)].copy()
                top2_sub = top2_sub.sort_values('selected_rank').drop_duplicates(subset=['query_variant', 'recipe'])
                rep_pairs = [(str(r['query_variant']), str(r['recipe'])) for _, r in top2_sub.iterrows()]
            else:
                rep_rank = (
                    sub.groupby(['query_variant', 'recipe'], dropna=False)['val_recall5_mean']
                    .mean()
                    .reset_index()
                    .sort_values('val_recall5_mean', ascending=False)
                )
                rep_pairs = [(str(r['query_variant']), str(r['recipe'])) for _, r in rep_rank.head(2).iterrows()]
            rep_labels = [representation_display_name(q, r) for q, r in rep_pairs]
            for loss in STEP10B_LOSS_ORDER:
                for (qv, rec), rep_label in zip(rep_pairs, rep_labels):
                    match = sub[(sub['query_variant'].astype(str) == str(qv)) & (sub['recipe'].astype(str) == str(rec)) & (sub['loss'].astype(str) == str(loss))]
                    val = float(match.iloc[0]['val_recall5_mean']) if not match.empty else float('nan')
                    tidy_rows.append(
                        {
                            'model_id': model_id,
                            'model': model_display_name(model_id),
                            'loss': loss_display_name(loss),
                            'representation': rep_label,
                            'validation_recall5': val,
                        }
                    )
            best = sub.sort_values(['val_recall5_mean', 'val_mrr100_mean'], ascending=[False, False]).iloc[0]
            winner_rows.append(
                {
                    'model': model_display_name(model_id),
                    'selected_representation': representation_display_name(best['query_variant'], best['recipe']),
                    'selected_loss': loss_display_name(best['loss']),
                    'validation_recall5': float(best['val_recall5_mean']),
                    'validation_mrr100': float(best['val_mrr100_mean']),
                }
            )
        tidy = pd.DataFrame(tidy_rows)
        if tidy.empty:
            raise ValueError('No Step 10b screening data could be assembled into 2x2 panels.')
        tidy.to_csv(out_dir / 'step10b_joint_screening_panels.csv', index=False)
        outputs = {'csv': out_dir / 'step10b_joint_screening_panels.csv'}
        winners_df = pd.DataFrame(winner_rows)
        if not winners_df.empty:
            winners_df.to_csv(out_dir / 'step10b_joint_screening_winners_by_model.csv', index=False)
            outputs['winners_csv'] = out_dir / 'step10b_joint_screening_winners_by_model.csv'
        n_models = len(sorted(tidy['model_id'].drop_duplicates().astype(str).tolist(), key=_model_order_key))
        ncols = 3
        nrows = int(math.ceil(float(n_models) / float(ncols))) if n_models else 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, max(4.0, 3.8 * nrows)), constrained_layout=True)
        axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
        image = None
        ordered_models = sorted(tidy['model_id'].drop_duplicates().astype(str).tolist(), key=_model_order_key)
        for ax in axes_arr.ravel():
            ax.set_visible(False)
        for panel_idx, model_id in enumerate(ordered_models):
            ax = axes_arr.ravel()[panel_idx]
            ax.set_visible(True)
            sub = tidy[tidy['model_id'].astype(str) == str(model_id)].copy()
            rep_order = list(dict.fromkeys(sub['representation'].tolist()))
            loss_order = [loss_display_name(x) for x in STEP10B_LOSS_ORDER]
            matrix = sub.pivot(index='loss', columns='representation', values='validation_recall5').reindex(index=loss_order, columns=rep_order)
            panel_values = matrix.to_numpy(dtype=float)
            panel_vmin = float(np.nanmin(panel_values)) if np.isfinite(panel_values).any() else None
            panel_vmax = float(np.nanmax(panel_values)) if np.isfinite(panel_values).any() else None
            image = ax.imshow(panel_values, aspect='auto', cmap=HEATMAP_CMAP, vmin=panel_vmin, vmax=panel_vmax)
            ax.set_xticks(np.arange(matrix.shape[1]))
            ax.set_yticks(np.arange(matrix.shape[0]))
            ax.set_xticklabels([x.replace(' × ', '\n× ') for x in matrix.columns], rotation=0)
            if panel_idx % ncols == 0:
                ax.set_yticklabels(matrix.index.tolist())
            else:
                ax.set_yticklabels([])
            ax.set_title(model_display_name(model_id))
            _annotate_heatmap(ax, matrix, vmin=panel_vmin, vmax=panel_vmax)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle('Step 10b joint screening: validation Recall@5')
        outputs.update(_save_figure(fig, out_dir / 'step10b_joint_screening_panels'))
        return outputs

    def build_step10c(self) -> Dict[str, Path]:
        out_dir = _ensure_dir(self.out_root / 'step10c')
        winners = pd.read_csv(Path(self.step10c_winners_csv)).copy()
        winners['model'] = winners['base_model_id'].map(model_display_name)
        winners['selected_representation'] = winners.apply(lambda row: representation_display_name(row['query_variant'], row['recipe']), axis=1)
        winners['selected_loss_display'] = winners['selected_loss'].map(loss_display_name)
        winners = winners.sort_values('base_model_id', key=lambda s: s.map(_model_order_key)).reset_index(drop=True)
        counts = _phase_winner_counts(winners)
        counts_csv = out_dir / 'step10c_hparam_counts.csv'
        counts.to_csv(counts_csv, index=False)
        outputs = {'hparam_counts_csv': counts_csv}
        outputs.update(_plot_hparam_count_panels(counts, out_base=out_dir / 'step10c_hparam_counts', title_prefix='Step 10c Phase 1 winner counts'))
        best = pd.DataFrame(
            {
                'model': winners['model'],
                'selected_representation': winners['selected_representation'],
                'selected_loss': winners['selected_loss_display'],
                'learning_rate': winners['lr'],
                'batch_size': winners['batch_size'],
                'temperature': winners['temperature'],
                'epochs': winners['epochs'],
                'validation_recall5': winners['val_recall5_mean'],
                'validation_mrr100': winners['val_mrr100_mean'],
            }
        )
        best_csv = out_dir / 'step10c_best_by_model.csv'
        best.to_csv(best_csv, index=False)
        outputs['best_by_model_csv'] = best_csv
        outputs.update(_plot_best_by_model(best, out_base=out_dir / 'step10c_best_by_model'))
        winners_table = pd.DataFrame(
            {
                'model': winners['model'],
                'selected representation': winners['selected_representation'],
                'selected loss': winners['selected_loss_display'],
                'learning rate': winners['lr'],
                'batch size': winners['batch_size'],
                'temperature': winners['temperature'],
                'epochs': winners['epochs'],
                'validation Recall@5': winners['val_recall5_mean'],
                'validation MRR@100': winners['val_mrr100_mean'],
            }
        )
        latex_path = out_dir / 'step10c_phase1_winners_table.tex' if self.with_latex else None
        table_exports = _write_table_exports(
            winners_table,
            out_dir / 'step10c_phase1_winners_table.csv',
            md_path=out_dir / 'step10c_phase1_winners_table.md',
            latex_path=latex_path,
        )
        outputs.update({f'phase1_winners_table_{k}': v for k, v in table_exports.items()})
        return outputs

    def build_step13(self) -> Dict[str, Path]:
        out_dir = _ensure_dir(self.out_root / 'step13')
        phase2 = pd.read_csv(Path(self.step13_winners_csv)).copy()
        phase2['model'] = phase2['base_model_id'].map(model_display_name)
        phase2['selected_representation'] = phase2.apply(lambda row: representation_display_name(row['query_variant'], row['recipe']), axis=1)
        phase2['selected_loss_display'] = phase2['selected_loss'].map(loss_display_name)
        phase2 = phase2.sort_values('base_model_id', key=lambda s: s.map(_model_order_key)).reset_index(drop=True)
        outputs: Dict[str, Path] = {}
        counts = _phase_winner_counts(phase2)
        counts_csv = out_dir / 'step13_hparam_counts.csv'
        counts.to_csv(counts_csv, index=False)
        outputs['hparam_counts_csv'] = counts_csv
        outputs.update(_plot_hparam_count_panels(counts, out_base=out_dir / 'step13_hparam_counts', title_prefix='Step 13 Phase 2 winner counts'))
        hardneg = (
            phase2.groupby('mining_strategy', dropna=False)['base_model_id']
            .nunique()
            .reset_index(name='count')
            .rename(columns={'mining_strategy': 'label'})
        )
        hardneg['label'] = hardneg['label'].replace({
            'hard_top25': 'hard',
            'hardcurr_1_50__1_25__1_15': 'hardcurr',
        })
        hardneg = hardneg.sort_values(['count', 'label'], ascending=[False, True]).reset_index(drop=True)
        hardneg.to_csv(out_dir / 'step13_hardneg_strategy_counts.csv', index=False)
        outputs['hardneg_counts_csv'] = out_dir / 'step13_hardneg_strategy_counts.csv'
        outputs.update(_plot_count_bar(hardneg, out_base=out_dir / 'step13_hardneg_strategy_counts', title='Step 13 hard-negative strategy winners', xlabel='Hard-negative strategy'))
        phase1 = pd.read_csv(Path(self.step10c_winners_csv)).copy()
        aggregate_counts = _phase_winner_counts(pd.concat([phase1, phase2], axis=0, ignore_index=True, sort=False))
        aggregate_csv = out_dir / 'step10c_step13_aggregate_hparam_counts.csv'
        aggregate_counts.to_csv(aggregate_csv, index=False)
        outputs['aggregate_hparam_counts_csv'] = aggregate_csv
        outputs.update(_plot_hparam_count_panels(aggregate_counts, out_base=out_dir / 'step10c_step13_aggregate_hparam_counts', title_prefix='Aggregate Phase 1 + Phase 2 winner counts'))
        split_metrics = _collect_phase2_split_metrics(Path(self.step13_stage_root)) if Path(self.step13_stage_root).exists() else pd.DataFrame()
        split_rows: List[Dict[str, Any]] = []
        split_order = list(SPLIT_DISPLAY.keys())
        for _, winner in phase2.iterrows():
            winner_dict = winner.to_dict()
            matched = pd.DataFrame()
            if not split_metrics.empty:
                match_cols = [col for col in PHASE2_GROUP_COLS if col in split_metrics.columns and col in phase2.columns]
                mask = _match_mask(split_metrics, winner_dict, match_cols)
                matched = split_metrics[mask].copy()
            source_mode = 'seed_mean'
            if matched.empty and str(winner.get('representative_run_dir') or '').strip():
                rep_run_dir = Path(str(winner['representative_run_dir']))
                rep_metrics_path = _resolve_phase2_metrics_path(Path(self.step13_stage_root), rep_run_dir)
                if rep_metrics_path.exists():
                    payload = json.loads(rep_metrics_path.read_text(encoding='utf-8'))
                    for split, vals in (payload.get('metrics_by_split') or {}).items():
                        matched = pd.concat(
                            [
                                matched,
                                pd.DataFrame(
                                    [
                                        {
                                            'split': split,
                                            'recall@5': float((vals or {}).get('recall@5', float('nan'))),
                                            'mrr@100': float((vals or {}).get('mrr@100', float('nan'))),
                                        }
                                    ]
                                ),
                            ],
                            axis=0,
                            ignore_index=True,
                        )
                    source_mode = 'representative_run'
            if matched.empty:
                warnings.warn(f'No Phase 2 split metrics found for winner {winner_dict.get("base_model_id")}. Final results table will contain blanks for that row.')
                continue
            agg = matched.groupby('split', dropna=False).agg({'recall@5': 'mean', 'mrr@100': 'mean'}).reset_index()
            for split in split_order:
                row = agg[agg['split'].astype(str) == split]
                split_rows.append(
                    {
                        'model': model_display_name(winner_dict['base_model_id']),
                        'base_model_id': winner_dict['base_model_id'],
                        'split': split,
                        'split_display': split_display_name(split),
                        'recall@5': float(row.iloc[0]['recall@5']) if not row.empty else float('nan'),
                        'mrr@100': float(row.iloc[0]['mrr@100']) if not row.empty else float('nan'),
                        'source_mode': source_mode,
                    }
                )
        split_df = pd.DataFrame(split_rows)
        if split_df.empty:
            raise ValueError('Could not assemble any Phase 2 test/holdout metrics for the final results table.')
        split_df = split_df.sort_values(['base_model_id', 'split'], key=lambda s: s.map(_model_order_key) if s.name == 'base_model_id' else s).reset_index(drop=True)
        split_df.to_csv(out_dir / 'step13_final_results_long.csv', index=False)
        outputs['final_results_long_csv'] = out_dir / 'step13_final_results_long.csv'
        row_order = [model_display_name(mid) for mid in sorted(phase2['base_model_id'].astype(str).tolist(), key=_model_order_key)]
        col_order = [split_display_name(x) for x in split_order]
        recall_table = split_df.pivot(index='model', columns='split_display', values='recall@5').reindex(index=row_order, columns=col_order)
        recall_table_df = recall_table.reset_index().rename_axis(columns=None)
        latex_path = out_dir / 'step13_final_results_table_recall5.tex' if self.with_latex else None
        table_exports = _write_table_exports(recall_table_df, out_dir / 'step13_final_results_table_recall5.csv', md_path=out_dir / 'step13_final_results_table_recall5.md', latex_path=latex_path)
        outputs.update({f'final_results_recall5_{k}': v for k, v in table_exports.items()})
        if self.with_final_mrr_table:
            mrr_table = split_df.pivot(index='model', columns='split_display', values='mrr@100').reindex(index=row_order, columns=col_order)
            mrr_df = mrr_table.reset_index().rename_axis(columns=None)
            mrr_latex = out_dir / 'step13_final_results_table_mrr100.tex' if self.with_latex else None
            mrr_exports = _write_table_exports(mrr_df, out_dir / 'step13_final_results_table_mrr100.csv', md_path=out_dir / 'step13_final_results_table_mrr100.md', latex_path=mrr_latex)
            outputs.update({f'final_results_mrr100_{k}': v for k, v in mrr_exports.items()})
        return outputs

    def build_all(self, *, steps: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Path]]:
        requested = [str(x).lower() for x in (steps or ['step5', 'step6', 'step7', 'step8', 'step9', 'step10b', 'step10c', 'step13'])]
        out: Dict[str, Dict[str, Path]] = {}
        if 'step5' in requested:
            out['step5'] = self.build_step5()
        if 'step6' in requested:
            out['step6'] = self.build_step6()
        if 'step7' in requested:
            out['step7'] = self.build_step7()
        if 'step8' in requested:
            out['step8'] = self.build_step8()
        if 'step9' in requested:
            out['step9'] = self.build_step9()
        if 'step10b' in requested:
            out['step10b'] = self.build_step10b()
        if 'step10c' in requested:
            out['step10c'] = self.build_step10c()
        if 'step13' in requested:
            out['step13'] = self.build_step13()
        return out


__all__ = [
    'FINALIST_REPRESENTATIONS',
    'MODEL_DISPLAY',
    'MODEL_PAPER_ORDER',
    'PublicationFigureBuilder',
    'QUERY_DISPLAY',
    'QUERY_PAPER_ORDER',
    'SPLIT_DISPLAY',
    'loss_display_name',
    'model_display_name',
    'query_display_name',
    'recipe_display_name',
    'recipe_plot_order_key',
    'representation_display_name',
    'split_display_name',
]
