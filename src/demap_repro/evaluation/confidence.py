"""confidence.py

Confidence calibration + reporting utilities.

This module supports the evaluation protocol described in Evaluation Plan.docx:

* Compute a scalar confidence feature per query (e.g., margin s1-s2).
* Calibrate that feature to an estimated probability of Top-1 correctness
  using the validation split.
* Emit reliability curve tables and Expected Calibration Error (ECE).

The functions are intentionally lightweight and dependency-minimal
(numpy/pandas + scikit-learn).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CalibratorResult:
    available: bool
    method: str
    feature_col: str
    target_col: str
    n_fit: int
    notes: str = ""


def _finite_mask(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x.astype(float))


def fit_calibrator(
    df: pd.DataFrame,
    *,
    feature_col: str,
    target_col: str,
    method: str = "isotonic",
) -> Tuple[Optional[Any], CalibratorResult]:
    """Fit a 1D calibrator mapping `feature_col` -> P(correct).

    Parameters
    ----------
    df:
        DataFrame containing feature/target columns.
    feature_col:
        Column containing a scalar confidence feature (e.g., margin_1_2).
    target_col:
        Binary indicator of correctness (e.g., is_correct_top1).
    method:
        "isotonic" (default) or "platt".

    Returns
    -------
    (model, meta)
        model is a scikit-learn calibrator or None if unavailable.
    """

    if feature_col not in df.columns or target_col not in df.columns:
        return (
            None,
            CalibratorResult(
                available=False,
                method=method,
                feature_col=feature_col,
                target_col=target_col,
                n_fit=0,
                notes=f"missing required columns: {feature_col} or {target_col}",
            ),
        )

    x = df[feature_col].astype(float).to_numpy()
    y = df[target_col].astype(int).to_numpy()

    m = _finite_mask(x) & _finite_mask(y)
    x_fit = x[m]
    y_fit = y[m]

    if len(x_fit) < 50:
        return (
            None,
            CalibratorResult(
                available=False,
                method=method,
                feature_col=feature_col,
                target_col=target_col,
                n_fit=int(len(x_fit)),
                notes="not enough finite calibration examples",
            ),
        )

    method_l = (method or "").lower().strip()
    if method_l not in {"isotonic", "platt"}:
        raise ValueError(f"Unknown calibration method: {method}. Expected isotonic or platt")

    if method_l == "isotonic":
        from sklearn.isotonic import IsotonicRegression  # type: ignore

        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(x_fit, y_fit)
    else:
        from sklearn.linear_model import LogisticRegression  # type: ignore

        # Platt scaling on a single scalar feature.
        model = LogisticRegression(solver="lbfgs")
        model.fit(x_fit.reshape(-1, 1), y_fit)

    meta = CalibratorResult(
        available=True,
        method=method_l,
        feature_col=feature_col,
        target_col=target_col,
        n_fit=int(len(x_fit)),
        notes="",
    )
    return model, meta


def apply_calibrator(model: Any, x: np.ndarray) -> np.ndarray:
    """Apply a calibrator model to a 1D feature array."""
    if model is None:
        return np.full_like(x.astype(float), np.nan, dtype=float)

    # IsotonicRegression exposes predict(x)
    if hasattr(model, "predict") and not hasattr(model, "predict_proba"):
        p = model.predict(x.astype(float))
        return np.clip(p.astype(float), 0.0, 1.0)

    # LogisticRegression exposes predict_proba
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x.astype(float).reshape(-1, 1))[:, 1]
        return np.clip(p.astype(float), 0.0, 1.0)

    raise TypeError(f"Unsupported calibrator model type: {type(model)}")


def calibrate_rankings(
    rankings: pd.DataFrame,
    *,
    fit_split: str = "val",
    feature_col: str = "confidence_raw",
    target_col: str = "is_correct_top1",
    out_col: str = "p_correct",
    method: str = "isotonic",
) -> Tuple[pd.DataFrame, Optional[Any], CalibratorResult]:
    """Fit on `fit_split` then attach calibrated probabilities to all rows."""

    if "split" not in rankings.columns:
        meta = CalibratorResult(
            available=False,
            method=method,
            feature_col=feature_col,
            target_col=target_col,
            n_fit=0,
            notes="rankings missing 'split' column",
        )
        out = rankings.copy()
        out[out_col] = np.nan
        return out, None, meta

    df_fit = rankings[rankings["split"].astype(str) == str(fit_split)].copy()
    model, meta = fit_calibrator(df_fit, feature_col=feature_col, target_col=target_col, method=method)

    out = rankings.copy()
    if not meta.available or model is None:
        out[out_col] = np.nan
        return out, None, meta

    x_all = out[feature_col].astype(float).to_numpy()
    m = _finite_mask(x_all)
    p = np.full((len(out),), np.nan, dtype=float)
    if m.any():
        p[m] = apply_calibrator(model, x_all[m])
    out[out_col] = p
    return out, model, meta


def reliability_table(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute a standard reliability curve table.

    Returns a DataFrame with one row per bin.
    """

    y = y_true.astype(float)
    p = p_pred.astype(float)
    m = _finite_mask(y) & _finite_mask(p)
    y = y[m]
    p = np.clip(p[m], 0.0, 1.0)

    if len(p) == 0:
        return pd.DataFrame(
            {
                "bin_idx": [],
                "bin_lo": [],
                "bin_hi": [],
                "n": [],
                "avg_confidence": [],
                "accuracy": [],
            }
        )

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)

    # Assign bins. Include the rightmost edge in the final bin.
    bin_idx = np.minimum(np.searchsorted(edges, p, side="right") - 1, len(edges) - 2)
    bin_idx = np.maximum(bin_idx, 0)

    rows = []
    for b in range(len(edges) - 1):
        mask_b = bin_idx == b
        if not mask_b.any():
            rows.append(
                {
                    "bin_idx": int(b),
                    "bin_lo": float(edges[b]),
                    "bin_hi": float(edges[b + 1]),
                    "n": 0,
                    "avg_confidence": float("nan"),
                    "accuracy": float("nan"),
                }
            )
            continue

        rows.append(
            {
                "bin_idx": int(b),
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "n": int(mask_b.sum()),
                "avg_confidence": float(np.mean(p[mask_b])),
                "accuracy": float(np.mean(y[mask_b])),
            }
        )

    return pd.DataFrame(rows)


def expected_calibration_error(rel: pd.DataFrame) -> float:
    """Compute ECE from a reliability_table DataFrame."""
    if rel is None or len(rel) == 0:
        return float("nan")
    if "n" not in rel.columns:
        return float("nan")

    n_total = float(rel["n"].sum())
    if n_total <= 0:
        return float("nan")

    # Ignore empty bins.
    rr = rel[rel["n"] > 0].copy()
    if len(rr) == 0:
        return float("nan")

    gap = (rr["accuracy"].astype(float) - rr["avg_confidence"].astype(float)).abs()
    w = rr["n"].astype(float) / n_total
    return float((gap * w).sum())


def precision_and_coverage_at_threshold(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    threshold: float = 0.9,
) -> Dict[str, float]:
    """Compute operational metrics at a probability threshold."""
    y = y_true.astype(float)
    p = p_pred.astype(float)
    m = _finite_mask(y) & _finite_mask(p)
    y = y[m]
    p = p[m]
    if len(p) == 0:
        return {"precision": float("nan"), "coverage": 0.0}

    sel = p >= float(threshold)
    coverage = float(sel.mean())
    if sel.sum() == 0:
        return {"precision": float("nan"), "coverage": coverage}
    precision = float(np.mean(y[sel]))
    return {"precision": precision, "coverage": coverage}


def per_split_reliability_and_ece(
    rankings: pd.DataFrame,
    *,
    split_col: str = "split",
    y_col: str = "is_correct_top1",
    p_col: str = "p_correct",
    n_bins: int = 10,
    threshold: float = 0.9,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Compute reliability curves + ECE per split.

    Returns
    -------
    (reliability_df, ece_json)
        reliability_df is concatenated per-split reliability_table with a `split` column.
        ece_json is a nested dict keyed by split.
    """

    if split_col not in rankings.columns:
        raise KeyError(f"rankings missing split column: {split_col}")

    rel_rows = []
    ece: Dict[str, Dict[str, Any]] = {}

    for split, g in rankings.groupby(split_col, dropna=False):
        split_name = str(split)

        if y_col not in g.columns or p_col not in g.columns:
            ece[split_name] = {
                "available": False,
                "reason": f"missing columns: {y_col} or {p_col}",
                "n": int(len(g)),
            }
            continue

        y = g[y_col].astype(float).to_numpy()
        p = g[p_col].astype(float).to_numpy()

        # If we have no probabilities, mark unavailable.
        if not np.isfinite(p).any():
            ece[split_name] = {
                "available": False,
                "reason": "no finite predicted probabilities",
                "n": int(len(g)),
            }
            continue

        rel = reliability_table(y, p, n_bins=n_bins)
        rel["split"] = split_name
        rel_rows.append(rel)

        e = expected_calibration_error(rel)
        op = precision_and_coverage_at_threshold(y, p, threshold=threshold)
        ece[split_name] = {
            "available": True,
            "n": int(len(g)),
            "n_bins": int(n_bins),
            "ece": float(e),
            "precision_at_confidence_threshold": float(op["precision"]),
            "coverage_at_confidence_threshold": float(op["coverage"]),
            "confidence_threshold": float(threshold),
        }

    reliability_df = pd.concat(rel_rows, ignore_index=True) if rel_rows else reliability_table(np.array([]), np.array([]), n_bins=n_bins)
    return reliability_df, ece
