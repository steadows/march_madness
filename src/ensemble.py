"""Ensemble pipeline for NCAA tournament prediction.

Runs multiple models through expanding-window CV, collects OOF predictions,
and optimizes ensemble weights to minimize Brier score.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.calibration import clip_predictions
from src.cv import expanding_window_cv, evaluate_brier
from src.models import (
    train_logistic_baseline,
    train_xgboost,
    train_lightgbm,
    train_catboost,
    train_ridge,
)


def _predict_ridge(model, X_val):
    """Predict with Ridge pipeline, handling NaN via stored medians."""
    X_val = np.asarray(X_val, dtype=float)
    medians = getattr(model, '_col_medians', np.zeros(X_val.shape[1]))
    mask = np.isnan(X_val)
    X_filled = X_val.copy()
    for j in range(X_filled.shape[1]):
        X_filled[mask[:, j], j] = medians[j]
    return model.predict_proba(X_filled)[:, 1]


def run_all_models_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = 'target',
    gender: str = 'M',
    model_params: dict | None = None,
) -> dict:
    """Run all 5 models through expanding-window CV and collect OOF predictions.

    Models: logistic (seed_diff only), ridge (all features), XGBoost, LightGBM, CatBoost.

    Args:
        df: Feature DataFrame with Season and target columns.
        feature_cols: All _diff feature column names.
        target_col: Name of target column.
        gender: 'M' or 'W' (for logging).
        model_params: Optional dict of tuned hyperparameters per model,
            e.g. {'xgboost': {...}, 'lightgbm': {...}, 'catboost': {...}}.
            If None, uses default params for all models.

    Returns:
        Dict: {model_name: {'fold_briers': [...], 'mean_brier': float, 'oof_preds': array, 'oof_targets': array}}
    """
    folds = expanding_window_cv(df, min_train_end=2019)

    # Initialize collectors for each model
    model_names = ['logistic', 'ridge', 'xgboost', 'lightgbm', 'catboost']
    results = {name: {'fold_briers': [], 'oof_preds_list': [], 'oof_targets_list': []}
               for name in model_names}

    seed_col = 'seed_num_diff'

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]

        X_train_raw = train_df[feature_cols]
        X_val_raw = val_df[feature_cols]
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        val_season = int(val_df['Season'].iloc[0])

        print(f"  [{gender}] Fold {fold_i+1}: val={val_season}, "
              f"train={len(train_df)}, val={len(val_df)}")

        # --- Logistic: seed_diff only
        X_log_tr = X_train_raw[[seed_col]].fillna(0).values
        X_log_val = X_val_raw[[seed_col]].fillna(0).values
        log_model = train_logistic_baseline(X_log_tr, y_train)
        log_preds = clip_predictions(log_model.predict_proba(X_log_val)[:, 1])
        results['logistic']['fold_briers'].append(evaluate_brier(y_val, log_preds))
        results['logistic']['oof_preds_list'].append(log_preds)
        results['logistic']['oof_targets_list'].append(y_val)

        # --- Ridge: all features (NaN-filled)
        ridge_model = train_ridge(X_train_raw.values, y_train)
        ridge_preds = clip_predictions(_predict_ridge(ridge_model, X_val_raw.values))
        results['ridge']['fold_briers'].append(evaluate_brier(y_val, ridge_preds))
        results['ridge']['oof_preds_list'].append(ridge_preds)
        results['ridge']['oof_targets_list'].append(y_val)

        # --- XGBoost: all features (handles NaN natively)
        xgb_params = model_params.get('xgboost') if model_params else None
        xgb_model = train_xgboost(X_train_raw.values, y_train, X_val_raw.values, y_val, params=xgb_params)
        xgb_preds = clip_predictions(xgb_model.predict_proba(X_val_raw.values)[:, 1])
        results['xgboost']['fold_briers'].append(evaluate_brier(y_val, xgb_preds))
        results['xgboost']['oof_preds_list'].append(xgb_preds)
        results['xgboost']['oof_targets_list'].append(y_val)

        # --- LightGBM: all features (handles NaN natively)
        lgb_params = model_params.get('lightgbm') if model_params else None
        lgb_model = train_lightgbm(X_train_raw.values, y_train, X_val_raw.values, y_val, params=lgb_params)
        lgb_preds = clip_predictions(lgb_model.predict_proba(X_val_raw.values)[:, 1])
        results['lightgbm']['fold_briers'].append(evaluate_brier(y_val, lgb_preds))
        results['lightgbm']['oof_preds_list'].append(lgb_preds)
        results['lightgbm']['oof_targets_list'].append(y_val)

        # --- CatBoost: all features (handles NaN natively)
        cat_params = model_params.get('catboost') if model_params else None
        cat_model = train_catboost(X_train_raw.values, y_train, X_val_raw.values, y_val, params=cat_params)
        cat_preds = clip_predictions(cat_model.predict_proba(X_val_raw.values)[:, 1])
        results['catboost']['fold_briers'].append(evaluate_brier(y_val, cat_preds))
        results['catboost']['oof_preds_list'].append(cat_preds)
        results['catboost']['oof_targets_list'].append(y_val)

    # Concatenate OOF predictions across folds
    for name in model_names:
        results[name]['oof_preds'] = np.concatenate(results[name]['oof_preds_list'])
        results[name]['oof_targets'] = np.concatenate(results[name]['oof_targets_list'])
        results[name]['mean_brier'] = float(np.mean(results[name]['fold_briers']))
        results[name]['std_brier'] = float(np.std(results[name]['fold_briers']))
        del results[name]['oof_preds_list']
        del results[name]['oof_targets_list']

    return results


def simple_average_ensemble(oof_preds_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Simple mean of all model OOF predictions.

    Args:
        oof_preds_dict: {model_name: oof_pred_array}

    Returns:
        Averaged prediction array.
    """
    preds = np.array(list(oof_preds_dict.values()))
    return np.mean(preds, axis=0)


def optimize_ensemble_weights(
    oof_preds_dict: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> dict[str, float]:
    """Find optimal weights that minimize Brier score on OOF predictions.

    Uses scipy.optimize.minimize with Nelder-Mead.
    Weights are constrained to be non-negative and sum to 1.

    Args:
        oof_preds_dict: {model_name: oof_pred_array}
        y_true: True binary labels.

    Returns:
        {model_name: optimal_weight}
    """
    names = list(oof_preds_dict.keys())
    preds_list = [oof_preds_dict[n] for n in names]
    n_models = len(names)

    def brier_loss(weights):
        weights = np.abs(weights)  # ensure non-negative
        w_sum = weights.sum()
        if w_sum == 0:
            return 1.0
        weights = weights / w_sum  # normalize to sum=1
        blended = sum(w * p for w, p in zip(weights, preds_list))
        return float(np.mean((blended - y_true) ** 2))

    result = minimize(
        brier_loss,
        x0=[1.0 / n_models] * n_models,
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10},
    )

    raw_weights = np.abs(result.x)
    normalized = raw_weights / raw_weights.sum()
    return {name: float(w) for name, w in zip(names, normalized)}


def weighted_ensemble(
    oof_preds_dict: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """Apply given weights to model predictions.

    Args:
        oof_preds_dict: {model_name: prediction_array}
        weights: {model_name: weight} (should sum to 1).

    Returns:
        Weighted average prediction array.
    """
    blended = np.zeros_like(list(oof_preds_dict.values())[0], dtype=float)
    for name, preds in oof_preds_dict.items():
        blended += weights.get(name, 0.0) * preds
    return blended


def compute_sample_weights(seasons: np.ndarray) -> np.ndarray:
    """Compute linear recency weights: oldest=0.3, newest=1.0.

    Args:
        seasons: Array of season values (one per row).

    Returns:
        Weight array (same length as seasons).
    """
    seasons = np.asarray(seasons)
    unique_sorted = np.sort(np.unique(seasons))
    n = len(unique_sorted)
    if n <= 1:
        return np.ones(len(seasons))
    weight_map = {s: w for s, w in zip(unique_sorted, np.linspace(0.3, 1.0, n))}
    return np.array([weight_map[s] for s in seasons])


def log_ensemble_results(
    results_m: dict,
    results_w: dict,
    weights_m: dict,
    weights_w: dict,
    output_path: str = 'artifacts/ensemble_results.txt',
) -> None:
    """Write all ensemble CV results to a text file."""
    lines = ['=== Ensemble CV Results ===', '']

    for gender, results, weights in [('M', results_m, weights_m), ('W', results_w, weights_w)]:
        lines.append(f'--- {gender} ---')
        y_true = None
        oof_dict = {}

        for name, data in results.items():
            briers = data['fold_briers']
            lines.append(f'  {name}: mean={data["mean_brier"]:.4f} '
                         f'± {data["std_brier"]:.4f}  '
                         f'folds={[round(b, 4) for b in briers]}')
            oof_dict[name] = data['oof_preds']
            y_true = data['oof_targets']

        # Simple average
        avg_preds = simple_average_ensemble(oof_dict)
        avg_brier = evaluate_brier(y_true, avg_preds)
        lines.append(f'  simple_avg: {avg_brier:.4f}')

        # Weighted ensemble
        wtd_preds = weighted_ensemble(oof_dict, weights)
        wtd_brier = evaluate_brier(y_true, wtd_preds)
        lines.append(f'  weighted_ensemble: {wtd_brier:.4f}')

        lines.append(f'  Optimal weights: {weights}')
        lines.append('')

    report = '\n'.join(lines)
    print(report)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report + '\n')
