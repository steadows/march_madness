"""Shared hyperparameter tuning infrastructure with TensorBoard HParams logging.

Defines search spaces and evaluation functions used by both EOA and Ax optimizers.
"""
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from src.calibration import clip_predictions
from src.cv import expanding_window_cv, evaluate_brier
from src.models import train_xgboost, train_lightgbm, train_catboost


# --- Search Space Definitions ---

SEARCH_SPACES = {
    'xgboost': {
        'max_depth':        {'low': 3,    'high': 10,   'type': 'int'},
        'learning_rate':    {'low': 0.01, 'high': 0.3,  'type': 'float', 'log_scale': True},
        'subsample':        {'low': 0.5,  'high': 1.0,  'type': 'float'},
        'colsample_bytree': {'low': 0.5,  'high': 1.0,  'type': 'float'},
        'min_child_weight': {'low': 1,    'high': 10,   'type': 'int'},
        'reg_alpha':        {'low': 0.0,  'high': 1.0,  'type': 'float'},
        'reg_lambda':       {'low': 0.1,  'high': 5.0,  'type': 'float'},
    },
    'lightgbm': {
        'max_depth':        {'low': 3,    'high': 10,   'type': 'int'},
        'learning_rate':    {'low': 0.01, 'high': 0.3,  'type': 'float', 'log_scale': True},
        'subsample':        {'low': 0.5,  'high': 1.0,  'type': 'float'},
        'colsample_bytree': {'low': 0.5,  'high': 1.0,  'type': 'float'},
        'min_child_samples': {'low': 5,   'high': 50,   'type': 'int'},
        'reg_alpha':        {'low': 0.0,  'high': 1.0,  'type': 'float'},
        'reg_lambda':       {'low': 0.1,  'high': 5.0,  'type': 'float'},
    },
    'catboost': {
        'depth':            {'low': 3,    'high': 10,   'type': 'int'},
        'learning_rate':    {'low': 0.01, 'high': 0.3,  'type': 'float', 'log_scale': True},
        'l2_leaf_reg':      {'low': 0.1,  'high': 10.0, 'type': 'float'},
    },
}

# Fixed params that don't get tuned
FIXED_PARAMS = {
    'xgboost': {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': 500,
        'verbosity': 0,
        'random_state': 42,
    },
    'lightgbm': {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'n_estimators': 500,
        'verbose': -1,
        'random_state': 42,
    },
    'catboost': {
        'iterations': 500,
        'eval_metric': 'Logloss',
        'verbose': 0,
        'random_state': 42,
    },
}

TRAIN_FNS = {
    'xgboost': train_xgboost,
    'lightgbm': train_lightgbm,
    'catboost': train_catboost,
}


def get_search_space(model_name: str) -> dict:
    """Return search space definition for a given model."""
    return SEARCH_SPACES[model_name]


def _cast_params(params: dict, space: dict) -> dict:
    """Cast params to correct types based on search space definition."""
    cast = {}
    for name, val in params.items():
        if name in space and space[name]['type'] == 'int':
            cast[name] = int(round(val))
        else:
            cast[name] = float(val)
    return cast


def evaluate_params(
    model_name: str,
    params: dict,
    df: pd.DataFrame,
    feature_cols: list[str],
    gender: str = 'M',
    writer: SummaryWriter | None = None,
    trial_idx: int | None = None,
    method: str = '',
) -> dict:
    """Evaluate hyperparameters via expanding-window CV. Returns Brier stats.

    Args:
        model_name: 'xgboost', 'lightgbm', or 'catboost'.
        params: Hyperparameters to evaluate (tunable only).
        df: Feature DataFrame with Season and target columns.
        feature_cols: Feature column names.
        gender: 'M' or 'W'.
        writer: TensorBoard SummaryWriter for HParams logging.
        trial_idx: Trial number (for TensorBoard run naming).
        method: 'eoa' or 'ax' (for TensorBoard hparam_dict).

    Returns:
        Dict with 'mean_brier', 'std_brier', 'fold_briers'.
    """
    space = get_search_space(model_name)
    cast = _cast_params(params, space)

    # Merge with fixed params
    full_params = {**FIXED_PARAMS[model_name], **cast}
    train_fn = TRAIN_FNS[model_name]

    folds = expanding_window_cv(df, min_train_end=2019)
    fold_briers = []

    for train_idx, val_idx in folds:
        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]

        X_train = train_df[feature_cols].values
        y_train = train_df['target'].values
        X_val = val_df[feature_cols].values
        y_val = val_df['target'].values

        model = train_fn(X_train, y_train, X_val, y_val, params=full_params)
        preds = clip_predictions(model.predict_proba(X_val)[:, 1])
        fold_briers.append(evaluate_brier(y_val, preds))

    result = {
        'mean_brier': float(np.mean(fold_briers)),
        'std_brier': float(np.std(fold_briers)),
        'fold_briers': fold_briers,
    }

    # TensorBoard HParams logging
    if writer is not None and trial_idx is not None:
        hparam_dict = {'method': method}
        hparam_dict.update(cast)

        metric_dict = {
            'brier_mean': result['mean_brier'],
            'brier_std': result['std_brier'],
        }
        for i, b in enumerate(fold_briers):
            metric_dict[f'brier_fold_{i}'] = b

        writer.add_hparams(
            hparam_dict=hparam_dict,
            metric_dict=metric_dict,
            run_name=f'trial_{trial_idx:03d}',
        )

    return result


def evaluate_ensemble_weights(
    weights_arr: np.ndarray,
    oof_preds_dict: dict[str, np.ndarray],
    y_true: np.ndarray,
    model_names: list[str] | None = None,
) -> float:
    """Evaluate ensemble weights — returns Brier score.

    Args:
        weights_arr: Raw weight array (will be normalized to sum=1).
        oof_preds_dict: {model_name: oof_prediction_array}.
        y_true: True labels.
        model_names: Ordered model names matching weights_arr positions.

    Returns:
        Brier score for the weighted ensemble.
    """
    if model_names is None:
        model_names = list(oof_preds_dict.keys())

    w = np.abs(weights_arr)
    w_sum = w.sum()
    if w_sum == 0:
        return 1.0
    w = w / w_sum

    blended = sum(
        w[i] * oof_preds_dict[name]
        for i, name in enumerate(model_names)
    )
    return float(np.mean((blended - y_true) ** 2))
