"""Ax/BoTorch Bayesian Optimization for hyperparameter tuning.

Uses Ax's AxClient with BoTorch GP backend for sample-efficient
optimization of model hyperparameters and ensemble weights.
"""
import numpy as np
import pandas as pd
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from torch.utils.tensorboard import SummaryWriter

from src.tuning import (
    get_search_space,
    evaluate_params,
    evaluate_ensemble_weights,
)


def _space_to_ax_params(space: dict) -> list[dict]:
    """Convert our search space format to Ax parameter definitions."""
    ax_params = []
    for name, spec in space.items():
        param = {
            'name': name,
            'type': 'range',
            'bounds': [spec['low'], spec['high']],
        }
        if spec['type'] == 'int':
            param['value_type'] = 'int'
        else:
            param['value_type'] = 'float'
        if spec.get('log_scale'):
            param['log_scale'] = True
        ax_params.append(param)
    return ax_params


def tune_model_ax(
    model_name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    gender: str = 'M',
    n_trials: int = 50,
    log_dir: str | None = None,
) -> dict:
    """Tune a single model's hyperparameters using Ax/BoTorch Bayesian Optimization.

    Args:
        model_name: 'xgboost', 'lightgbm', or 'catboost'.
        df: Feature DataFrame.
        feature_cols: Feature column names.
        gender: 'M' or 'W'.
        n_trials: Number of BO trials.
        log_dir: TensorBoard log directory (e.g. 'runs/xgboost_M_ax').

    Returns:
        Dict with 'best_params', 'best_brier', 'all_trials'.
    """
    space = get_search_space(model_name)
    ax_params = _space_to_ax_params(space)

    writer = None
    if log_dir:
        writer = SummaryWriter(log_dir=log_dir)

    ax_client = AxClient(verbose_logging=False)
    ax_client.create_experiment(
        parameters=ax_params,
        objectives={'brier': ObjectiveProperties(minimize=True)},
    )

    all_trials = []

    for trial_idx in range(n_trials):
        params, trial_index = ax_client.get_next_trial()

        result = evaluate_params(
            model_name, params, df, feature_cols, gender,
            writer=writer, trial_idx=trial_idx, method='ax',
        )

        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data={'brier': result['mean_brier']},
        )

        all_trials.append({
            'trial': trial_idx,
            'params': params,
            'brier': result['mean_brier'],
        })

    if writer:
        writer.close()

    best_params, metrics = ax_client.get_best_parameters()
    best_brier = metrics[0]['brier'] if metrics else min(t['brier'] for t in all_trials)

    # Cast int params
    for name in best_params:
        if name in space and space[name]['type'] == 'int':
            best_params[name] = int(round(best_params[name]))

    return {
        'best_params': best_params,
        'best_brier': float(best_brier),
        'n_trials': len(all_trials),
        'all_trials': all_trials,
    }


def tune_ensemble_weights_ax(
    oof_preds_dict: dict[str, np.ndarray],
    y_true: np.ndarray,
    n_trials: int = 100,
    log_dir: str | None = None,
) -> dict:
    """Optimize ensemble weights using Ax/BoTorch Bayesian Optimization.

    Args:
        oof_preds_dict: {model_name: oof_predictions}.
        y_true: True labels.
        n_trials: Number of BO trials.
        log_dir: TensorBoard log directory.

    Returns:
        Dict with 'weights' and 'brier'.
    """
    model_names = list(oof_preds_dict.keys())

    writer = None
    if log_dir:
        writer = SummaryWriter(log_dir=log_dir)

    ax_params = [
        {'name': name, 'type': 'range', 'bounds': [0.0, 1.0], 'value_type': 'float'}
        for name in model_names
    ]

    ax_client = AxClient(verbose_logging=False)
    ax_client.create_experiment(
        parameters=ax_params,
        objectives={'brier': ObjectiveProperties(minimize=True)},
    )

    for trial_idx in range(n_trials):
        params, trial_index = ax_client.get_next_trial()

        weights_arr = np.array([params[name] for name in model_names])
        brier = evaluate_ensemble_weights(weights_arr, oof_preds_dict, y_true, model_names)

        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data={'brier': brier},
        )

        if writer:
            w = np.abs(weights_arr)
            w = w / w.sum() if w.sum() > 0 else w
            hparams = {name: float(w[i]) for i, name in enumerate(model_names)}
            hparams['method'] = 'ax'
            writer.add_hparams(
                hparam_dict=hparams,
                metric_dict={'brier': brier},
                run_name=f'trial_{trial_idx:03d}',
            )

    if writer:
        writer.close()

    best_params, metrics = ax_client.get_best_parameters()
    raw_w = np.array([best_params[name] for name in model_names])
    raw_w = np.abs(raw_w)
    normalized = raw_w / raw_w.sum() if raw_w.sum() > 0 else np.ones(len(model_names)) / len(model_names)
    weights = {name: float(normalized[i]) for i, name in enumerate(model_names)}

    best_brier = metrics[0]['brier'] if metrics else evaluate_ensemble_weights(
        raw_w, oof_preds_dict, y_true, model_names,
    )

    return {
        'weights': weights,
        'brier': float(best_brier),
        'n_trials': n_trials,
    }
