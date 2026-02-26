"""Earthworm Optimization Algorithm (EOA) for hyperparameter tuning.

Uses mealpy's EOA.OriginalEOA to optimize model hyperparameters
and ensemble weights via population-based metaheuristic search.
"""
import numpy as np
import pandas as pd
from mealpy import FloatVar, EOA, Termination
from torch.utils.tensorboard import SummaryWriter

from src.tuning import (
    get_search_space,
    evaluate_params,
    evaluate_ensemble_weights,
)


def tune_model_eoa(
    model_name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    gender: str = 'M',
    epoch: int = 100,
    pop_size: int = 30,
    patience: int = 15,
    log_dir: str | None = None,
) -> dict:
    """Tune a single model's hyperparameters using Earthworm Optimization.

    Args:
        model_name: 'xgboost', 'lightgbm', or 'catboost'.
        df: Feature DataFrame.
        feature_cols: Feature column names.
        gender: 'M' or 'W'.
        epoch: Number of EOA generations.
        pop_size: Population size.
        log_dir: TensorBoard log directory (e.g. 'runs/xgboost_M_eoa').

    Returns:
        Dict with 'best_params', 'best_brier', 'all_results'.
    """
    space = get_search_space(model_name)
    param_names = list(space.keys())

    # Build bounds arrays
    lb = [space[p]['low'] for p in param_names]
    ub = [space[p]['high'] for p in param_names]

    writer = None
    if log_dir:
        writer = SummaryWriter(log_dir=log_dir)

    trial_counter = [0]  # mutable counter for closure

    def objective_function(solution):
        # Map solution vector back to named params
        params = {}
        for i, name in enumerate(param_names):
            val = solution[i]
            if space[name]['type'] == 'int':
                val = int(round(val))
            params[name] = val

        result = evaluate_params(
            model_name, params, df, feature_cols, gender,
            writer=writer, trial_idx=trial_counter[0], method='eoa',
        )
        trial_counter[0] += 1
        return result['mean_brier']

    problem_dict = {
        'bounds': FloatVar(lb=lb, ub=ub),
        'minmax': 'min',
        'obj_func': objective_function,
    }

    term = Termination(max_epoch=epoch, max_early_stop=patience)

    model = EOA.OriginalEOA(
        epoch=epoch,
        pop_size=pop_size,
        p_c=0.9,
        p_m=0.01,
        n_best=2,
        alpha=0.98,
        beta=0.9,
        gama=0.9,
    )

    g_best = model.solve(problem_dict, termination=term)

    if writer:
        writer.close()

    # Extract best params
    best_params = {}
    for i, name in enumerate(param_names):
        val = g_best.solution[i]
        if space[name]['type'] == 'int':
            val = int(round(val))
        best_params[name] = val

    return {
        'best_params': best_params,
        'best_brier': float(g_best.target.fitness),
        'n_evaluations': trial_counter[0],
    }


def tune_ensemble_weights_eoa(
    oof_preds_dict: dict[str, np.ndarray],
    y_true: np.ndarray,
    epoch: int = 200,
    pop_size: int = 20,
    patience: int = 15,
    log_dir: str | None = None,
) -> dict:
    """Optimize ensemble weights using Earthworm Optimization.

    Args:
        oof_preds_dict: {model_name: oof_predictions}.
        y_true: True labels.
        epoch: EOA generations.
        pop_size: Population size.
        log_dir: TensorBoard log directory.

    Returns:
        Dict with 'weights' and 'brier'.
    """
    model_names = list(oof_preds_dict.keys())
    n_models = len(model_names)

    writer = None
    if log_dir:
        writer = SummaryWriter(log_dir=log_dir)

    trial_counter = [0]

    def objective_function(solution):
        brier = evaluate_ensemble_weights(
            np.array(solution), oof_preds_dict, y_true, model_names,
        )

        if writer:
            w = np.abs(np.array(solution))
            w = w / w.sum() if w.sum() > 0 else w
            hparams = {name: float(w[i]) for i, name in enumerate(model_names)}
            hparams['method'] = 'eoa'
            writer.add_hparams(
                hparam_dict=hparams,
                metric_dict={'brier': brier},
                run_name=f'trial_{trial_counter[0]:03d}',
            )
        trial_counter[0] += 1
        return brier

    problem_dict = {
        'bounds': FloatVar(lb=[0.0] * n_models, ub=[1.0] * n_models),
        'minmax': 'min',
        'obj_func': objective_function,
    }

    term = Termination(max_epoch=epoch, max_early_stop=patience)

    model = EOA.OriginalEOA(
        epoch=epoch,
        pop_size=pop_size,
        p_c=0.9,
        p_m=0.01,
        n_best=2,
        alpha=0.98,
        beta=0.9,
        gama=0.9,
    )

    g_best = model.solve(problem_dict, termination=term)

    if writer:
        writer.close()

    raw_w = np.abs(np.array(g_best.solution))
    normalized = raw_w / raw_w.sum()
    weights = {name: float(normalized[i]) for i, name in enumerate(model_names)}

    return {
        'weights': weights,
        'brier': float(g_best.target.fitness),
        'n_evaluations': trial_counter[0],
    }
