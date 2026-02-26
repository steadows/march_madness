"""End-to-end hyperparameter tuning pipeline.

Runs EOA and Ax/BoTorch independently on each model, compares results,
then re-optimizes ensemble weights with tuned models.
All trials logged to TensorBoard HParams.
"""
import json
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

from src.tuning import evaluate_params
from src.tuning_eoa import tune_model_eoa, tune_ensemble_weights_eoa
from src.tuning_ax import tune_model_ax, tune_ensemble_weights_ax
from src.ensemble import (
    run_all_models_cv,
    simple_average_ensemble,
    optimize_ensemble_weights,
    weighted_ensemble,
)
from src.cv import evaluate_brier


def tune_all_models(df, feature_cols, gender, eoa_epochs=100, ax_trials=50):
    """Tune XGBoost, LightGBM, CatBoost with both EOA and Ax."""
    models_to_tune = ['xgboost', 'lightgbm', 'catboost']
    results = {}

    for model_name in models_to_tune:
        print(f"\n{'='*60}")
        print(f"Tuning {model_name} ({gender})")
        print(f"{'='*60}")

        # EOA
        print(f"\n  [EOA] Running {eoa_epochs} epochs...")
        eoa_result = tune_model_eoa(
            model_name, df, feature_cols, gender,
            epoch=eoa_epochs, pop_size=30,
            log_dir=f'runs/{model_name}_{gender}_eoa',
        )
        print(f"  [EOA] Best Brier: {eoa_result['best_brier']:.4f}")
        print(f"  [EOA] Best params: {eoa_result['best_params']}")

        # Ax/BoTorch
        print(f"\n  [Ax] Running {ax_trials} trials...")
        ax_result = tune_model_ax(
            model_name, df, feature_cols, gender,
            n_trials=ax_trials,
            log_dir=f'runs/{model_name}_{gender}_ax',
        )
        print(f"  [Ax] Best Brier: {ax_result['best_brier']:.4f}")
        print(f"  [Ax] Best params: {ax_result['best_params']}")

        # Pick winner
        if eoa_result['best_brier'] <= ax_result['best_brier']:
            winner = 'eoa'
            best_params = eoa_result['best_params']
            best_brier = eoa_result['best_brier']
        else:
            winner = 'ax'
            best_params = ax_result['best_params']
            best_brier = ax_result['best_brier']

        print(f"\n  Winner: {winner} (Brier {best_brier:.4f})")

        results[model_name] = {
            'eoa': {'params': eoa_result['best_params'], 'brier': eoa_result['best_brier']},
            'ax': {'params': ax_result['best_params'], 'brier': ax_result['best_brier']},
            'winner': winner,
            'best_params': best_params,
            'best_brier': best_brier,
        }

    return results


def tune_ensemble_weights_both(oof_preds_dict, y_true, gender):
    """Optimize ensemble weights with both EOA and Ax."""
    print(f"\n{'='*60}")
    print(f"Optimizing ensemble weights ({gender})")
    print(f"{'='*60}")

    # EOA
    print("\n  [EOA] Running weight optimization...")
    eoa_result = tune_ensemble_weights_eoa(
        oof_preds_dict, y_true,
        epoch=200, pop_size=20,
        log_dir=f'runs/ensemble_weights_{gender}_eoa',
    )
    print(f"  [EOA] Brier: {eoa_result['brier']:.4f}")
    print(f"  [EOA] Weights: {eoa_result['weights']}")

    # Ax
    print("\n  [Ax] Running weight optimization...")
    ax_result = tune_ensemble_weights_ax(
        oof_preds_dict, y_true,
        n_trials=100,
        log_dir=f'runs/ensemble_weights_{gender}_ax',
    )
    print(f"  [Ax] Brier: {ax_result['brier']:.4f}")
    print(f"  [Ax] Weights: {ax_result['weights']}")

    # Also compare with scipy (current method)
    from src.ensemble import optimize_ensemble_weights as scipy_opt
    scipy_weights = scipy_opt(oof_preds_dict, y_true)
    scipy_preds = weighted_ensemble(oof_preds_dict, scipy_weights)
    scipy_brier = evaluate_brier(y_true, scipy_preds)
    print(f"\n  [Scipy] Brier: {scipy_brier:.4f}")
    print(f"  [Scipy] Weights: {scipy_weights}")

    return {
        'eoa': eoa_result,
        'ax': ax_result,
        'scipy': {'weights': scipy_weights, 'brier': scipy_brier},
    }


def main():
    df_m = pd.read_csv('artifacts/features_men.csv')
    df_w = pd.read_csv('artifacts/features_women.csv')
    feature_cols = [c for c in df_m.columns if c.endswith('_diff')]

    print(f"Men: {df_m.shape}, Women: {df_w.shape}")
    print(f"Features: {len(feature_cols)}")

    all_results = {}

    # --- Tune model hyperparameters ---
    for gender, df in [('M', df_m), ('W', df_w)]:
        print(f"\n\n{'#'*60}")
        print(f"### {gender} — Model Hyperparameter Tuning ###")
        print(f"{'#'*60}")

        model_results = tune_all_models(df, feature_cols, gender)
        all_results[f'{gender}_models'] = model_results

    # --- Re-run ensemble CV with tuned params ---
    for gender, df in [('M', df_m), ('W', df_w)]:
        print(f"\n\n{'#'*60}")
        print(f"### {gender} — Tuned Ensemble CV ###")
        print(f"{'#'*60}")

        # Get tuned params for GBMs
        model_results = all_results[f'{gender}_models']
        tuned_hp = {name: model_results[name]['best_params']
                    for name in ['xgboost', 'lightgbm', 'catboost']}

        # Run ensemble CV with tuned params
        tuned_results = run_all_models_cv(df, feature_cols, gender=gender,
                                          model_params=tuned_hp)

        oof_dict = {n: tuned_results[n]['oof_preds'] for n in tuned_results}
        y_true = tuned_results['logistic']['oof_targets']

        # Report individual scores
        for name, data in tuned_results.items():
            print(f"  {name}: {data['mean_brier']:.4f}")

        # Simple avg
        avg_preds = simple_average_ensemble(oof_dict)
        print(f"  simple_avg: {evaluate_brier(y_true, avg_preds):.4f}")

        # Weight optimization with both methods
        weight_results = tune_ensemble_weights_both(oof_dict, y_true, gender)
        all_results[f'{gender}_weights'] = weight_results

    # --- Save results ---
    # Convert numpy types for JSON serialization
    def jsonify(obj):
        if isinstance(obj, dict):
            return {k: jsonify(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [jsonify(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open('artifacts/tuning_results.json', 'w') as f:
        json.dump(jsonify(all_results), f, indent=2)
    print("\nSaved artifacts/tuning_results.json")

    # Save best params per model per gender
    tuned_params = {}
    for gender in ['M', 'W']:
        tuned_params[gender] = {}
        model_results = all_results[f'{gender}_models']
        for model_name in ['xgboost', 'lightgbm', 'catboost']:
            tuned_params[gender][model_name] = model_results[model_name]['best_params']

        # Best weights
        wr = all_results[f'{gender}_weights']
        best_method = min(['eoa', 'ax', 'scipy'], key=lambda m: wr[m]['brier'])
        tuned_params[gender]['ensemble_weights'] = wr[best_method]['weights']
        tuned_params[gender]['weight_method'] = best_method

    with open('artifacts/tuned_params.json', 'w') as f:
        json.dump(jsonify(tuned_params), f, indent=2)
    print("Saved artifacts/tuned_params.json")

    print("\n\nTuning complete! View TensorBoard: tensorboard --logdir runs/")


if __name__ == '__main__':
    main()
