"""Generate submissions with Barttorvik features using existing tuned HPs.

Reads tuned params from artifacts/tuned_params_pre_barttorvik.json (original),
retrains with expanded 47-feature set, re-optimizes ensemble weights,
and writes to NEW output paths only. Does NOT overwrite any existing artifacts.
"""
import json
import sys
import time
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

from src.models import (
    train_logistic_baseline,
    train_xgboost,
    train_lightgbm,
    train_catboost,
    train_ridge,
)
from src.ensemble import (
    run_all_models_cv,
    optimize_ensemble_weights,
    weighted_ensemble,
    simple_average_ensemble,
    _predict_ridge,
)
from src.tuning_eoa import tune_ensemble_weights_eoa
from src.tuning_ax import tune_ensemble_weights_ax
from src.calibration import clip_predictions
from src.cv import evaluate_brier
from src.submission import generate_submission


def train_ensemble_models(df, feature_cols, target_col='target', model_params=None):
    """Train all 5 models on full dataset with optional tuned params."""
    X = df[feature_cols]
    y = df[target_col].values

    seed_col = 'seed_num_diff'
    models = {}

    # Logistic: seed_diff only (no tunable HPs)
    X_log = X[[seed_col]].fillna(0).values
    models['logistic'] = train_logistic_baseline(X_log, y)

    # Ridge: all features (no tunable HPs)
    models['ridge'] = train_ridge(X.values, y)

    # XGBoost
    xgb_params = model_params.get('xgboost') if model_params else None
    models['xgboost'] = train_xgboost(X.values, y, params=xgb_params)

    # LightGBM
    lgb_params = model_params.get('lightgbm') if model_params else None
    models['lightgbm'] = train_lightgbm(X.values, y, params=lgb_params)

    # CatBoost
    cat_params = model_params.get('catboost') if model_params else None
    models['catboost'] = train_catboost(X.values, y, params=cat_params)

    return models


def make_ensemble_predict_fn(models_m, models_w, weights_m, weights_w):
    """Create predict_fn for generate_submission that uses weighted ensemble."""
    seed_col = 'seed_num_diff'

    def predict_fn(X_df, gender):
        models = models_m if gender == 'M' else models_w
        weights = weights_m if gender == 'M' else weights_w
        diff_cols = [c for c in X_df.columns if c.endswith('_diff')]

        X = X_df[diff_cols]
        X_vals = X.values

        preds = {}
        X_log = X[[seed_col]].fillna(0).values
        preds['logistic'] = models['logistic'].predict_proba(X_log)[:, 1]
        preds['ridge'] = _predict_ridge(models['ridge'], X_vals)
        preds['xgboost'] = models['xgboost'].predict_proba(X_vals)[:, 1]
        preds['lightgbm'] = models['lightgbm'].predict_proba(X_vals)[:, 1]
        preds['catboost'] = models['catboost'].predict_proba(X_vals)[:, 1]

        blended = np.zeros(len(X_df), dtype=float)
        for name, w in weights.items():
            blended += w * preds[name]

        return clip_predictions(blended)

    return predict_fn


def validate_submission(path, expected_rows=None):
    """Validate submission file."""
    df = pd.read_csv(path)
    print(f"  Rows: {len(df)}")
    print(f"  Pred range: [{df['Pred'].min():.4f}, {df['Pred'].max():.4f}]")
    print(f"  Pred mean: {df['Pred'].mean():.4f}, std: {df['Pred'].std():.4f}")
    assert df['Pred'].notna().all(), "NaN predictions!"
    assert (df['Pred'] >= 0.05).all() and (df['Pred'] <= 0.95).all(), "Out of clip range!"
    assert df['Pred'].std() > 0.05, f"Std too low: {df['Pred'].std():.4f}"
    if expected_rows:
        assert len(df) == expected_rows, f"Expected {expected_rows}, got {len(df)}"
    print("  VALID")


def main():
    t_total = time.time()

    # Load data
    df_m = pd.read_csv('artifacts/features_men.csv')
    df_w = pd.read_csv('artifacts/features_women.csv')
    feature_cols = [c for c in df_m.columns if c.endswith('_diff')]

    # Load ORIGINAL tuned params (not overwritten)
    with open('artifacts/tuned_params_pre_barttorvik.json') as f:
        tuned = json.load(f)

    print(f"Men: {df_m.shape}, Women: {df_w.shape}")
    print(f"Features: {len(feature_cols)}")

    # --- Step 1: Re-run ensemble CV with tuned HPs ---
    all_weights = {}
    all_methods = {}
    for gender, df in [('M', df_m), ('W', df_w)]:
        print(f"\n{'='*60}")
        print(f"  {gender} — Ensemble CV with tuned hyperparameters")
        print(f"{'='*60}")

        gender_params = tuned[gender]
        model_hp = {name: gender_params[name]
                    for name in ['xgboost', 'lightgbm', 'catboost']}

        print(f"  XGBoost params: {model_hp['xgboost']}")
        print(f"  LightGBM params: {model_hp['lightgbm']}")
        print(f"  CatBoost params: {model_hp['catboost']}")

        t0 = time.time()
        results = run_all_models_cv(df, feature_cols, gender=gender,
                                    model_params=model_hp)
        cv_elapsed = time.time() - t0
        print(f"\n  CV completed in {cv_elapsed:.1f}s")

        # Report individual model scores
        print(f"\n  Individual model Brier scores ({gender}):")
        for name, data in results.items():
            print(f"    {name}: {data['mean_brier']:.4f} +/- {data['std_brier']:.4f}")

        # Collect OOF predictions
        oof_dict = {n: results[n]['oof_preds'] for n in results}
        y_true = results['logistic']['oof_targets']

        # Simple average
        avg_preds = simple_average_ensemble(oof_dict)
        avg_brier = evaluate_brier(y_true, avg_preds)
        print(f"    simple_avg: {avg_brier:.4f}")

        # --- Three-way weight optimization ---
        print(f"\n  Weight optimization ({gender}):")

        # Scipy (Nelder-Mead)
        t0 = time.time()
        scipy_weights = optimize_ensemble_weights(oof_dict, y_true)
        scipy_preds = weighted_ensemble(oof_dict, scipy_weights)
        scipy_brier = evaluate_brier(y_true, scipy_preds)
        print(f"    [Scipy]  Brier: {scipy_brier:.4f}  ({time.time() - t0:.1f}s)")
        print(f"             {scipy_weights}")

        # EOA
        t0 = time.time()
        eoa_result = tune_ensemble_weights_eoa(
            oof_dict, y_true,
            epoch=200, pop_size=20,
            log_dir=f'runs/ensemble_weights_barttorvik_{gender}_eoa',
        )
        eoa_preds = weighted_ensemble(oof_dict, eoa_result['weights'])
        eoa_brier = evaluate_brier(y_true, eoa_preds)
        print(f"    [EOA]    Brier: {eoa_brier:.4f}  ({time.time() - t0:.1f}s)")
        print(f"             {eoa_result['weights']}")

        # Ax/BoTorch
        t0 = time.time()
        ax_result = tune_ensemble_weights_ax(
            oof_dict, y_true,
            n_trials=100,
            log_dir=f'runs/ensemble_weights_barttorvik_{gender}_ax',
        )
        ax_preds = weighted_ensemble(oof_dict, ax_result['weights'])
        ax_brier = evaluate_brier(y_true, ax_preds)
        print(f"    [Ax]     Brier: {ax_brier:.4f}  ({time.time() - t0:.1f}s)")
        print(f"             {ax_result['weights']}")

        # Pick winner
        candidates = {
            'scipy': (scipy_weights, scipy_brier),
            'eoa': (eoa_result['weights'], eoa_brier),
            'ax': (ax_result['weights'], ax_brier),
        }
        best_method = min(candidates, key=lambda m: candidates[m][1])
        best_weights, best_brier = candidates[best_method]
        print(f"\n    Winner: {best_method} (Brier {best_brier:.4f})")

        all_weights[gender] = best_weights
        all_methods[gender] = best_method

    # --- Step 2: Update tuned_params.json with new weights ---
    for gender in ['M', 'W']:
        tuned[gender]['ensemble_weights'] = all_weights[gender]
        tuned[gender]['weight_method'] = all_methods[gender]

    with open('artifacts/tuned_params_barttorvik.json', 'w') as f:
        json.dump(tuned, f, indent=2)
    print("\nSaved artifacts/tuned_params_barttorvik.json")

    # --- Step 3: Train final models on ALL data with tuned HPs ---
    t0 = time.time()
    print("\nTraining final Men's models with tuned HPs...")
    model_hp_m = {n: tuned['M'][n] for n in ['xgboost', 'lightgbm', 'catboost']}
    models_m = train_ensemble_models(df_m, feature_cols, model_params=model_hp_m)

    print("Training final Women's models with tuned HPs...")
    model_hp_w = {n: tuned['W'][n] for n in ['xgboost', 'lightgbm', 'catboost']}
    models_w = train_ensemble_models(df_w, feature_cols, model_params=model_hp_w)
    print(f"Final model training completed in {time.time() - t0:.1f}s")

    predict_fn = make_ensemble_predict_fn(
        models_m, models_w, all_weights['M'], all_weights['W']
    )

    # --- Step 4: Generate submissions ---
    t0 = time.time()
    print("\nGenerating Stage 1 submission...")
    generate_submission(
        predict_fn=predict_fn,
        stage=1,
        output_path='submissions/ensemble_barttorvik_v1.csv',
    )
    print("Validating ensemble_barttorvik_v1.csv:")
    validate_submission('submissions/ensemble_barttorvik_v1.csv', expected_rows=519144)

    print("\nGenerating Stage 2 submission...")
    generate_submission(
        predict_fn=predict_fn,
        stage=2,
        output_path='submissions/ensemble_barttorvik_2026.csv',
    )
    print("Validating ensemble_barttorvik_2026.csv:")
    validate_submission('submissions/ensemble_barttorvik_2026.csv')
    print(f"Submission generation completed in {time.time() - t0:.1f}s")

    print(f"\nTotal time: {time.time() - t_total:.1f}s")
    print("Done!")


if __name__ == '__main__':
    main()
