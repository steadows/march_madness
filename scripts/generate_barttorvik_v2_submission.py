"""Generate submissions using Barttorvik v2 tuned params.

Loads tuned_params_barttorvik_v2.json (HPs + weights already optimized
against OOF from those same HPs), trains final models on all data,
and generates submissions. No CV re-run needed.

Output: submissions/ensemble_barttorvik_v2_{stage1,2026}.csv
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
from src.ensemble import _predict_ridge
from src.calibration import clip_predictions
from src.submission import generate_submission


def train_ensemble_models(df, feature_cols, target_col='target', model_params=None):
    """Train all 5 models on full dataset with tuned params."""
    X = df[feature_cols]
    y = df[target_col].values
    seed_col = 'seed_num_diff'

    models = {}
    X_log = X[[seed_col]].fillna(0).values
    models['logistic'] = train_logistic_baseline(X_log, y)
    models['ridge'] = train_ridge(X.values, y)
    models['xgboost'] = train_xgboost(X.values, y, params=model_params.get('xgboost'))
    models['lightgbm'] = train_lightgbm(X.values, y, params=model_params.get('lightgbm'))
    models['catboost'] = train_catboost(X.values, y, params=model_params.get('catboost'))
    return models


def make_predict_fn(models_m, models_w, weights_m, weights_w):
    seed_col = 'seed_num_diff'

    def predict_fn(X_df, gender):
        models = models_m if gender == 'M' else models_w
        weights = weights_m if gender == 'M' else weights_w
        diff_cols = [c for c in X_df.columns if c.endswith('_diff')]

        X = X_df[diff_cols]
        X_vals = X.values
        X_log = X[[seed_col]].fillna(0).values

        preds = {
            'logistic': models['logistic'].predict_proba(X_log)[:, 1],
            'ridge':    _predict_ridge(models['ridge'], X_vals),
            'xgboost':  models['xgboost'].predict_proba(X_vals)[:, 1],
            'lightgbm': models['lightgbm'].predict_proba(X_vals)[:, 1],
            'catboost': models['catboost'].predict_proba(X_vals)[:, 1],
        }

        blended = sum(w * preds[name] for name, w in weights.items())
        return clip_predictions(blended)

    return predict_fn


def validate_submission(path, expected_rows=None):
    df = pd.read_csv(path)
    print(f"  Rows: {len(df)}")
    print(f"  Pred range: [{df['Pred'].min():.4f}, {df['Pred'].max():.4f}]")
    print(f"  Pred mean:  {df['Pred'].mean():.4f}, std: {df['Pred'].std():.4f}")
    assert df['Pred'].notna().all(), "NaN predictions!"
    assert (df['Pred'] >= 0.05).all() and (df['Pred'] <= 0.95).all(), "Out of clip range!"
    assert df['Pred'].std() > 0.05, f"Std too low: {df['Pred'].std():.4f}"
    if expected_rows:
        assert len(df) == expected_rows, f"Expected {expected_rows}, got {len(df)}"
    print("  VALID")


def main():
    t_total = time.time()

    df_m = pd.read_csv('artifacts/features_men.csv')
    df_w = pd.read_csv('artifacts/features_women.csv')
    feature_cols = [c for c in df_m.columns if c.endswith('_diff')]

    with open('artifacts/tuned_params_barttorvik_v2.json') as f:
        tuned = json.load(f)

    # Old weights (pre-barttorvik) — for the "new HPs, old weights" variant
    with open('artifacts/tuned_params_pre_barttorvik.json') as f:
        old_params = json.load(f)
    old_weights = {g: old_params[g]['ensemble_weights'] for g in ['M', 'W']}

    print(f"Men: {df_m.shape}, Women: {df_w.shape}")
    print(f"Features: {len(feature_cols)}")

    for gender in ['M', 'W']:
        p = tuned[gender]
        print(f"\n{gender} new weights ({p['weight_method']}): {p['ensemble_weights']}")
        print(f"{gender} old weights:                        {old_weights[gender]}")

    # Train final models on all data
    t0 = time.time()
    print("\nTraining Men's models...")
    hp_m = {n: tuned['M'][n] for n in ['xgboost', 'lightgbm', 'catboost']}
    models_m = train_ensemble_models(df_m, feature_cols, model_params=hp_m)

    print("Training Women's models...")
    hp_w = {n: tuned['W'][n] for n in ['xgboost', 'lightgbm', 'catboost']}
    models_w = train_ensemble_models(df_w, feature_cols, model_params=hp_w)
    print(f"Training complete in {time.time() - t0:.1f}s")

    predict_fn = make_predict_fn(
        models_m, models_w,
        tuned['M']['ensemble_weights'],
        tuned['W']['ensemble_weights'],
    )

    # --- Variant A: new HPs + new weights ---
    t0 = time.time()
    predict_fn_new = make_predict_fn(
        models_m, models_w,
        tuned['M']['ensemble_weights'],
        tuned['W']['ensemble_weights'],
    )
    print("\n[Variant A] New HPs + new weights")
    print("Generating Stage 1...")
    generate_submission(
        predict_fn=predict_fn_new,
        stage=1,
        output_path='submissions/ensemble_barttorvik_v2_newweights_stage1.csv',
    )
    validate_submission('submissions/ensemble_barttorvik_v2_newweights_stage1.csv', expected_rows=519144)

    print("Generating Stage 2...")
    generate_submission(
        predict_fn=predict_fn_new,
        stage=2,
        output_path='submissions/ensemble_barttorvik_v2_newweights_2026.csv',
    )
    validate_submission('submissions/ensemble_barttorvik_v2_newweights_2026.csv')

    # --- Variant B: new HPs + old weights ---
    predict_fn_old = make_predict_fn(
        models_m, models_w,
        old_weights['M'],
        old_weights['W'],
    )
    print("\n[Variant B] New HPs + old weights")
    print("Generating Stage 1...")
    generate_submission(
        predict_fn=predict_fn_old,
        stage=1,
        output_path='submissions/ensemble_barttorvik_v2_oldweights_stage1.csv',
    )
    validate_submission('submissions/ensemble_barttorvik_v2_oldweights_stage1.csv', expected_rows=519144)

    print("Generating Stage 2...")
    generate_submission(
        predict_fn=predict_fn_old,
        stage=2,
        output_path='submissions/ensemble_barttorvik_v2_oldweights_2026.csv',
    )
    validate_submission('submissions/ensemble_barttorvik_v2_oldweights_2026.csv')

    print(f"\nSubmission generation complete in {time.time() - t0:.1f}s")
    print(f"Total time: {time.time() - t_total:.1f}s")
    print("\nFiles written:")
    print("  submissions/ensemble_barttorvik_v2_newweights_stage1.csv  (new HPs + new weights)")
    print("  submissions/ensemble_barttorvik_v2_oldweights_stage1.csv  (new HPs + old weights)")


if __name__ == '__main__':
    main()
