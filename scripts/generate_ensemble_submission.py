"""Train final ensemble on all data and generate submissions."""
import json
import sys
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


def train_ensemble_models(df, feature_cols, target_col='target'):
    """Train all 5 models on full dataset."""
    X = df[feature_cols]
    y = df[target_col].values

    seed_col = 'seed_num_diff'
    models = {}

    # Logistic: seed_diff only
    X_log = X[[seed_col]].fillna(0).values
    models['logistic'] = train_logistic_baseline(X_log, y)

    # Ridge: all features
    models['ridge'] = train_ridge(X.values, y)

    # XGBoost
    models['xgboost'] = train_xgboost(X.values, y)

    # LightGBM
    models['lightgbm'] = train_lightgbm(X.values, y)

    # CatBoost
    models['catboost'] = train_catboost(X.values, y)

    return models


def make_ensemble_predict_fn(models_m, models_w, weights_m, weights_w, feature_cols):
    """Create predict_fn for generate_submission that uses weighted ensemble."""
    seed_col = 'seed_num_diff'

    def predict_fn(X_df, gender):
        models = models_m if gender == 'M' else models_w
        weights = weights_m if gender == 'M' else weights_w
        diff_cols = [c for c in X_df.columns if c.endswith('_diff')]

        X = X_df[diff_cols]
        X_vals = X.values

        preds = {}

        # Logistic
        X_log = X[[seed_col]].fillna(0).values
        preds['logistic'] = models['logistic'].predict_proba(X_log)[:, 1]

        # Ridge
        preds['ridge'] = _predict_ridge(models['ridge'], X_vals)

        # XGBoost
        preds['xgboost'] = models['xgboost'].predict_proba(X_vals)[:, 1]

        # LightGBM
        preds['lightgbm'] = models['lightgbm'].predict_proba(X_vals)[:, 1]

        # CatBoost
        preds['catboost'] = models['catboost'].predict_proba(X_vals)[:, 1]

        # Weighted average
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
    # Load data and weights
    df_m = pd.read_csv('artifacts/features_men.csv')
    df_w = pd.read_csv('artifacts/features_women.csv')
    feature_cols = [c for c in df_m.columns if c.endswith('_diff')]

    with open('artifacts/ensemble_weights.json') as f:
        all_weights = json.load(f)
    weights_m = all_weights['M']
    weights_w = all_weights['W']

    print(f"Training on M: {df_m.shape}, W: {df_w.shape}")
    print(f"M weights: {weights_m}")
    print(f"W weights: {weights_w}")
    print()

    # Train final models on ALL data
    print("Training Men's models on all data...")
    models_m = train_ensemble_models(df_m, feature_cols)
    print("Training Women's models on all data...")
    models_w = train_ensemble_models(df_w, feature_cols)

    predict_fn = make_ensemble_predict_fn(
        models_m, models_w, weights_m, weights_w, feature_cols
    )

    # Generate Stage 1 submission
    print("\nGenerating Stage 1 submission...")
    generate_submission(
        predict_fn=predict_fn,
        stage=1,
        output_path='submissions/ensemble_v1.csv',
    )
    print("Validating ensemble_v1.csv:")
    validate_submission('submissions/ensemble_v1.csv', expected_rows=519144)

    # Generate Stage 2 submission
    print("\nGenerating Stage 2 submission...")
    generate_submission(
        predict_fn=predict_fn,
        stage=2,
        output_path='submissions/ensemble_2026.csv',
    )
    print("Validating ensemble_2026.csv:")
    validate_submission('submissions/ensemble_2026.csv')

    print("\nDone!")


if __name__ == '__main__':
    main()
