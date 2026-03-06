"""Generate submissions using stacking meta-learner.

Pipeline:
1. Run 5-fold expanding-window CV to get OOF predictions from all 5 base models
2. Train logistic meta-learner on OOF predictions (+ optional seed_num_diff)
3. Train base models on ALL data
4. At inference: base model preds -> meta-learner -> calibration -> clip

Compares stacked vs weighted ensemble on OOF Brier score before generating.
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
    _predict_ridge,
    build_meta_features,
    meta_learner_predict,
    run_all_models_cv,
    train_meta_learner,
    weighted_ensemble,
)
from src.calibration import clip_predictions
from src.cv import evaluate_brier
from src.submission import generate_submission


EXTRA_META_COLS = ['seed_num_diff']


def train_ensemble_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = 'target',
    model_params: dict | None = None,
) -> dict:
    """Train all 5 base models on full dataset."""
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


def make_stacked_predict_fn(
    models_m: dict,
    models_w: dict,
    meta_m: dict,
    meta_w: dict,
    extra_meta_cols: list[str] | None = None,
):
    """Create predict_fn that pipes base model outputs through meta-learner."""
    seed_col = 'seed_num_diff'

    def predict_fn(X_df: pd.DataFrame, gender: str) -> np.ndarray:
        models = models_m if gender == 'M' else models_w
        meta = meta_m if gender == 'M' else meta_w
        diff_cols = [c for c in X_df.columns if c.endswith('_diff')]

        X = X_df[diff_cols]
        X_vals = X.values
        X_log = X[[seed_col]].fillna(0).values

        base_preds = {
            'logistic': models['logistic'].predict_proba(X_log)[:, 1],
            'ridge':    _predict_ridge(models['ridge'], X_vals),
            'xgboost':  models['xgboost'].predict_proba(X_vals)[:, 1],
            'lightgbm': models['lightgbm'].predict_proba(X_vals)[:, 1],
            'catboost': models['catboost'].predict_proba(X_vals)[:, 1],
        }

        extra = None
        if extra_meta_cols:
            extra = X_df[extra_meta_cols].fillna(0).values

        return meta_learner_predict(meta, base_preds, extra)

    return predict_fn


def make_weighted_predict_fn(models_m, models_w, weights_m, weights_w):
    """Create predict_fn using standard weighted ensemble (for comparison)."""
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


def validate_submission(path: str, expected_rows: int | None = None) -> None:
    """Validate a submission CSV."""
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

    # Load data and params
    df_m = pd.read_csv('artifacts/features_men.csv')
    df_w = pd.read_csv('artifacts/features_women.csv')
    feature_cols = [c for c in df_m.columns if c.endswith('_diff')]

    with open('artifacts/tuned_params_barttorvik_v2.json') as f:
        tuned = json.load(f)

    print(f"Men: {df_m.shape}, Women: {df_w.shape}")
    print(f"Features: {len(feature_cols)}")

    # ---- Step 1: Run CV to get OOF predictions ----
    print("\n=== Step 1: Running CV for OOF predictions ===")
    hp_m = {n: tuned['M'][n] for n in ['xgboost', 'lightgbm', 'catboost']}
    hp_w = {n: tuned['W'][n] for n in ['xgboost', 'lightgbm', 'catboost']}

    t0 = time.time()
    print("\nMen's CV...")
    cv_m = run_all_models_cv(df_m, feature_cols, gender='M', model_params=hp_m)
    print("\nWomen's CV...")
    cv_w = run_all_models_cv(df_w, feature_cols, gender='W', model_params=hp_w)
    print(f"CV complete in {time.time() - t0:.1f}s")

    # Print base model Brier scores
    for gender, cv in [('M', cv_m), ('W', cv_w)]:
        print(f"\n{gender} base model Brier scores:")
        for name in ['logistic', 'ridge', 'xgboost', 'lightgbm', 'catboost']:
            print(f"  {name}: {cv[name]['mean_brier']:.4f}")

    # ---- Step 2: Build meta-features & train meta-learner ----
    print("\n=== Step 2: Training stacking meta-learner ===")

    X_meta_m, y_meta_m = build_meta_features(cv_m, df_m, feature_cols, EXTRA_META_COLS)
    X_meta_w, y_meta_w = build_meta_features(cv_w, df_w, feature_cols, EXTRA_META_COLS)
    print(f"Meta features shape: M={X_meta_m.shape}, W={X_meta_w.shape}")

    meta_m = train_meta_learner(X_meta_m, y_meta_m, calibrate=True)
    meta_w = train_meta_learner(X_meta_w, y_meta_w, calibrate=True)

    # Evaluate stacked OOF Brier
    stacked_oof_m = meta_learner_predict(
        meta_m,
        {name: cv_m[name]['oof_preds'] for name in ['logistic', 'ridge', 'xgboost', 'lightgbm', 'catboost']},
        X_meta_m[:, 5:] if EXTRA_META_COLS else None,  # extra cols start at index 5
    )
    stacked_oof_w = meta_learner_predict(
        meta_w,
        {name: cv_w[name]['oof_preds'] for name in ['logistic', 'ridge', 'xgboost', 'lightgbm', 'catboost']},
        X_meta_w[:, 5:] if EXTRA_META_COLS else None,
    )

    stacked_brier_m = evaluate_brier(y_meta_m, stacked_oof_m)
    stacked_brier_w = evaluate_brier(y_meta_w, stacked_oof_w)

    # Compare with weighted ensemble OOF Brier
    oof_dict_m = {name: cv_m[name]['oof_preds'] for name in ['logistic', 'ridge', 'xgboost', 'lightgbm', 'catboost']}
    oof_dict_w = {name: cv_w[name]['oof_preds'] for name in ['logistic', 'ridge', 'xgboost', 'lightgbm', 'catboost']}

    wtd_m = weighted_ensemble(oof_dict_m, tuned['M']['ensemble_weights'])
    wtd_w = weighted_ensemble(oof_dict_w, tuned['W']['ensemble_weights'])
    wtd_brier_m = evaluate_brier(y_meta_m, wtd_m)
    wtd_brier_w = evaluate_brier(y_meta_w, wtd_w)

    print(f"\n{'Method':<25} {'Men Brier':>10} {'Women Brier':>12}")
    print(f"{'-'*47}")
    print(f"{'Weighted ensemble':<25} {wtd_brier_m:>10.4f} {wtd_brier_w:>12.4f}")
    print(f"{'Stacked meta-learner':<25} {stacked_brier_m:>10.4f} {stacked_brier_w:>12.4f}")
    diff_m = stacked_brier_m - wtd_brier_m
    diff_w = stacked_brier_w - wtd_brier_w
    print(f"{'Delta (stacked - wtd)':<25} {diff_m:>+10.4f} {diff_w:>+12.4f}")

    # Meta-learner coefficients
    model_names = ['logistic', 'ridge', 'xgboost', 'lightgbm', 'catboost']
    col_names = model_names + (EXTRA_META_COLS or [])
    print(f"\nMeta-learner coefficients:")
    for gender, meta in [('M', meta_m), ('W', meta_w)]:
        coefs = meta['model'].coef_[0]
        print(f"  {gender}: " + ", ".join(f"{n}={c:.3f}" for n, c in zip(col_names, coefs)))

    # ---- Step 3: Train base models on ALL data ----
    print("\n=== Step 3: Training final base models on all data ===")
    t0 = time.time()
    print("Training Men's models...")
    models_m = train_ensemble_models(df_m, feature_cols, model_params=hp_m)
    print("Training Women's models...")
    models_w = train_ensemble_models(df_w, feature_cols, model_params=hp_w)

    # Re-train meta-learner on full OOF (already done above — same data)
    # The meta-learner was trained on OOF preds which are proper held-out predictions
    print(f"Training complete in {time.time() - t0:.1f}s")

    # ---- Step 4: Generate submissions ----
    print("\n=== Step 4: Generating submissions ===")

    # Stacked submission
    predict_fn_stacked = make_stacked_predict_fn(
        models_m, models_w, meta_m, meta_w, EXTRA_META_COLS,
    )

    print("\n[Stacked] Generating Stage 1...")
    generate_submission(
        predict_fn=predict_fn_stacked,
        stage=1,
        output_path='submissions/ensemble_stacked_v1_stage1.csv',
    )
    validate_submission('submissions/ensemble_stacked_v1_stage1.csv', expected_rows=519144)

    print("[Stacked] Generating Stage 2...")
    generate_submission(
        predict_fn=predict_fn_stacked,
        stage=2,
        output_path='submissions/ensemble_stacked_v1_2026.csv',
    )
    validate_submission('submissions/ensemble_stacked_v1_2026.csv')

    # Also generate weighted ensemble (for A/B comparison on Kaggle)
    predict_fn_weighted = make_weighted_predict_fn(
        models_m, models_w,
        tuned['M']['ensemble_weights'],
        tuned['W']['ensemble_weights'],
    )

    print("\n[Weighted] Generating Stage 1...")
    generate_submission(
        predict_fn=predict_fn_weighted,
        stage=1,
        output_path='submissions/ensemble_weighted_v3_stage1.csv',
    )
    validate_submission('submissions/ensemble_weighted_v3_stage1.csv', expected_rows=519144)

    elapsed = time.time() - t_total
    print(f"\nTotal time: {elapsed:.1f}s")
    print("\nFiles written:")
    print("  submissions/ensemble_stacked_v1_stage1.csv     (STACKED meta-learner)")
    print("  submissions/ensemble_stacked_v1_2026.csv       (STACKED, stage 2)")
    print("  submissions/ensemble_weighted_v3_stage1.csv    (weighted baseline, same base models)")


if __name__ == '__main__':
    main()
