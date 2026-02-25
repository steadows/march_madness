"""Model training wrappers for NCAA tournament prediction."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier


def train_logistic_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    features: list[str] | None = None,
) -> Pipeline:
    """Train logistic regression on seed_diff only (or all provided features).

    Intended as a floor baseline: Brier should be < 0.25.

    Args:
        X_train: Training features array.
        y_train: Binary target array.
        features: Feature names (for selecting seed_diff column if X is a DataFrame).

    Returns:
        Fitted sklearn Pipeline (StandardScaler + LogisticRegression).
    """
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    params: dict | None = None,
) -> XGBClassifier:
    """Train XGBoost classifier with optional early stopping.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features for early stopping (optional).
        y_val: Validation targets for early stopping (optional).
        params: XGBoost hyperparameters. Uses sensible defaults if None.

    Returns:
        Fitted XGBClassifier.
    """
    default_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_estimators': 500,
        'verbosity': 0,
        'random_state': 42,
    }
    if params:
        default_params.update(params)

    use_early_stopping = X_val is not None and y_val is not None
    if use_early_stopping:
        default_params['early_stopping_rounds'] = 50

    model = XGBClassifier(**default_params)

    if use_early_stopping:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)

    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    params: dict | None = None,
) -> lgb.LGBMClassifier:
    """Train LightGBM classifier with optional early stopping.

    Returns fitted LGBMClassifier with .predict_proba() interface.
    """
    default_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_estimators': 500,
        'verbose': -1,
        'random_state': 42,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)

    use_early_stopping = X_val is not None and y_val is not None
    if use_early_stopping:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )
    else:
        model.fit(X_train, y_train)

    return model


def train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    params: dict | None = None,
) -> CatBoostClassifier:
    """Train CatBoost classifier with optional early stopping.

    Returns fitted CatBoostClassifier with .predict_proba() interface.
    """
    default_params = {
        'iterations': 500,
        'depth': 6,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'eval_metric': 'Logloss',
        'verbose': 0,
        'random_state': 42,
    }
    if params:
        default_params.update(params)

    model = CatBoostClassifier(**default_params)

    use_early_stopping = X_val is not None and y_val is not None
    if use_early_stopping:
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
        )
    else:
        model.fit(X_train, y_train)

    return model


def train_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Pipeline:
    """Train L2-regularized logistic regression (Ridge equivalent).

    Uses StandardScaler + LogisticRegression(C=0.1, penalty='l2').
    Handles NaN by filling with column medians before scaling.

    Returns fitted Pipeline with .predict_proba() interface.
    """
    X_train = np.asarray(X_train, dtype=float)
    # Fill NaN with column medians for linear model
    col_medians = np.nanmedian(X_train, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    mask = np.isnan(X_train)
    X_filled = X_train.copy()
    for j in range(X_filled.shape[1]):
        X_filled[mask[:, j], j] = col_medians[j]

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=0.1, l1_ratio=0, max_iter=1000, random_state=42)),
    ])
    # Store medians for use at predict time
    pipe.fit(X_filled, y_train)
    pipe._col_medians = col_medians
    return pipe


def get_feature_importances(model, feature_names: list[str]) -> pd.Series:
    """Extract feature importances from a trained model.

    Args:
        model: Fitted model with feature_importances_ attribute.
        feature_names: Feature column names.

    Returns:
        Series of importances sorted descending.
    """
    if hasattr(model, 'feature_importances_'):
        return pd.Series(
            model.feature_importances_, index=feature_names
        ).sort_values(ascending=False)
    elif hasattr(model, 'named_steps'):
        # Pipeline — get from final estimator
        clf = model.named_steps.get('clf')
        if hasattr(clf, 'coef_'):
            return pd.Series(
                np.abs(clf.coef_[0]), index=feature_names
            ).sort_values(ascending=False)
    return pd.Series(dtype=float)


def run_cv_baseline(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = 'target',
    gender: str = 'M',
    output_path=None,
) -> dict:
    """Run both logistic and XGBoost baselines through expanding-window CV.

    Trains on seasons up to 2019, validates on 2020-2024.
    Handles NaN features by median imputation for logistic, native for XGBoost.

    Args:
        df: Feature DataFrame with Season and target columns.
        feature_cols: Feature column names.
        target_col: Name of target column.
        gender: 'M' or 'W' (for logging).
        output_path: If provided, write results to this file.

    Returns:
        Dict with 'logistic' and 'xgboost' CV results.
    """
    from src.cv import expanding_window_cv, evaluate_brier
    from src.calibration import clip_predictions

    folds = expanding_window_cv(df, min_train_end=2019)

    logistic_briers = []
    xgb_briers = []
    fold_seasons = []

    for train_idx, val_idx in folds:
        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]

        X_train_raw = train_df[feature_cols]
        X_val_raw = val_df[feature_cols]
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values

        val_season = int(val_df['Season'].iloc[0])
        fold_seasons.append(val_season)

        # --- Logistic: use seed_diff only (most reliable single feature)
        seed_col = 'seed_num_diff'
        if seed_col in feature_cols:
            X_log_train = X_train_raw[[seed_col]].fillna(0).values
            X_log_val = X_val_raw[[seed_col]].fillna(0).values
        else:
            # Fallback: use elo_diff
            X_log_train = X_train_raw[['elo_diff']].fillna(0).values
            X_log_val = X_val_raw[['elo_diff']].fillna(0).values

        log_model = train_logistic_baseline(X_log_train, y_train)
        log_preds = clip_predictions(log_model.predict_proba(X_log_val)[:, 1])
        logistic_briers.append(evaluate_brier(y_val, log_preds))

        # --- XGBoost: all features (handles NaN natively)
        X_xgb_train = X_train_raw.values
        X_xgb_val = X_val_raw.values
        xgb_model = train_xgboost(X_xgb_train, y_train, X_xgb_val, y_val)
        xgb_preds = clip_predictions(xgb_model.predict_proba(X_xgb_val)[:, 1])
        xgb_briers.append(evaluate_brier(y_val, xgb_preds))

    results = {
        'logistic': {
            'fold_briers': logistic_briers,
            'fold_seasons': fold_seasons,
            'mean_brier': float(np.mean(logistic_briers)),
            'std_brier': float(np.std(logistic_briers)),
        },
        'xgboost': {
            'fold_briers': xgb_briers,
            'fold_seasons': fold_seasons,
            'mean_brier': float(np.mean(xgb_briers)),
            'std_brier': float(np.std(xgb_briers)),
        },
    }

    lines = [
        f'=== Baseline CV Results — {gender} ===',
        '',
        'Logistic (seed_diff only):',
    ]
    for season, brier in zip(fold_seasons, logistic_briers):
        lines.append(f'  {season}: {brier:.4f}')
    lines.append(f'  Mean: {results["logistic"]["mean_brier"]:.4f} '
                 f'± {results["logistic"]["std_brier"]:.4f}')
    lines.append('')
    lines.append('XGBoost (all features):')
    for season, brier in zip(fold_seasons, xgb_briers):
        lines.append(f'  {season}: {brier:.4f}')
    lines.append(f'  Mean: {results["xgboost"]["mean_brier"]:.4f} '
                 f'± {results["xgboost"]["std_brier"]:.4f}')

    report = '\n'.join(lines)
    print(report)

    if output_path:
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'a') as f:
            f.write(report + '\n\n')

    return results
