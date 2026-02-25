"""Cross-validation framework for NCAA tournament prediction.

Uses expanding window CV: train on seasons up to year N,
validate on season N+1. Never leaks future seasons into training.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def expanding_window_cv(
    df: pd.DataFrame,
    season_col: str = 'Season',
    min_train_end: int = 2019,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window train/validation index splits.

    Each fold: train = all seasons up to N, val = season N+1.
    Produces folds for val seasons: 2020, 2021, 2022, 2023, 2024.

    Args:
        df: DataFrame containing a season column.
        season_col: Name of the season column.
        min_train_end: Train window ends at this season; first val = min_train_end+1.

    Returns:
        List of (train_indices, val_indices) numpy arrays.
    """
    seasons = sorted(df[season_col].unique())
    val_seasons = [s for s in seasons if s > min_train_end]

    folds = []
    for val_season in val_seasons:
        train_idx = df.index[df[season_col] < val_season].to_numpy()
        val_idx = df.index[df[season_col] == val_season].to_numpy()
        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))

    return folds


def evaluate_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Brier Score (lower = better, coin flip = 0.25).

    Args:
        y_true: Binary labels (0 or 1).
        y_pred: Predicted probabilities in [0, 1].

    Returns:
        Brier score.
    """
    return float(brier_score_loss(y_true, y_pred))


def cv_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_fn,
    predict_fn,
    season_col: str = 'Season',
    min_train_end: int = 2019,
) -> dict:
    """Run expanding-window CV and return per-fold and mean Brier scores.

    Args:
        df: Feature DataFrame with season and target columns.
        feature_cols: List of feature column names.
        target_col: Name of the target column.
        train_fn: Callable(X_train, y_train) -> model.
        predict_fn: Callable(model, X_val) -> np.ndarray of probabilities.
        season_col: Name of the season column.
        min_train_end: First validation season = min_train_end + 1.

    Returns:
        Dict with keys: 'fold_briers', 'mean_brier', 'std_brier', 'fold_seasons'.
    """
    folds = expanding_window_cv(df, season_col=season_col, min_train_end=min_train_end)

    fold_briers = []
    fold_seasons = []

    for train_idx, val_idx in folds:
        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_val = val_df[feature_cols].values
        y_val = val_df[target_col].values

        model = train_fn(X_train, y_train)
        preds = predict_fn(model, X_val)
        brier = evaluate_brier(y_val, preds)

        fold_briers.append(brier)
        val_season = int(val_df[season_col].iloc[0])
        fold_seasons.append(val_season)

    return {
        'fold_briers': fold_briers,
        'mean_brier': float(np.mean(fold_briers)) if fold_briers else float('nan'),
        'std_brier': float(np.std(fold_briers)) if fold_briers else float('nan'),
        'fold_seasons': fold_seasons,
    }
