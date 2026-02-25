"""Submission file generation for March Madness predictions."""
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src import config
from src import data_loader as dl
from src.calibration import clip_predictions
from src.elo import compute_elo_ratings
from src.feature_engineering import SeasonFeatureCache, build_matchup_features


def generate_submission(
    predict_fn: Callable[[pd.DataFrame, str], np.ndarray],
    stage: int = 1,
    output_path: Optional[Path] = None,
    elo_ratings_m: Optional[dict] = None,
    elo_ratings_w: Optional[dict] = None,
) -> pd.DataFrame:
    """Generate a valid submission CSV for the given stage.

    For each matchup ID in the sample submission, builds features and
    calls predict_fn to get a win probability. All predictions are clipped
    to [0.05, 0.95].

    Args:
        predict_fn: Callable(X: pd.DataFrame, gender: str) -> np.ndarray.
            Receives a feature DataFrame (rows = matchups, cols = _diff features)
            and gender string ('M' or 'W'). Returns probability array.
        stage: 1 (seasons 2022-2025) or 2 (season 2026).
        output_path: If provided, write CSV to this path.
        elo_ratings_m: Pre-computed M Elo ratings (recomputed if None).
        elo_ratings_w: Pre-computed W Elo ratings (recomputed if None).

    Returns:
        DataFrame with columns [ID, Pred] matching the sample submission order.
    """
    if elo_ratings_m is None:
        elo_ratings_m = compute_elo_ratings(dl.load_regular_season('M'))
    if elo_ratings_w is None:
        elo_ratings_w = compute_elo_ratings(dl.load_regular_season('W'))

    sub = dl.load_sample_submission(stage=stage)

    # Parse IDs into components
    parsed = sub['ID'].str.split('_', expand=True).astype(int)
    parsed.columns = ['season', 'team_a', 'team_b']

    pred_map: dict[str, float] = {}

    # Process each season × gender combination
    for season in sorted(parsed['season'].unique()):
        for gender, id_range, elo_ratings in [
            ('M', config.MEN_ID_RANGE, elo_ratings_m),
            ('W', config.WOMEN_ID_RANGE, elo_ratings_w),
        ]:
            mask = (
                (parsed['season'] == season)
                & parsed['team_a'].between(*id_range)
            )
            season_sub = parsed[mask]
            if len(season_sub) == 0:
                continue

            cache = SeasonFeatureCache(season, gender, elo_ratings)
            feature_rows = []
            ids = []

            for idx, row in season_sub.iterrows():
                team_a = int(row['team_a'])
                team_b = int(row['team_b'])
                feats = build_matchup_features(team_a, team_b, cache)
                feature_rows.append(feats)
                ids.append(sub.loc[idx, 'ID'])

            if not feature_rows:
                continue

            X = pd.DataFrame(feature_rows)
            feature_cols = [c for c in X.columns if c.endswith('_diff')]
            X_features = X[feature_cols]

            probs = predict_fn(X_features, gender)
            probs = clip_predictions(np.asarray(probs))

            for row_id, prob in zip(ids, probs):
                pred_map[row_id] = float(prob)

    # Align with original submission order
    result = pd.DataFrame({
        'ID': sub['ID'],
        'Pred': [pred_map.get(row_id, 0.5) for row_id in sub['ID']],
    })

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)

    return result


def make_predict_fn(
    model_m,
    model_w,
    feature_cols_m: Optional[list[str]] = None,
    feature_cols_w: Optional[list[str]] = None,
) -> Callable[[pd.DataFrame, str], np.ndarray]:
    """Create a predict_fn that routes to the correct gender model.

    Args:
        model_m: Fitted M model with predict_proba method.
        model_w: Fitted W model with predict_proba method.
        feature_cols_m: Feature columns expected by model_m (uses all _diff cols if None).
        feature_cols_w: Feature columns expected by model_w (uses all _diff cols if None).

    Returns:
        Callable(X: pd.DataFrame, gender: str) -> np.ndarray
    """
    def predict_fn(X: pd.DataFrame, gender: str) -> np.ndarray:
        if gender == 'M':
            model = model_m
            cols = feature_cols_m
        else:
            model = model_w
            cols = feature_cols_w

        if cols is None:
            cols = [c for c in X.columns if c.endswith('_diff')]

        X_input = X[cols] if cols else X
        return model.predict_proba(X_input)[:, 1]

    return predict_fn
