"""Tests for src/cv.py — cross-validation framework."""
import numpy as np
import pandas as pd
import pytest

from src.cv import expanding_window_cv, evaluate_brier


@pytest.fixture
def sample_df():
    """Small DataFrame with seasons 2015-2024, 10 rows each."""
    rows = []
    for season in range(2015, 2025):
        for _ in range(10):
            rows.append({'Season': season, 'target': np.random.randint(0, 2)})
    return pd.DataFrame(rows)


class TestExpandingWindowCV:
    def test_fold_count(self, sample_df):
        folds = expanding_window_cv(sample_df, min_train_end=2019)
        # Val seasons: 2020, 2021, 2022, 2023, 2024 = 5 folds
        assert len(folds) == 5

    def test_no_index_overlap(self, sample_df):
        folds = expanding_window_cv(sample_df, min_train_end=2019)
        for train_idx, val_idx in folds:
            overlap = set(train_idx) & set(val_idx)
            assert len(overlap) == 0, f"Train/val overlap: {overlap}"

    def test_val_seasons_after_train(self, sample_df):
        folds = expanding_window_cv(sample_df, min_train_end=2019)
        for train_idx, val_idx in folds:
            train_seasons = set(sample_df.loc[train_idx, 'Season'])
            val_seasons = set(sample_df.loc[val_idx, 'Season'])
            assert max(train_seasons) < min(val_seasons), \
                f"Train max season {max(train_seasons)} >= val min {min(val_seasons)}"

    def test_expanding_train_window(self, sample_df):
        """Each successive fold has more training data."""
        folds = expanding_window_cv(sample_df, min_train_end=2019)
        train_sizes = [len(t) for t, _ in folds]
        assert train_sizes == sorted(train_sizes), \
            f"Train sizes not expanding: {train_sizes}"

    def test_custom_min_train_end(self, sample_df):
        folds = expanding_window_cv(sample_df, min_train_end=2021)
        val_seasons = [
            set(sample_df.loc[v, 'Season']) for _, v in folds
        ]
        for vs in val_seasons:
            assert min(vs) >= 2022


class TestEvaluateBrier:
    def test_perfect_prediction(self):
        y = np.array([1, 0, 1, 0])
        preds = np.array([1.0, 0.0, 1.0, 0.0])
        assert evaluate_brier(y, preds) == pytest.approx(0.0)

    def test_coin_flip(self):
        y = np.array([1, 0, 1, 0] * 100)
        preds = np.full(400, 0.5)
        assert evaluate_brier(y, preds) == pytest.approx(0.25)

    def test_returns_float(self):
        assert isinstance(evaluate_brier(np.array([1, 0]), np.array([0.7, 0.3])), float)
