"""Tests for src/ensemble.py and new model wrappers in src/models.py."""
import numpy as np
import pandas as pd
import pytest

from src.models import train_lightgbm, train_catboost, train_ridge
from src.ensemble import (
    simple_average_ensemble,
    optimize_ensemble_weights,
    weighted_ensemble,
    compute_sample_weights,
)


@pytest.fixture
def binary_dataset():
    """Small binary classification dataset with some NaN."""
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    # Inject some NaN
    X[0, 2] = np.nan
    X[5, 4] = np.nan
    return X, y


@pytest.fixture
def train_val_split(binary_dataset):
    """Split binary_dataset into train/val."""
    X, y = binary_dataset
    return X[:150], y[:150], X[150:], y[150:]


class TestTrainLightGBM:
    def test_returns_predict_proba(self, train_val_split):
        X_tr, y_tr, X_val, y_val = train_val_split
        model = train_lightgbm(X_tr, y_tr, X_val, y_val)
        probs = model.predict_proba(X_val)
        assert probs.shape == (50, 2)

    def test_predictions_in_unit_interval(self, train_val_split):
        X_tr, y_tr, X_val, y_val = train_val_split
        model = train_lightgbm(X_tr, y_tr)
        preds = model.predict_proba(X_val)[:, 1]
        assert np.all(preds >= 0) and np.all(preds <= 1)


class TestTrainCatBoost:
    def test_returns_predict_proba(self, train_val_split):
        X_tr, y_tr, X_val, y_val = train_val_split
        model = train_catboost(X_tr, y_tr, X_val, y_val)
        probs = model.predict_proba(X_val)
        assert probs.shape == (50, 2)

    def test_predictions_in_unit_interval(self, train_val_split):
        X_tr, y_tr, X_val, y_val = train_val_split
        model = train_catboost(X_tr, y_tr)
        preds = model.predict_proba(X_val)[:, 1]
        assert np.all(preds >= 0) and np.all(preds <= 1)


class TestTrainRidge:
    def test_returns_predict_proba(self, train_val_split):
        X_tr, y_tr, X_val, _ = train_val_split
        model = train_ridge(X_tr, y_tr)
        probs = model.predict_proba(X_val)
        assert probs.shape == (50, 2)

    def test_predictions_in_unit_interval(self, train_val_split):
        X_tr, y_tr, X_val, _ = train_val_split
        model = train_ridge(X_tr, y_tr)
        preds = model.predict_proba(X_val)[:, 1]
        assert np.all(preds >= 0) and np.all(preds <= 1)


class TestSimpleAverageEnsemble:
    def test_output_shape(self):
        preds = {
            'a': np.array([0.3, 0.7, 0.5]),
            'b': np.array([0.4, 0.6, 0.8]),
        }
        avg = simple_average_ensemble(preds)
        assert avg.shape == (3,)

    def test_values_are_mean(self):
        preds = {
            'a': np.array([0.2, 0.8]),
            'b': np.array([0.4, 0.6]),
        }
        avg = simple_average_ensemble(preds)
        np.testing.assert_allclose(avg, [0.3, 0.7])


class TestOptimizeEnsembleWeights:
    def test_weights_sum_to_one(self):
        rng = np.random.RandomState(42)
        y = rng.randint(0, 2, 100)
        preds = {
            'a': rng.uniform(0.2, 0.8, 100),
            'b': rng.uniform(0.2, 0.8, 100),
            'c': rng.uniform(0.2, 0.8, 100),
        }
        weights = optimize_ensemble_weights(preds, y)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_all_weights_non_negative(self):
        rng = np.random.RandomState(42)
        y = rng.randint(0, 2, 100)
        preds = {
            'a': rng.uniform(0.2, 0.8, 100),
            'b': rng.uniform(0.2, 0.8, 100),
        }
        weights = optimize_ensemble_weights(preds, y)
        assert all(w >= 0 for w in weights.values())


class TestWeightedEnsemble:
    def test_weighted_beats_or_matches_worst_single(self):
        rng = np.random.RandomState(42)
        y = (rng.randn(200) > 0).astype(float)
        # Model A: decent predictions
        a_preds = np.clip(y + rng.randn(200) * 0.3, 0.05, 0.95)
        # Model B: noisier
        b_preds = np.clip(y + rng.randn(200) * 0.5, 0.05, 0.95)
        preds = {'a': a_preds, 'b': b_preds}
        weights = optimize_ensemble_weights(preds, y)
        wtd = weighted_ensemble(preds, weights)
        from src.cv import evaluate_brier
        brier_wtd = evaluate_brier(y, wtd)
        brier_worst = max(evaluate_brier(y, a_preds), evaluate_brier(y, b_preds))
        assert brier_wtd <= brier_worst + 1e-6


class TestComputeSampleWeights:
    def test_most_recent_has_highest_weight(self):
        seasons = np.array([2015, 2015, 2016, 2017, 2017, 2018])
        w = compute_sample_weights(seasons)
        assert w[-1] > w[0]
        assert w[-2] > w[0]

    def test_shape(self):
        seasons = np.array([2020, 2021, 2022, 2020])
        w = compute_sample_weights(seasons)
        assert w.shape == (4,)

    def test_range(self):
        seasons = np.array([2010, 2015, 2020])
        w = compute_sample_weights(seasons)
        assert np.isclose(w[0], 0.3)
        assert np.isclose(w[-1], 1.0)
