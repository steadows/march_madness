"""Tests for tuning infrastructure, EOA, and Ax/BoTorch tuners."""
import numpy as np
import pandas as pd
import pytest

from src.tuning import get_search_space, evaluate_params, evaluate_ensemble_weights


@pytest.fixture
def small_cv_df():
    """Small DataFrame mimicking tournament features for fast CV."""
    rng = np.random.RandomState(42)
    rows = []
    for season in range(2015, 2025):
        n = 20
        X = rng.randn(n, 3)
        y = (X[:, 0] > 0).astype(int)
        for i in range(n):
            rows.append({
                'Season': season,
                'target': y[i],
                'feat_a_diff': X[i, 0],
                'feat_b_diff': X[i, 1],
                'feat_c_diff': X[i, 2],
            })
    return pd.DataFrame(rows)


@pytest.fixture
def feature_cols():
    return ['feat_a_diff', 'feat_b_diff', 'feat_c_diff']


class TestGetSearchSpace:
    def test_xgboost_has_expected_params(self):
        space = get_search_space('xgboost')
        assert 'max_depth' in space
        assert 'learning_rate' in space
        assert 'subsample' in space
        assert space['max_depth']['type'] == 'int'
        assert space['learning_rate']['type'] == 'float'

    def test_lightgbm_has_expected_params(self):
        space = get_search_space('lightgbm')
        assert 'min_child_samples' in space
        assert space['min_child_samples']['type'] == 'int'

    def test_catboost_has_expected_params(self):
        space = get_search_space('catboost')
        assert 'depth' in space
        assert 'l2_leaf_reg' in space
        assert len(space) == 3  # fewer params than XGB/LGB

    def test_all_ranges_valid(self):
        for model_name in ['xgboost', 'lightgbm', 'catboost']:
            space = get_search_space(model_name)
            for name, spec in space.items():
                assert spec['low'] < spec['high'], f"{model_name}.{name}: low >= high"


class TestEvaluateParams:
    def test_returns_brier_in_valid_range(self, small_cv_df, feature_cols):
        params = {'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8,
                  'colsample_bytree': 0.8, 'min_child_weight': 3,
                  'reg_alpha': 0.1, 'reg_lambda': 1.0}
        result = evaluate_params('xgboost', params, small_cv_df, feature_cols, 'M')
        assert 0 < result['mean_brier'] < 0.5
        assert len(result['fold_briers']) == 5  # 2020-2024

    def test_catboost_params(self, small_cv_df, feature_cols):
        params = {'depth': 4, 'learning_rate': 0.1, 'l2_leaf_reg': 3.0}
        result = evaluate_params('catboost', params, small_cv_df, feature_cols, 'M')
        assert 0 < result['mean_brier'] < 0.5


class TestEvaluateEnsembleWeights:
    def test_equal_weights(self):
        rng = np.random.RandomState(42)
        y = rng.randint(0, 2, 100).astype(float)
        preds = {
            'a': np.clip(y + rng.randn(100) * 0.3, 0.05, 0.95),
            'b': np.clip(y + rng.randn(100) * 0.3, 0.05, 0.95),
        }
        brier = evaluate_ensemble_weights(
            np.array([0.5, 0.5]), preds, y, ['a', 'b'],
        )
        assert 0 < brier < 0.5

    def test_zero_weights_returns_high_brier(self):
        y = np.array([0, 1, 0, 1], dtype=float)
        preds = {
            'a': np.array([0.5, 0.5, 0.5, 0.5]),
        }
        brier = evaluate_ensemble_weights(
            np.array([0.0]), preds, y, ['a'],
        )
        assert brier == 1.0  # all-zero weights


class TestEOATuner:
    @pytest.mark.slow
    def test_eoa_returns_valid_params(self, small_cv_df, feature_cols):
        """Quick smoke test — 3 epochs, pop 10 (EOA needs pop >= 10 for crossover)."""
        from src.tuning_eoa import tune_model_eoa
        result = tune_model_eoa(
            'catboost', small_cv_df, feature_cols, 'M',
            epoch=3, pop_size=10,
        )
        assert 'best_params' in result
        assert 'best_brier' in result
        assert 0 < result['best_brier'] < 0.5
        # Params within bounds
        space = get_search_space('catboost')
        for name, val in result['best_params'].items():
            assert space[name]['low'] <= val <= space[name]['high'], \
                f"{name}={val} out of [{space[name]['low']}, {space[name]['high']}]"

    @pytest.mark.slow
    def test_eoa_weight_optimization(self):
        from src.tuning_eoa import tune_ensemble_weights_eoa
        rng = np.random.RandomState(42)
        y = rng.randint(0, 2, 100).astype(float)
        preds = {
            'a': np.clip(y + rng.randn(100) * 0.2, 0.05, 0.95),
            'b': np.clip(0.5 * np.ones(100), 0.05, 0.95),  # useless model
        }
        result = tune_ensemble_weights_eoa(preds, y, epoch=10, pop_size=10)
        assert abs(sum(result['weights'].values()) - 1.0) < 1e-6
        assert result['weights']['a'] > result['weights']['b']


class TestAxTuner:
    @pytest.mark.slow
    def test_ax_returns_valid_params(self, small_cv_df, feature_cols):
        """Quick smoke test — 5 trials."""
        from src.tuning_ax import tune_model_ax
        result = tune_model_ax(
            'catboost', small_cv_df, feature_cols, 'M',
            n_trials=5,
        )
        assert 'best_params' in result
        assert 'best_brier' in result
        assert 0 < result['best_brier'] < 0.5
        space = get_search_space('catboost')
        for name, val in result['best_params'].items():
            assert space[name]['low'] <= val <= space[name]['high']

    @pytest.mark.slow
    def test_ax_weight_optimization(self):
        from src.tuning_ax import tune_ensemble_weights_ax
        rng = np.random.RandomState(42)
        y = rng.randint(0, 2, 100).astype(float)
        preds = {
            'a': np.clip(y + rng.randn(100) * 0.2, 0.05, 0.95),
            'b': np.clip(0.5 * np.ones(100), 0.05, 0.95),
        }
        result = tune_ensemble_weights_ax(preds, y, n_trials=10)
        assert abs(sum(result['weights'].values()) - 1.0) < 1e-6
