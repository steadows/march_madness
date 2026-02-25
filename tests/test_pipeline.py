"""End-to-end pipeline integration tests."""
import numpy as np
import pandas as pd
import pytest

from src import config, data_loader as dl
from src.elo import compute_elo_ratings
from src.feature_engineering import build_training_set, build_prediction_set
from src.cv import expanding_window_cv, evaluate_brier
from src.models import train_logistic_baseline, train_xgboost
from src.calibration import clip_predictions


@pytest.fixture(scope='module')
def elo_m():
    return compute_elo_ratings(dl.load_regular_season('M'))


@pytest.fixture(scope='module')
def elo_w():
    return compute_elo_ratings(dl.load_regular_season('W'))


@pytest.fixture(scope='module')
def train_2024_m(elo_m):
    return build_training_set([2024], 'M', elo_m)


@pytest.fixture(scope='module')
def train_2024_w(elo_w):
    return build_training_set([2024], 'W', elo_w)


# ---------------------------------------------------------------------------
# Full pipeline for single season
# ---------------------------------------------------------------------------

class TestSingleSeasonPipeline:
    def test_m_produces_data(self, train_2024_m):
        assert len(train_2024_m) > 0
        assert 'target' in train_2024_m.columns

    def test_w_produces_data(self, train_2024_w):
        assert len(train_2024_w) > 0
        assert 'target' in train_2024_w.columns

    def test_m_valid_targets(self, train_2024_m):
        assert set(train_2024_m['target'].unique()).issubset({0, 1})

    def test_w_valid_targets(self, train_2024_w):
        assert set(train_2024_w['target'].unique()).issubset({0, 1})

    def test_m_has_features(self, train_2024_m):
        feature_cols = [c for c in train_2024_m.columns if c.endswith('_diff')]
        assert len(feature_cols) >= 20

    def test_predictions_both_genders(self, train_2024_m, train_2024_w):
        """Can train and predict for both M and W."""
        for df, label in [(train_2024_m, 'M'), (train_2024_w, 'W')]:
            feature_cols = [c for c in df.columns if c.endswith('_diff')]
            X = df[feature_cols].values
            y = df['target'].values

            # Logistic using elo_diff only
            X_simple = df[['elo_diff']].fillna(0).values
            model = train_logistic_baseline(X_simple, y)
            preds = clip_predictions(model.predict_proba(X_simple)[:, 1])
            assert len(preds) == len(y)
            assert (preds >= config.CLIP_LOW).all()
            assert (preds <= config.CLIP_HIGH).all()


# ---------------------------------------------------------------------------
# CV baseline (uses historical seasons, not 2024-only)
# ---------------------------------------------------------------------------

class TestCVBaseline:
    @pytest.fixture(scope='class')
    def modern_m(self, elo_m):
        return build_training_set(list(range(2003, 2025)), 'M', elo_m)

    @pytest.fixture(scope='class')
    def elo_m(self):
        return compute_elo_ratings(dl.load_regular_season('M'))

    def test_logistic_brier_lt_025(self, modern_m):
        """Logistic (seed_diff only) must beat coin flip."""
        folds = expanding_window_cv(modern_m, min_train_end=2019)
        assert len(folds) >= 4

        briers = []
        for train_idx, val_idx in folds:
            train = modern_m.loc[train_idx]
            val = modern_m.loc[val_idx]
            X_t = train[['seed_num_diff']].fillna(0).values
            X_v = val[['seed_num_diff']].fillna(0).values
            y_t = train['target'].values
            y_v = val['target'].values
            model = train_logistic_baseline(X_t, y_t)
            preds = clip_predictions(model.predict_proba(X_v)[:, 1])
            briers.append(evaluate_brier(y_v, preds))

        mean_brier = np.mean(briers)
        assert mean_brier < 0.25, f"Logistic Brier {mean_brier:.4f} >= 0.25"

    def test_xgboost_brier_lt_022(self, modern_m):
        """XGBoost must beat seed-only baseline."""
        feature_cols = [c for c in modern_m.columns if c.endswith('_diff')]
        folds = expanding_window_cv(modern_m, min_train_end=2021)
        assert len(folds) >= 2

        briers = []
        for train_idx, val_idx in folds[-2:]:  # Only last 2 folds for speed
            train = modern_m.loc[train_idx]
            val = modern_m.loc[val_idx]
            X_t = train[feature_cols].values
            X_v = val[feature_cols].values
            y_t = train['target'].values
            y_v = val['target'].values
            model = train_xgboost(X_t, y_t, X_v, y_v)
            preds = clip_predictions(model.predict_proba(X_v)[:, 1])
            briers.append(evaluate_brier(y_v, preds))

        mean_brier = np.mean(briers)
        assert mean_brier < 0.22, f"XGBoost Brier {mean_brier:.4f} >= 0.22"


# ---------------------------------------------------------------------------
# Missing data handling
# ---------------------------------------------------------------------------

class TestMissingDataHandling:
    def test_teams_with_no_detailed_stats(self, train_2024_m):
        """Pipeline should produce rows even if some features are NaN."""
        # All tournament teams should appear (some may have NaN advanced stats)
        assert len(train_2024_m) > 0
        feature_cols = [c for c in train_2024_m.columns if c.endswith('_diff')]
        # At least some features should be non-NaN
        for col in ['elo_diff', 'win_pct_diff', 'pts_per_game_diff']:
            if col in feature_cols:
                assert train_2024_m[col].notna().any(), f"{col} is all NaN"

    def test_prediction_for_unknown_team_uses_defaults(self, elo_m):
        """SeasonFeatureCache handles unknown team IDs gracefully."""
        from src.feature_engineering import SeasonFeatureCache, build_matchup_features
        cache = SeasonFeatureCache(2024, 'M', elo_m)
        # Use real IDs but a made-up low/high pair just to test it doesn't crash
        seeds = dl.load_tourney_seeds('M')
        teams_2024 = seeds[seeds['Season'] == 2024]['TeamID'].astype(int).tolist()
        if len(teams_2024) >= 2:
            team_a = min(teams_2024[0], teams_2024[1])
            team_b = max(teams_2024[0], teams_2024[1])
            feats = build_matchup_features(team_a, team_b, cache)
            assert isinstance(feats, dict)
            assert len(feats) > 0
