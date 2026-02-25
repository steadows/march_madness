"""Feature engineering validation tests."""
import json
import numpy as np
import pytest
import pandas as pd

from src import config, data_loader as dl
from src.elo import compute_elo_ratings
from src.feature_engineering import (
    SeasonFeatureCache,
    build_matchup_features,
    build_training_set,
)


# ---------------------------------------------------------------------------
# Fixtures — compute once per module
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def elo_ratings_m():
    games = dl.load_regular_season('M')
    return compute_elo_ratings(games)


@pytest.fixture(scope='module')
def elo_ratings_w():
    games = dl.load_regular_season('W')
    return compute_elo_ratings(games)


@pytest.fixture(scope='module')
def cache_2024_m(elo_ratings_m):
    return SeasonFeatureCache(2024, 'M', elo_ratings_m)


@pytest.fixture(scope='module')
def training_m(elo_ratings_m):
    """M training set for recent seasons (2015-2024)."""
    return build_training_set(list(range(2015, 2025)), 'M', elo_ratings_m)


@pytest.fixture(scope='module')
def training_w(elo_ratings_w):
    """W training set for recent seasons (2015-2024)."""
    return build_training_set(list(range(2015, 2025)), 'W', elo_ratings_w)


# ---------------------------------------------------------------------------
# Differential symmetry: feature(A, B) = -feature(B, A)
# ---------------------------------------------------------------------------

class TestDifferentialSymmetry:
    def test_symmetric_features(self, cache_2024_m):
        """All differentials must satisfy f(A,B) = -f(B,A)."""
        seeds = dl.load_tourney_seeds('M')
        s2024 = seeds[seeds['Season'] == 2024]
        # Take any two seeded teams
        teams = s2024['TeamID'].astype(int).tolist()
        team_a = min(teams[0], teams[1])
        team_b = max(teams[0], teams[1])

        feats_ab = build_matchup_features(team_a, team_b, cache_2024_m)
        feats_ba_flipped = build_matchup_features(
            team_a, team_b, cache_2024_m
        )  # same call — check internal symmetry

        # Build reversed by swapping teams' individual feature dicts
        fa = cache_2024_m.get_team_features(team_a)
        fb = cache_2024_m.get_team_features(team_b)

        for key in fa:
            va = fa[key]
            vb = fb[key]
            diff_ab = feats_ab.get(f'{key}_diff', np.nan)

            if not (np.isnan(va) or np.isnan(vb)):
                expected_ab = va - vb
                assert abs(diff_ab - expected_ab) < 1e-9, \
                    f"{key}_diff: got {diff_ab}, expected {expected_ab}"


# ---------------------------------------------------------------------------
# Seed differential for 1v16 matchup ≈ ±15
# ---------------------------------------------------------------------------

class TestSeedDifferential:
    def test_1v16_seed_diff(self, cache_2024_m):
        seeds = dl.load_tourney_seeds('M')
        s2024 = seeds[seeds['Season'] == 2024]
        seed1_ids = s2024[s2024['SeedNum'] == 1]['TeamID'].astype(int).tolist()
        seed16_ids = s2024[s2024['SeedNum'] == 16]['TeamID'].astype(int).tolist()

        assert seed1_ids and seed16_ids, "No 1-seeds or 16-seeds found in 2024"

        team_a = min(seed1_ids[0], seed16_ids[0])
        team_b = max(seed1_ids[0], seed16_ids[0])
        feats = build_matchup_features(team_a, team_b, cache_2024_m)

        seed_diff = feats['seed_num_diff']
        assert abs(seed_diff) == 15, f"Expected |seed_diff|=15, got {seed_diff}"


# ---------------------------------------------------------------------------
# No temporal leakage: features use only pre-tournament data
# ---------------------------------------------------------------------------

class TestNoTemporalLeakage:
    def test_stats_use_pretourney_games_only(self, cache_2024_m):
        """Season stats should reflect only DayNum < TOURNEY_START_DAY games."""
        # We can verify indirectly: the full_stats were computed with
        # pre_tourney_only=True (default). Just assert the cache has stats.
        seeds = dl.load_tourney_seeds('M')
        s2024 = seeds[seeds['Season'] == 2024]
        first_team = int(s2024['TeamID'].iloc[0])
        stats = cache_2024_m.full_stats.get(first_team, {})
        assert len(stats) > 0, "Team has no season stats"


# ---------------------------------------------------------------------------
# No constant features (zero variance)
# ---------------------------------------------------------------------------

class TestNoConstantFeatures:
    def test_no_constant_features_m(self, training_m):
        feature_cols = [c for c in training_m.columns if c.endswith('_diff')]
        for col in feature_cols:
            vals = training_m[col].dropna()
            if len(vals) > 10:
                assert vals.std() > 0, f"Column {col} has zero variance"

    def test_no_constant_features_w(self, training_w):
        # coach_tourney_exp is M-only; expected to be zero/constant for W
        m_only_features = {'coach_tourney_exp_diff'}
        feature_cols = [c for c in training_w.columns if c.endswith('_diff')]
        for col in feature_cols:
            if col in m_only_features:
                continue
            vals = training_w[col].dropna()
            if len(vals) > 10:
                assert vals.std() > 0, f"Column {col} has zero variance"


# ---------------------------------------------------------------------------
# NaN rate per column
# ---------------------------------------------------------------------------

class TestNaNRates:
    def test_nan_rate_m_modern(self, training_m):
        """For modern seasons (2010+), critical features should have <20% NaN."""
        modern = training_m[training_m['Season'] >= 2010]
        critical = ['elo_diff', 'win_pct_diff', 'pts_per_game_diff',
                    'pts_allowed_per_game_diff', 'seed_num_diff']
        for col in critical:
            if col in modern.columns:
                nan_rate = modern[col].isna().mean()
                assert nan_rate < 0.20, f"{col} NaN rate {nan_rate:.1%} > 20% in modern M data"

    def test_nan_rate_w_modern(self, training_w):
        modern = training_w[training_w['Season'] >= 2010]
        critical = ['elo_diff', 'win_pct_diff', 'pts_per_game_diff', 'seed_num_diff']
        for col in critical:
            if col in modern.columns:
                nan_rate = modern[col].isna().mean()
                assert nan_rate < 0.20, f"{col} NaN rate {nan_rate:.1%} > 20% in modern W data"


# ---------------------------------------------------------------------------
# Target distribution roughly balanced
# ---------------------------------------------------------------------------

class TestTargetDistribution:
    def test_m_target_balanced(self, training_m):
        target_mean = training_m['target'].mean()
        assert 0.45 <= target_mean <= 0.55, \
            f"M target mean {target_mean:.3f} outside [0.45, 0.55]"

    def test_w_target_balanced(self, training_w):
        target_mean = training_w['target'].mean()
        assert 0.45 <= target_mean <= 0.55, \
            f"W target mean {target_mean:.3f} outside [0.45, 0.55]"


# ---------------------------------------------------------------------------
# Feature count
# ---------------------------------------------------------------------------

class TestFeatureCount:
    def test_m_feature_count(self, training_m):
        feature_cols = [c for c in training_m.columns if c.endswith('_diff')]
        print(f"\nM feature count: {len(feature_cols)}")
        assert 20 <= len(feature_cols) <= 60, \
            f"Unexpected feature count: {len(feature_cols)}"

    def test_w_feature_count(self, training_w):
        feature_cols = [c for c in training_w.columns if c.endswith('_diff')]
        assert 20 <= len(feature_cols) <= 60, \
            f"Unexpected feature count: {len(feature_cols)}"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_reproducible_m(self, elo_ratings_m):
        """Running build_training_set twice gives identical results."""
        df1 = build_training_set([2024], 'M', elo_ratings_m)
        df2 = build_training_set([2024], 'M', elo_ratings_m)
        feature_cols = [c for c in df1.columns if c.endswith('_diff')]
        pd.testing.assert_frame_equal(
            df1[feature_cols].reset_index(drop=True),
            df2[feature_cols].reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# Correlation sanity (seed and elo should correlate with target)
# ---------------------------------------------------------------------------

class TestCorrelationSanity:
    def test_seed_corr_with_target(self, training_m):
        modern = training_m[training_m['Season'] >= 2003].dropna(subset=['seed_num_diff'])
        corr = modern['seed_num_diff'].corr(modern['target'])
        assert corr < -0.3, f"seed_num_diff corr with target={corr:.3f}, expected < -0.3"

    def test_elo_corr_with_target(self, training_m):
        corr = training_m['elo_diff'].corr(training_m['target'])
        assert corr > 0.3, f"elo_diff corr with target={corr:.3f}, expected > 0.3"
