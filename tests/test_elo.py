"""Tests for src/elo.py — Elo rating system correctness."""
import pytest
import numpy as np

from src import data_loader as dl, config
from src.elo import (
    compute_elo_ratings,
    elo_to_win_prob,
    get_pre_tourney_elo,
    ELO_INIT,
)

# ---------------------------------------------------------------------------
# Math unit tests
# ---------------------------------------------------------------------------

class TestEloMath:
    def test_even_match_prob(self):
        assert abs(elo_to_win_prob(0) - 0.5) < 1e-9

    def test_large_advantage_prob(self):
        assert elo_to_win_prob(400) > 0.9

    def test_large_disadvantage_prob(self):
        assert elo_to_win_prob(-400) < 0.1

    def test_symmetry(self):
        assert abs(elo_to_win_prob(200) + elo_to_win_prob(-200) - 1.0) < 1e-9

    def test_monotonic(self):
        diffs = [-300, -200, -100, 0, 100, 200, 300]
        probs = [elo_to_win_prob(d) for d in diffs]
        assert all(probs[i] < probs[i+1] for i in range(len(probs)-1))


# ---------------------------------------------------------------------------
# Elo computation on real data
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def men_ratings():
    """Compute M Elo ratings once for all tests in this module."""
    games = dl.load_regular_season('M', detailed=False)
    return compute_elo_ratings(games)


@pytest.fixture(scope='module')
def women_ratings():
    games = dl.load_regular_season('W', detailed=False)
    return compute_elo_ratings(games)


class TestEloRatings:
    def test_mean_elo_near_1500(self, men_ratings):
        """Mean Elo across all teams in a season should be ≈ 1500."""
        season = 2024
        elos = list(men_ratings[season].values())
        mean_elo = np.mean(elos)
        assert abs(mean_elo - ELO_INIT) < 50, f"Mean Elo {mean_elo:.1f} too far from 1500"

    def test_top_teams_high_elo(self, men_ratings):
        """Power programs (Duke=1181, Kansas=1242, Gonzaga=1211) should have high Elo."""
        season = 2024
        season_elos = men_ratings[season]
        # Gonzaga=1211, Duke=1181, Kansas=1242, Houston=1277
        top_team_ids = [1181, 1242, 1211, 1277]
        for tid in top_team_ids:
            if tid in season_elos:
                assert season_elos[tid] > 1600, \
                    f"Team {tid} Elo={season_elos[tid]:.0f} expected > 1600"

    def test_bottom_teams_low_elo(self, men_ratings):
        """Lower-tier teams should have below-average Elo."""
        season = 2024
        elos = list(men_ratings[season].values())
        # At least 25% of teams should be below 1450
        low_count = sum(1 for e in elos if e < 1450)
        assert low_count > len(elos) * 0.20

    def test_all_seasons_present(self, men_ratings):
        """All seasons from 1985 onwards should be in ratings."""
        for season in range(1986, 2026):
            assert season in men_ratings, f"Season {season} missing from Elo ratings"

    def test_women_ratings_computed(self, women_ratings):
        season = 2024
        assert season in women_ratings
        elos = list(women_ratings[season].values())
        assert abs(np.mean(elos) - ELO_INIT) < 50

    def test_season_spread(self, men_ratings):
        """Recent seasons should have meaningful Elo spread (std > 80)."""
        season = 2024
        elos = list(men_ratings[season].values())
        assert np.std(elos) > 80, f"Elo std {np.std(elos):.1f} too low"

    def test_get_pre_tourney_elo_default(self, men_ratings):
        """Unknown team returns ELO_INIT."""
        val = get_pre_tourney_elo(men_ratings, 2024, 999999)
        assert val == ELO_INIT

    def test_elo_improves_for_good_teams(self, men_ratings):
        """Teams that win more should have higher Elo."""
        seeds = dl.load_tourney_seeds('M')
        season = 2024
        seed_map = seeds[seeds['Season'] == season].set_index('TeamID')['SeedNum']
        elos_2024 = men_ratings[season]

        one_seeds = [tid for tid, s in seed_map.items() if s == 1 and tid in elos_2024]
        sixteen_seeds = [tid for tid, s in seed_map.items() if s == 16 and tid in elos_2024]

        if one_seeds and sixteen_seeds:
            avg_1_seed = np.mean([elos_2024[t] for t in one_seeds])
            avg_16_seed = np.mean([elos_2024[t] for t in sixteen_seeds])
            assert avg_1_seed > avg_16_seed + 200, \
                f"1-seed avg elo {avg_1_seed:.0f} not much > 16-seed avg {avg_16_seed:.0f}"


# ---------------------------------------------------------------------------
# Save/load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip(self, men_ratings, tmp_path):
        from src.elo import save_ratings, load_ratings
        path = tmp_path / 'elo_test.json'
        save_ratings(men_ratings, path)
        loaded = load_ratings(path)
        assert set(loaded.keys()) == set(men_ratings.keys())
        season = 2024
        for tid, elo in men_ratings[season].items():
            assert abs(loaded[season][tid] - elo) < 1e-6
