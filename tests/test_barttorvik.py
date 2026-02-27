"""Tests for Barttorvik data loading and team name mapping."""

import numpy as np
import pandas as pd
import pytest
from src.barttorvik import (
    build_name_mapping,
    load_barttorvik_ratings,
    BARTTORVIK_FEATURES,
    _CONF_NAMES,
    _MANUAL_OVERRIDES_M,
    _MANUAL_OVERRIDES_W,
)


class TestNameMapping:
    """Tests for build_name_mapping()."""

    def test_m_mapping_no_unmatched(self):
        """Every men's Barttorvik team name maps to a TeamID."""
        mapping, unmatched = build_name_mapping("M")
        assert len(unmatched) == 0, f"Unmatched M teams: {unmatched}"

    def test_w_mapping_no_unmatched(self):
        """Every women's Barttorvik team name maps to a TeamID."""
        mapping, unmatched = build_name_mapping("W")
        assert len(unmatched) == 0, f"Unmatched W teams: {unmatched}"

    def test_m_mapping_has_reasonable_count(self):
        """Men's mapping should have 350-400 teams."""
        mapping, _ = build_name_mapping("M")
        assert 350 <= len(mapping) <= 410

    def test_w_mapping_has_reasonable_count(self):
        """Women's mapping should have 350-400 teams."""
        mapping, _ = build_name_mapping("W")
        assert 350 <= len(mapping) <= 410

    def test_m_team_ids_in_valid_range(self):
        """Men's TeamIDs should be in the 1xxx range."""
        mapping, _ = build_name_mapping("M")
        for name, tid in mapping.items():
            assert 1000 <= tid <= 1999, f"{name} has invalid M TeamID: {tid}"

    def test_w_team_ids_in_valid_range(self):
        """Women's TeamIDs should be in the 3xxx range."""
        mapping, _ = build_name_mapping("W")
        for name, tid in mapping.items():
            assert 3000 <= tid <= 3999, f"{name} has invalid W TeamID: {tid}"

    def test_no_duplicate_team_ids_m(self):
        """Each TeamID should map from at most one Barttorvik name."""
        mapping, _ = build_name_mapping("M")
        id_to_names = {}
        for name, tid in mapping.items():
            id_to_names.setdefault(tid, []).append(name)
        dupes = {tid: names for tid, names in id_to_names.items() if len(names) > 1}
        assert len(dupes) == 0, f"Duplicate TeamID mappings: {dupes}"

    def test_no_duplicate_team_ids_w(self):
        """Each W TeamID should map from at most one Barttorvik name."""
        mapping, _ = build_name_mapping("W")
        id_to_names = {}
        for name, tid in mapping.items():
            id_to_names.setdefault(tid, []).append(name)
        dupes = {tid: names for tid, names in id_to_names.items() if len(names) > 1}
        assert len(dupes) == 0, f"Duplicate TeamID mappings: {dupes}"

    def test_known_teams_m(self):
        """Spot-check known men's team mappings."""
        mapping, _ = build_name_mapping("M")
        # These are stable, well-known programs
        assert mapping["duke"] == 1181
        assert mapping["north carolina"] == 1314
        assert mapping["kansas"] == 1242
        assert mapping["kentucky"] == 1246
        assert mapping["michigan"] == 1276

    def test_known_teams_w(self):
        """Spot-check known women's team mappings."""
        mapping, _ = build_name_mapping("W")
        assert mapping["connecticut"] == 3163
        assert mapping["south carolina"] == 3376
        assert mapping["stanford"] == 3390

    def test_manual_overrides_used(self):
        """Manual overrides should be reachable (not shadowed by other strategies)."""
        mapping, _ = build_name_mapping("M")
        for name, expected_id in _MANUAL_OVERRIDES_M.items():
            assert name in mapping, f"Override '{name}' not in mapping"
            assert mapping[name] == expected_id, (
                f"Override '{name}': expected {expected_id}, got {mapping[name]}"
            )

    def test_conf_names_excluded(self):
        """Conference summary rows should not appear in mapping."""
        mapping, _ = build_name_mapping("M")
        for conf in _CONF_NAMES:
            assert conf.lower() not in mapping, f"Conf '{conf}' found in mapping"

    def test_tournament_teams_covered_m(self):
        """Every men's tournament team (2008-2025) should have a mapping."""
        mapping, _ = build_name_mapping("M")
        mapped_ids = set(mapping.values())
        seeds = pd.read_csv("data/MNCAATourneySeeds.csv")
        tourney_ids = set(seeds[seeds["Season"].between(2008, 2025)]["TeamID"].unique())
        missing = tourney_ids - mapped_ids
        assert len(missing) == 0, f"Tournament teams missing from mapping: {missing}"

    def test_tournament_teams_covered_w(self):
        """Every women's tournament team (2021-2025) should have a mapping."""
        mapping, _ = build_name_mapping("W")
        mapped_ids = set(mapping.values())
        seeds = pd.read_csv("data/WNCAATourneySeeds.csv")
        tourney_ids = set(seeds[seeds["Season"].between(2021, 2025)]["TeamID"].unique())
        missing = tourney_ids - mapped_ids
        assert len(missing) == 0, f"Tournament teams missing from mapping: {missing}"


class TestLoadRatings:
    """Tests for load_barttorvik_ratings()."""

    @pytest.fixture(scope="class")
    def m_ratings(self):
        return load_barttorvik_ratings("M")

    @pytest.fixture(scope="class")
    def w_ratings(self):
        return load_barttorvik_ratings("W")

    def test_m_shape(self, m_ratings):
        """Men's data should have ~6600+ rows and 11 columns."""
        assert m_ratings.shape[1] == 11
        assert m_ratings.shape[0] > 6000

    def test_w_shape(self, w_ratings):
        """Women's data should have ~2100+ rows and 11 columns."""
        assert w_ratings.shape[1] == 11
        assert w_ratings.shape[0] > 2000

    def test_m_seasons(self, m_ratings):
        """Men's data should span 2008-2026."""
        assert m_ratings["Season"].min() == 2008
        assert m_ratings["Season"].max() == 2026

    def test_w_seasons(self, w_ratings):
        """Women's data should span 2021-2026."""
        assert w_ratings["Season"].min() == 2021
        assert w_ratings["Season"].max() == 2026

    def test_no_duplicate_season_team_m(self, m_ratings):
        """No duplicate (Season, TeamID) rows in men's data."""
        dupes = m_ratings.duplicated(subset=["Season", "TeamID"], keep=False)
        assert dupes.sum() == 0

    def test_no_duplicate_season_team_w(self, w_ratings):
        """No duplicate (Season, TeamID) rows in women's data."""
        dupes = w_ratings.duplicated(subset=["Season", "TeamID"], keep=False)
        assert dupes.sum() == 0

    def test_columns_present(self, m_ratings):
        """All expected columns should be present."""
        expected = ["Season", "TeamID"] + BARTTORVIK_FEATURES
        assert list(m_ratings.columns) == expected

    def test_feature_dtypes(self, m_ratings):
        """All feature columns should be float64."""
        for col in BARTTORVIK_FEATURES:
            assert m_ratings[col].dtype == "float64", f"{col} is {m_ratings[col].dtype}"

    def test_m_team_ids_valid(self, m_ratings):
        """Men's TeamIDs should be in 1xxx range."""
        assert m_ratings["TeamID"].min() >= 1000
        assert m_ratings["TeamID"].max() <= 1999

    def test_w_team_ids_valid(self, w_ratings):
        """Women's TeamIDs should be in 3xxx range."""
        assert w_ratings["TeamID"].min() >= 3000
        assert w_ratings["TeamID"].max() <= 3999

    def test_no_nans_in_core_features(self, m_ratings):
        """Core features (adjoe, adjde, barthag) should have no NaN."""
        for col in ["adjoe", "adjde", "barthag"]:
            assert m_ratings[col].isna().sum() == 0, f"{col} has NaN"

    def test_adjoe_range(self, m_ratings):
        """Adjusted offensive efficiency should be in reasonable range."""
        assert m_ratings["adjoe"].min() > 60
        assert m_ratings["adjoe"].max() < 150

    def test_adjde_range(self, m_ratings):
        """Adjusted defensive efficiency should be in reasonable range."""
        assert m_ratings["adjde"].min() > 60
        assert m_ratings["adjde"].max() < 150

    def test_barthag_range(self, m_ratings):
        """Barthag (power rating) should be between 0 and 1."""
        assert m_ratings["barthag"].min() >= 0
        assert m_ratings["barthag"].max() <= 1

    def test_wab_range(self, m_ratings):
        """WAB should be in reasonable range."""
        assert m_ratings["wab"].min() > -30
        assert m_ratings["wab"].max() < 20

    def test_teams_per_season_m(self, m_ratings):
        """Each men's season should have 330-370 teams."""
        tps = m_ratings.groupby("Season").size()
        assert tps.min() >= 330, f"Min teams/season: {tps.min()}"
        assert tps.max() <= 370, f"Max teams/season: {tps.max()}"

    def test_teams_per_season_w(self, w_ratings):
        """Each women's season should have 350-370 teams."""
        tps = w_ratings.groupby("Season").size()
        assert tps.min() >= 350, f"Min teams/season: {tps.min()}"
        assert tps.max() <= 370, f"Max teams/season: {tps.max()}"

    def test_old_file_column_alignment(self):
        """Pre-2023 files (44 cols) should load with correct column alignment.

        Verifies that the column fix correctly maps Kansas 2008 data.
        """
        df = load_barttorvik_ratings("M")
        kansas_2008 = df[(df["Season"] == 2008) & (df["TeamID"] == 1242)]
        assert len(kansas_2008) == 1
        row = kansas_2008.iloc[0]
        # Kansas 2008: adjoe ~122, adjde ~86, barthag ~0.98
        assert 120 < row["adjoe"] < 125
        assert 84 < row["adjde"] < 90
        assert 0.97 < row["barthag"] < 0.99


class TestDifferentialFeatures:
    """Tests for Barttorvik differential features in the feature matrix."""

    @pytest.fixture(scope="class")
    def m_features(self):
        return pd.read_csv("artifacts/features_men.csv")

    @pytest.fixture(scope="class")
    def w_features(self):
        return pd.read_csv("artifacts/features_women.csv")

    def test_feature_count(self, m_features):
        """Feature matrix should have 47 _diff columns (38 old + 9 Barttorvik)."""
        diff_cols = [c for c in m_features.columns if c.endswith("_diff")]
        assert len(diff_cols) == 47

    def test_barttorvik_diff_cols_present(self, m_features):
        """All 9 Barttorvik _diff columns should exist."""
        expected = [f"{f}_diff" for f in BARTTORVIK_FEATURES]
        actual = [c for c in m_features.columns if c.endswith("_diff")]
        for col in expected:
            assert col in actual, f"{col} missing from feature matrix"

    def test_row_count_unchanged(self, m_features, w_features):
        """Row counts should match pre-Barttorvik values."""
        assert len(m_features) == 2585
        assert len(w_features) == 1717

    def test_nan_only_in_uncovered_seasons_m(self, m_features):
        """Men's Barttorvik NaN should only be in pre-2008 seasons."""
        bt_col = "barthag_diff"
        covered = m_features[m_features["Season"] >= 2008]
        uncovered = m_features[m_features["Season"] < 2008]
        assert covered[bt_col].isna().sum() == 0
        assert uncovered[bt_col].isna().all()

    def test_nan_only_in_uncovered_seasons_w(self, w_features):
        """Women's Barttorvik NaN should only be in pre-2021 seasons."""
        bt_col = "barthag_diff"
        covered = w_features[w_features["Season"] >= 2021]
        uncovered = w_features[w_features["Season"] < 2021]
        assert covered[bt_col].isna().sum() == 0
        assert uncovered[bt_col].isna().all()

    def test_barthag_diff_positive_correlation(self, m_features):
        """barthag_diff should positively correlate with target."""
        df = m_features.dropna(subset=["barthag_diff"])
        corr = df["barthag_diff"].corr(df["target"])
        assert corr > 0.3, f"barthag_diff correlation {corr} too low"

    def test_wab_diff_positive_correlation(self, m_features):
        """wab_diff should positively correlate with target."""
        df = m_features.dropna(subset=["wab_diff"])
        corr = df["wab_diff"].corr(df["target"])
        assert corr > 0.3, f"wab_diff correlation {corr} too low"

    def test_adjde_diff_negative_correlation(self, m_features):
        """adjde_diff should negatively correlate (higher = worse defense)."""
        df = m_features.dropna(subset=["adjde_diff"])
        corr = df["adjde_diff"].corr(df["target"])
        assert corr < -0.3, f"adjde_diff correlation {corr} should be negative"

    def test_existing_features_not_corrupted(self):
        """Old features should exactly match pre-Barttorvik backup."""
        old = pd.read_csv("artifacts/features_men_pre_barttorvik.csv")
        new = pd.read_csv("artifacts/features_men.csv")
        for col in ["win_pct_diff", "elo_diff", "seed_num_diff", "off_eff_diff"]:
            assert np.allclose(old[col].values, new[col].values, equal_nan=True), (
                f"{col} values changed after Barttorvik integration"
            )
