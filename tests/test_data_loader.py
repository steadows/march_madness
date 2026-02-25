"""Tests for src/data_loader.py — every load function returns valid, typed DataFrames."""
import pytest
import pandas as pd

from src.data_loader import (
    load_teams,
    load_regular_season,
    load_tourney_results,
    load_tourney_seeds,
    load_tourney_slots,
    load_massey_ordinals,
    load_conferences,
    load_conference_list,
    load_coaches,
    load_cities,
    load_game_cities,
    load_seasons,
    load_conference_tourney,
    load_secondary_tourney,
    load_sample_submission,
    parse_seed,
    clear_cache,
)


@pytest.fixture(autouse=True)
def clear():
    """Clear cache before each test for isolation."""
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

class TestLoadTeams:
    def test_men_columns(self):
        df = load_teams('M')
        assert list(df.columns) == ['TeamID', 'TeamName', 'FirstD1Season', 'LastD1Season']

    def test_women_columns(self):
        df = load_teams('W')
        assert 'TeamID' in df.columns
        assert 'TeamName' in df.columns

    def test_men_row_count(self):
        df = load_teams('M')
        assert 300 < len(df) < 1000

    def test_women_row_count(self):
        df = load_teams('W')
        assert 200 < len(df) < 1000

    def test_no_null_team_id(self):
        for gender in ('M', 'W'):
            df = load_teams(gender)
            assert df['TeamID'].isna().sum() == 0

    def test_men_id_range(self):
        df = load_teams('M')
        assert df['TeamID'].between(1100, 1999).all()

    def test_women_id_range(self):
        df = load_teams('W')
        assert df['TeamID'].between(3000, 3999).all()

    def test_caching(self):
        df1 = load_teams('M')
        df2 = load_teams('M')
        assert df1 is df2


# ---------------------------------------------------------------------------
# Regular Season
# ---------------------------------------------------------------------------

class TestLoadRegularSeason:
    def test_compact_columns(self):
        df = load_regular_season('M', detailed=False)
        expected = ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT']
        assert list(df.columns) == expected

    def test_detailed_has_more_columns(self):
        df = load_regular_season('M', detailed=True)
        assert len(df.columns) > 10

    def test_men_row_count(self):
        df = load_regular_season('M')
        assert len(df) > 50_000

    def test_women_row_count(self):
        df = load_regular_season('W')
        assert len(df) > 30_000

    def test_no_null_season(self):
        df = load_regular_season('M')
        assert df['Season'].isna().sum() == 0

    def test_no_null_team_ids(self):
        df = load_regular_season('M')
        assert df['WTeamID'].isna().sum() == 0
        assert df['LTeamID'].isna().sum() == 0

    def test_men_season_range(self):
        df = load_regular_season('M')
        assert df['Season'].min() == 1985
        assert df['Season'].max() >= 2024


# ---------------------------------------------------------------------------
# Tournament Results
# ---------------------------------------------------------------------------

class TestLoadTourneyResults:
    def test_men_compact_columns(self):
        df = load_tourney_results('M', detailed=False)
        expected = ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT']
        assert list(df.columns) == expected

    def test_men_row_count(self):
        df = load_tourney_results('M')
        assert len(df) > 2_000

    def test_women_row_count(self):
        df = load_tourney_results('W')
        assert len(df) > 1_000

    def test_no_null_team_ids(self):
        df = load_tourney_results('M')
        assert df['WTeamID'].isna().sum() == 0
        assert df['LTeamID'].isna().sum() == 0


# ---------------------------------------------------------------------------
# Tournament Seeds
# ---------------------------------------------------------------------------

class TestLoadTourneySeeds:
    def test_columns(self):
        df = load_tourney_seeds('M')
        assert 'Season' in df.columns
        assert 'Seed' in df.columns
        assert 'TeamID' in df.columns
        assert 'SeedNum' in df.columns

    def test_seed_num_range(self):
        df = load_tourney_seeds('M')
        assert df['SeedNum'].between(1, 16).all()

    def test_no_null_season_teamid(self):
        df = load_tourney_seeds('M')
        assert df['Season'].isna().sum() == 0
        assert df['TeamID'].isna().sum() == 0

    def test_women_seeds(self):
        df = load_tourney_seeds('W')
        assert len(df) > 500
        assert df['SeedNum'].between(1, 16).all()


# ---------------------------------------------------------------------------
# Tournament Slots
# ---------------------------------------------------------------------------

class TestLoadTourneySlots:
    def test_columns(self):
        df = load_tourney_slots('M')
        assert 'Season' in df.columns
        assert 'Slot' in df.columns
        assert 'StrongSeed' in df.columns
        assert 'WeakSeed' in df.columns

    def test_row_count(self):
        df = load_tourney_slots('M')
        assert len(df) > 1_000


# ---------------------------------------------------------------------------
# Massey Ordinals
# ---------------------------------------------------------------------------

class TestLoadMasseyOrdinals:
    def test_columns(self):
        df = load_massey_ordinals()
        expected = ['Season', 'RankingDayNum', 'SystemName', 'TeamID', 'OrdinalRank']
        assert list(df.columns) == expected

    def test_large_row_count(self):
        df = load_massey_ordinals()
        assert len(df) > 1_000_000

    def test_no_null_key_cols(self):
        df = load_massey_ordinals()
        assert df['Season'].isna().sum() == 0
        assert df['TeamID'].isna().sum() == 0

    def test_top_systems_present(self):
        df = load_massey_ordinals()
        systems = df['SystemName'].unique().tolist()
        for sys in ['POM', 'SAG']:
            assert sys in systems, f"Expected system {sys} not found"

    def test_ordinal_rank_positive(self):
        # Sample check (full scan too slow)
        df = load_massey_ordinals().head(10_000)
        assert (df['OrdinalRank'] > 0).all()


# ---------------------------------------------------------------------------
# Conferences
# ---------------------------------------------------------------------------

class TestLoadConferences:
    def test_columns(self):
        df = load_conferences('M')
        assert list(df.columns) == ['Season', 'TeamID', 'ConfAbbrev']

    def test_women_columns(self):
        df = load_conferences('W')
        assert list(df.columns) == ['Season', 'TeamID', 'ConfAbbrev']

    def test_row_count(self):
        df = load_conferences('M')
        assert len(df) > 5_000

    def test_conference_list(self):
        df = load_conference_list()
        assert 'ConfAbbrev' in df.columns
        assert len(df) > 20


# ---------------------------------------------------------------------------
# Coaches
# ---------------------------------------------------------------------------

class TestLoadCoaches:
    def test_columns(self):
        df = load_coaches()
        assert list(df.columns) == ['Season', 'TeamID', 'FirstDayNum', 'LastDayNum', 'CoachName']

    def test_row_count(self):
        df = load_coaches()
        assert len(df) > 1_000

    def test_no_null_season(self):
        df = load_coaches()
        assert df['Season'].isna().sum() == 0


# ---------------------------------------------------------------------------
# Cities
# ---------------------------------------------------------------------------

class TestLoadCities:
    def test_columns(self):
        df = load_cities()
        assert list(df.columns) == ['CityID', 'City', 'State']

    def test_row_count(self):
        df = load_cities()
        assert len(df) > 50

    def test_game_cities_columns(self):
        df = load_game_cities('M')
        assert 'Season' in df.columns
        assert 'CityID' in df.columns

    def test_game_cities_row_count(self):
        df = load_game_cities('M')
        assert len(df) > 1_000


# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------

class TestLoadSeasons:
    def test_columns(self):
        df = load_seasons('M')
        assert 'Season' in df.columns
        assert 'DayZero' in df.columns

    def test_row_count(self):
        df = load_seasons('M')
        assert len(df) >= 30

    def test_women_seasons(self):
        df = load_seasons('W')
        assert len(df) >= 20


# ---------------------------------------------------------------------------
# Conference Tourney
# ---------------------------------------------------------------------------

class TestLoadConferenceTourney:
    def test_columns(self):
        df = load_conference_tourney('M')
        assert 'Season' in df.columns
        assert 'WTeamID' in df.columns
        assert 'LTeamID' in df.columns

    def test_row_count(self):
        df = load_conference_tourney('M')
        assert len(df) > 500


# ---------------------------------------------------------------------------
# Secondary Tourney
# ---------------------------------------------------------------------------

class TestLoadSecondaryTourney:
    def test_columns(self):
        df = load_secondary_tourney('M')
        assert 'Season' in df.columns
        assert 'WTeamID' in df.columns
        assert 'LTeamID' in df.columns

    def test_row_count(self):
        df = load_secondary_tourney('M')
        assert len(df) > 100


# ---------------------------------------------------------------------------
# Sample Submission
# ---------------------------------------------------------------------------

class TestLoadSampleSubmission:
    def test_columns(self):
        df = load_sample_submission(stage=1)
        assert list(df.columns) == ['ID', 'Pred']

    def test_stage1_row_count(self):
        df = load_sample_submission(stage=1)
        assert len(df) > 100_000

    def test_id_format(self):
        df = load_sample_submission(stage=1)
        sample = df['ID'].iloc[0]
        parts = sample.split('_')
        assert len(parts) == 3
        season, team_a, team_b = int(parts[0]), int(parts[1]), int(parts[2])
        assert 2022 <= season <= 2026
        assert team_a < team_b  # lower ID always first


# ---------------------------------------------------------------------------
# Seed Parsing
# ---------------------------------------------------------------------------

class TestParseSeed:
    @pytest.mark.parametrize("seed_str,expected", [
        ('W01', 1),
        ('X16', 16),
        ('Y11', 11),
        ('Z04', 4),
        ('W16a', 16),
        ('X11b', 11),
    ])
    def test_parse(self, seed_str, expected):
        assert parse_seed(seed_str) == expected
