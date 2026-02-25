"""Comprehensive data quality test suite. Every check must pass before Phase 2."""
import re

import pytest
import pandas as pd

from src import config
from src import data_loader as dl
from src.data_validator import (
    check_referential_integrity,
    check_temporal_consistency,
    check_score_consistency,
    check_seed_consistency,
    check_completeness,
    check_cross_gender,
    generate_report,
)


@pytest.fixture(autouse=True)
def clear_cache():
    dl.clear_cache()
    yield
    dl.clear_cache()


# ---------------------------------------------------------------------------
# Referential Integrity
# ---------------------------------------------------------------------------

class TestReferentialIntegrity:
    def test_m_regular_season_team_ids_exist(self):
        team_ids = set(dl.load_teams('M')['TeamID'])
        reg = dl.load_regular_season('M')
        assert set(reg['WTeamID']).issubset(team_ids)
        assert set(reg['LTeamID']).issubset(team_ids)

    def test_w_regular_season_team_ids_exist(self):
        team_ids = set(dl.load_teams('W')['TeamID'])
        reg = dl.load_regular_season('W')
        assert set(reg['WTeamID']).issubset(team_ids)
        assert set(reg['LTeamID']).issubset(team_ids)

    def test_m_tourney_team_ids_exist(self):
        team_ids = set(dl.load_teams('M')['TeamID'])
        tourney = dl.load_tourney_results('M')
        assert set(tourney['WTeamID']).issubset(team_ids)
        assert set(tourney['LTeamID']).issubset(team_ids)

    def test_w_tourney_team_ids_exist(self):
        team_ids = set(dl.load_teams('W')['TeamID'])
        tourney = dl.load_tourney_results('W')
        assert set(tourney['WTeamID']).issubset(team_ids)
        assert set(tourney['LTeamID']).issubset(team_ids)

    def test_m_seed_team_ids_exist(self):
        team_ids = set(dl.load_teams('M')['TeamID'])
        seeds = dl.load_tourney_seeds('M')
        assert set(seeds['TeamID']).issubset(team_ids)

    def test_w_seed_team_ids_exist(self):
        team_ids = set(dl.load_teams('W')['TeamID'])
        seeds = dl.load_tourney_seeds('W')
        assert set(seeds['TeamID']).issubset(team_ids)

    def test_m_conference_team_ids_exist(self):
        team_ids = set(dl.load_teams('M')['TeamID'])
        conf = dl.load_conferences('M')
        assert set(conf['TeamID']).issubset(team_ids)

    def test_w_conference_team_ids_exist(self):
        team_ids = set(dl.load_teams('W')['TeamID'])
        conf = dl.load_conferences('W')
        assert set(conf['TeamID']).issubset(team_ids)

    def test_conference_abbrevs_exist(self):
        valid_abbrevs = set(dl.load_conference_list()['ConfAbbrev'])
        for gender in ('M', 'W'):
            team_conf = dl.load_conferences(gender)
            assert set(team_conf['ConfAbbrev']).issubset(valid_abbrevs), \
                f"{gender} has unknown conference abbreviations"

    def test_m_game_city_ids_exist(self):
        city_ids = set(dl.load_cities()['CityID'])
        gc = dl.load_game_cities('M')
        assert set(gc['CityID']).issubset(city_ids)

    def test_w_game_city_ids_exist(self):
        city_ids = set(dl.load_cities()['CityID'])
        gc = dl.load_game_cities('W')
        assert set(gc['CityID']).issubset(city_ids)


# ---------------------------------------------------------------------------
# Temporal Consistency
# ---------------------------------------------------------------------------

class TestTemporalConsistency:
    def test_day_num_non_negative(self):
        for gender in ('M', 'W'):
            reg = dl.load_regular_season(gender)
            assert (reg['DayNum'] >= 0).all(), f"{gender} has negative DayNum"

    def test_day_num_not_too_large(self):
        for gender in ('M', 'W'):
            reg = dl.load_regular_season(gender)
            assert (reg['DayNum'] <= 200).all(), f"{gender} DayNum exceeds 200"

    def test_regular_season_before_tourney(self):
        for gender in ('M', 'W'):
            reg = dl.load_regular_season(gender)
            assert (reg['DayNum'] < config.TOURNEY_START_DAY).all(), \
                f"{gender} regular season has games on/after tourney start day"

    def test_tourney_games_after_day_132(self):
        for gender in ('M', 'W'):
            tourney = dl.load_tourney_results(gender)
            assert (tourney['DayNum'] >= 132).all(), \
                f"{gender} tourney games with DayNum < 132"

    def test_massey_day_num_range(self):
        massey = dl.load_massey_ordinals()
        assert (massey['RankingDayNum'] >= 0).all()
        assert (massey['RankingDayNum'] <= 200).all()


# ---------------------------------------------------------------------------
# Score & Stats Consistency
# ---------------------------------------------------------------------------

class TestScoreConsistency:
    def test_winner_beats_loser(self):
        for gender in ('M', 'W'):
            df = dl.load_regular_season(gender)
            assert (df['WScore'] > df['LScore']).all(), \
                f"{gender} compact: winner doesn't always outscore loser"

    def test_tourney_winner_beats_loser(self):
        for gender in ('M', 'W'):
            df = dl.load_tourney_results(gender)
            assert (df['WScore'] > df['LScore']).all(), \
                f"{gender} tourney: winner doesn't always outscore loser"

    def test_scores_in_realistic_range(self):
        for gender in ('M', 'W'):
            df = dl.load_regular_season(gender)
            assert (df['WScore'] >= 30).all()
            assert (df['WScore'] <= 200).all()
            assert (df['LScore'] >= 0).all()
            assert (df['LScore'] <= 180).all()

    def test_fgm_lte_fga(self):
        for gender in ('M', 'W'):
            df = dl.load_regular_season(gender, detailed=True)
            for prefix in ('W', 'L'):
                assert (df[f'{prefix}FGM'] <= df[f'{prefix}FGA']).all(), \
                    f"{gender} detailed {prefix}: FGM > FGA"

    def test_fgm3_lte_fga3(self):
        for gender in ('M', 'W'):
            df = dl.load_regular_season(gender, detailed=True)
            for prefix in ('W', 'L'):
                assert (df[f'{prefix}FGM3'] <= df[f'{prefix}FGA3']).all(), \
                    f"{gender} detailed {prefix}: FGM3 > FGA3"

    def test_ftm_lte_fta(self):
        for gender in ('M', 'W'):
            df = dl.load_regular_season(gender, detailed=True)
            for prefix in ('W', 'L'):
                assert (df[f'{prefix}FTM'] <= df[f'{prefix}FTA']).all(), \
                    f"{gender} detailed {prefix}: FTM > FTA"

    def test_fgm3_lte_fgm(self):
        for gender in ('M', 'W'):
            df = dl.load_regular_season(gender, detailed=True)
            for prefix in ('W', 'L'):
                assert (df[f'{prefix}FGM3'] <= df[f'{prefix}FGM']).all(), \
                    f"{gender} detailed {prefix}: FGM3 > FGM"

    def test_rebounds_positive(self):
        for gender in ('M', 'W'):
            df = dl.load_regular_season(gender, detailed=True)
            for prefix in ('W', 'L'):
                total_reb = df[f'{prefix}OR'] + df[f'{prefix}DR']
                assert (total_reb > 0).all(), \
                    f"{gender} detailed {prefix}: games with zero total rebounds"

    def test_score_reconstruction(self):
        """Score must exactly equal 2*(FGM-FGM3) + 3*FGM3 + FTM."""
        for gender in ('M', 'W'):
            df = dl.load_regular_season(gender, detailed=True)
            for prefix, score_col in [('W', 'WScore'), ('L', 'LScore')]:
                fgm = df[f'{prefix}FGM']
                fgm3 = df[f'{prefix}FGM3']
                ftm = df[f'{prefix}FTM']
                reconstructed = 2 * (fgm - fgm3) + 3 * fgm3 + ftm
                mismatch = (reconstructed != df[score_col]).sum()
                assert mismatch == 0, \
                    f"{gender} {prefix}: {mismatch} score reconstruction mismatches"


# ---------------------------------------------------------------------------
# Seed Consistency
# ---------------------------------------------------------------------------

class TestSeedConsistency:
    SEED_PATTERN = re.compile(r'^[WXYZ]\d{2}[ab]?$')

    def test_seed_format(self):
        for gender in ('M', 'W'):
            seeds_df = dl.load_tourney_seeds(gender)
            invalid = seeds_df[~seeds_df['Seed'].str.match(self.SEED_PATTERN)]
            assert len(invalid) == 0, \
                f"{gender} seeds: {len(invalid)} invalid format seeds: {invalid['Seed'].head().tolist()}"

    def test_seed_numeric_range(self):
        for gender in ('M', 'W'):
            seeds_df = dl.load_tourney_seeds(gender)
            assert seeds_df['SeedNum'].between(1, 16).all(), \
                f"{gender} seeds outside 1-16"

    def test_every_tourney_team_has_seed(self):
        for gender in ('M', 'W'):
            seeds_df = dl.load_tourney_seeds(gender)
            tourney_df = dl.load_tourney_results(gender)

            seasons = tourney_df['Season'].unique()
            for season in seasons:
                seeded = set(seeds_df[seeds_df['Season'] == season]['TeamID'])
                game_teams = (
                    set(tourney_df[tourney_df['Season'] == season]['WTeamID'])
                    | set(tourney_df[tourney_df['Season'] == season]['LTeamID'])
                )
                unseeded = game_teams - seeded
                assert len(unseeded) == 0, \
                    f"{gender} {season}: unseeded tourney teams {unseeded}"


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------

class TestCompleteness:
    def test_m_no_missing_regular_season(self):
        reg = dl.load_regular_season('M')
        seasons = set(reg['Season'].unique())
        expected = set(range(config.FIRST_COMPACT_SEASON_M, config.CURRENT_SEASON))
        missing = expected - seasons
        assert not missing, f"M regular season missing: {sorted(missing)}"

    def test_w_no_missing_regular_season(self):
        reg = dl.load_regular_season('W')
        seasons = set(reg['Season'].unique())
        expected = set(range(config.FIRST_COMPACT_SEASON_W, config.CURRENT_SEASON))
        missing = expected - seasons
        assert not missing, f"W regular season missing: {sorted(missing)}"

    def test_tourney_seasons_have_seeds_m(self):
        tourney = dl.load_tourney_results('M')
        seeds = dl.load_tourney_seeds('M')
        missing = set(tourney['Season'].unique()) - set(seeds['Season'].unique())
        assert not missing, f"M tourney seasons missing seeds: {missing}"

    def test_tourney_seasons_have_seeds_w(self):
        tourney = dl.load_tourney_results('W')
        seeds = dl.load_tourney_seeds('W')
        missing = set(tourney['Season'].unique()) - set(seeds['Season'].unique())
        assert not missing, f"W tourney seasons missing seeds: {missing}"

    def test_m_detailed_starts_at_2003(self):
        det = dl.load_regular_season('M', detailed=True)
        assert det['Season'].min() <= config.FIRST_DETAILED_SEASON_M

    def test_w_detailed_starts_at_2010(self):
        det = dl.load_regular_season('W', detailed=True)
        assert det['Season'].min() <= config.FIRST_DETAILED_SEASON_W

    def test_sample_submission_ids_parseable(self):
        sub = dl.load_sample_submission(stage=1)
        all_m_ids = set(dl.load_teams('M')['TeamID'])
        all_w_ids = set(dl.load_teams('W')['TeamID'])
        all_ids = all_m_ids | all_w_ids

        for row_id in sub['ID']:
            parts = row_id.split('_')
            assert len(parts) == 3, f"Bad ID: {row_id}"
            team_a, team_b = int(parts[1]), int(parts[2])
            assert team_a in all_ids, f"Unknown team {team_a} in submission"
            assert team_b in all_ids, f"Unknown team {team_b} in submission"


# ---------------------------------------------------------------------------
# Cross-Gender Consistency
# ---------------------------------------------------------------------------

class TestCrossGender:
    def test_m_w_team_ids_no_overlap(self):
        m_ids = set(dl.load_teams('M')['TeamID'])
        w_ids = set(dl.load_teams('W')['TeamID'])
        assert not (m_ids & w_ids), f"Overlapping M/W TeamIDs: {m_ids & w_ids}"

    def test_submission_has_m_pairs(self):
        sub = dl.load_sample_submission(stage=1)
        team_a_ids = sub['ID'].apply(lambda x: int(x.split('_')[1]))
        assert team_a_ids.between(1000, 1999).any(), "No M pairs in submission"

    def test_submission_has_w_pairs(self):
        sub = dl.load_sample_submission(stage=1)
        team_a_ids = sub['ID'].apply(lambda x: int(x.split('_')[1]))
        assert team_a_ids.between(3000, 3999).any(), "No W pairs in submission"

    def test_m_w_compact_same_columns(self):
        reg_m = dl.load_regular_season('M')
        reg_w = dl.load_regular_season('W')
        assert list(reg_m.columns) == list(reg_w.columns)

    def test_m_w_detailed_same_columns(self):
        det_m = dl.load_regular_season('M', detailed=True)
        det_w = dl.load_regular_season('W', detailed=True)
        assert list(det_m.columns) == list(det_w.columns)

    def test_m_w_tourney_same_columns(self):
        t_m = dl.load_tourney_results('M')
        t_w = dl.load_tourney_results('W')
        assert list(t_m.columns) == list(t_w.columns)


# ---------------------------------------------------------------------------
# Full Report Generation
# ---------------------------------------------------------------------------

class TestReportGeneration:
    def test_report_generates(self, tmp_path):
        report_path = tmp_path / "report.txt"
        report = generate_report(output_path=report_path)
        assert isinstance(report, str)
        assert len(report) > 100
        assert report_path.exists()

    def test_report_contains_section_headers(self):
        report = generate_report()
        assert 'Referential Integrity' in report or 'referential' in report.lower()
