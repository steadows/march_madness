"""Data quality validation module. Standalone assertions and reporting."""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import config
from src import data_loader as dl


class ValidationError(Exception):
    """Raised when a critical data quality check fails."""


def check_referential_integrity() -> list[str]:
    """Check that all foreign key references are valid.

    Returns:
        List of warning strings (empty = all clear).
    """
    warnings: list[str] = []

    for gender in ('M', 'W'):
        team_ids = set(dl.load_teams(gender)['TeamID'])

        # Game results reference valid teams
        for loader, label in [
            (lambda g: dl.load_regular_season(g), 'regular season'),
            (lambda g: dl.load_tourney_results(g), 'tourney results'),
        ]:
            df = loader(gender)
            bad_w = set(df['WTeamID']) - team_ids
            bad_l = set(df['LTeamID']) - team_ids
            if bad_w:
                warnings.append(f"{gender} {label}: {len(bad_w)} unknown WTeamIDs")
            if bad_l:
                warnings.append(f"{gender} {label}: {len(bad_l)} unknown LTeamIDs")

        # Seeds reference valid teams
        seeds_df = dl.load_tourney_seeds(gender)
        bad_seed = set(seeds_df['TeamID']) - team_ids
        if bad_seed:
            warnings.append(f"{gender} tourney seeds: {len(bad_seed)} unknown TeamIDs")

        # Conferences reference valid teams
        conf_df = dl.load_conferences(gender)
        bad_conf = set(conf_df['TeamID']) - team_ids
        if bad_conf:
            warnings.append(f"{gender} conferences: {len(bad_conf)} unknown TeamIDs")

    # Conference abbreviations exist in Conferences.csv
    conf_list = set(dl.load_conference_list()['ConfAbbrev'])
    for gender in ('M', 'W'):
        team_conf = dl.load_conferences(gender)
        bad_abbr = set(team_conf['ConfAbbrev']) - conf_list
        if bad_abbr:
            warnings.append(f"{gender} team conferences: unknown abbrevs {bad_abbr}")

    # CityIDs in game cities exist in Cities
    city_ids = set(dl.load_cities()['CityID'])
    for gender in ('M', 'W'):
        gc = dl.load_game_cities(gender)
        bad_cities = set(gc['CityID']) - city_ids
        if bad_cities:
            warnings.append(f"{gender} game cities: {len(bad_cities)} unknown CityIDs")

    return warnings


def check_temporal_consistency() -> list[str]:
    """Check that game days and season ordering are sane.

    Returns:
        List of warning strings.
    """
    warnings: list[str] = []

    for gender in ('M', 'W'):
        reg = dl.load_regular_season(gender)
        # DayNum bounds
        if (reg['DayNum'] < 0).any():
            warnings.append(f"{gender} regular season: negative DayNum found")
        if (reg['DayNum'] > 200).any():
            n = (reg['DayNum'] > 200).sum()
            warnings.append(f"{gender} regular season: {n} games with DayNum > 200")

        # Regular season games should be before tourney
        if (reg['DayNum'] >= config.TOURNEY_START_DAY).any():
            n = (reg['DayNum'] >= config.TOURNEY_START_DAY).sum()
            warnings.append(
                f"{gender} regular season: {n} games with DayNum >= {config.TOURNEY_START_DAY}"
            )

        tourney = dl.load_tourney_results(gender)
        # Tournament games: DayNum roughly >= 132
        if (tourney['DayNum'] < 132).any():
            n = (tourney['DayNum'] < 132).sum()
            warnings.append(f"{gender} tourney: {n} games with DayNum < 132")

    # Massey ordinals day range
    massey = dl.load_massey_ordinals()
    if (massey['RankingDayNum'] < 0).any():
        warnings.append("Massey: negative RankingDayNum found")
    if (massey['RankingDayNum'] > 200).any():
        n = (massey['RankingDayNum'] > 200).sum()
        warnings.append(f"Massey: {n} rows with RankingDayNum > 200")

    return warnings


def check_score_consistency() -> list[str]:
    """Verify score and box score data integrity.

    Returns:
        List of warning strings.
    """
    warnings: list[str] = []

    for gender in ('M', 'W'):
        compact = dl.load_regular_season(gender, detailed=False)

        # Winner must outscore loser
        if not (compact['WScore'] > compact['LScore']).all():
            n = (compact['WScore'] <= compact['LScore']).sum()
            warnings.append(f"{gender} compact: {n} games where WScore <= LScore")

        # Scores in realistic range
        if (compact['WScore'] < 30).any():
            n = (compact['WScore'] < 30).sum()
            warnings.append(f"{gender} compact: {n} games with WScore < 30")
        if (compact['WScore'] > 200).any():
            n = (compact['WScore'] > 200).sum()
            warnings.append(f"{gender} compact: {n} games with WScore > 200")

        # Detailed box score checks
        first_detailed = (
            config.FIRST_DETAILED_SEASON_M if gender == 'M'
            else config.FIRST_DETAILED_SEASON_W
        )
        detailed = dl.load_regular_season(gender, detailed=True)
        detailed = detailed[detailed['Season'] >= first_detailed]

        for prefix in ('W', 'L'):
            fgm = detailed[f'{prefix}FGM']
            fga = detailed[f'{prefix}FGA']
            fgm3 = detailed[f'{prefix}FGM3']
            fga3 = detailed[f'{prefix}FGA3']
            ftm = detailed[f'{prefix}FTM']
            fta = detailed[f'{prefix}FTA']
            oreb = detailed[f'{prefix}OR']
            dreb = detailed[f'{prefix}DR']

            if (fgm > fga).any():
                n = (fgm > fga).sum()
                warnings.append(f"{gender} detailed {prefix}: {n} rows FGM > FGA")
            if (fgm3 > fga3).any():
                n = (fgm3 > fga3).sum()
                warnings.append(f"{gender} detailed {prefix}: {n} rows FGM3 > FGA3")
            if (ftm > fta).any():
                n = (ftm > fta).sum()
                warnings.append(f"{gender} detailed {prefix}: {n} rows FTM > FTA")
            if (fgm3 > fgm).any():
                n = (fgm3 > fgm).sum()
                warnings.append(f"{gender} detailed {prefix}: {n} rows FGM3 > FGM")
            if ((oreb + dreb) <= 0).any():
                n = ((oreb + dreb) <= 0).sum()
                warnings.append(f"{gender} detailed {prefix}: {n} rows with zero rebounds")

        # Score reconstruction: Score ≈ 2*(FGM-FGM3) + 3*FGM3 + FTM
        for prefix, score_col in [('W', 'WScore'), ('L', 'LScore')]:
            fgm = detailed[f'{prefix}FGM']
            fgm3 = detailed[f'{prefix}FGM3']
            ftm = detailed[f'{prefix}FTM']
            reconstructed = 2 * (fgm - fgm3) + 3 * fgm3 + ftm
            mismatch = (reconstructed != detailed[score_col]).sum()
            if mismatch > 0:
                warnings.append(
                    f"{gender} detailed {prefix}: {mismatch} rows score reconstruction mismatch"
                )

    return warnings


def check_seed_consistency() -> list[str]:
    """Validate tournament seed data.

    Returns:
        List of warning strings.
    """
    warnings: list[str] = []

    import re

    for gender in ('M', 'W'):
        seeds_df = dl.load_tourney_seeds(gender)
        tourney_df = dl.load_tourney_results(gender)

        # Seed format validation
        valid_pattern = re.compile(r'^[WXYZ]\d{2}[ab]?$')
        invalid = seeds_df[~seeds_df['Seed'].str.match(valid_pattern)]
        if len(invalid) > 0:
            warnings.append(f"{gender} seeds: {len(invalid)} seeds with invalid format")

        # Numeric seed range
        if not seeds_df['SeedNum'].between(1, 16).all():
            warnings.append(f"{gender} seeds: seeds outside 1-16 range")

        # Every tourney team has a seed
        for season in tourney_df['Season'].unique():
            seeded = set(seeds_df[seeds_df['Season'] == season]['TeamID'])
            game_teams = set(
                tourney_df[tourney_df['Season'] == season]['WTeamID']
            ) | set(tourney_df[tourney_df['Season'] == season]['LTeamID'])
            unseeded = game_teams - seeded
            if unseeded:
                warnings.append(
                    f"{gender} {season}: {len(unseeded)} tourney teams with no seed"
                )

    return warnings


def check_completeness() -> list[str]:
    """Check that expected seasons and data are present.

    Returns:
        List of warning strings.
    """
    warnings: list[str] = []

    # Men: compact results from 1985
    reg_m = dl.load_regular_season('M')
    seasons_m = set(reg_m['Season'].unique())
    expected_m = set(range(config.FIRST_COMPACT_SEASON_M, config.CURRENT_SEASON))
    missing_m = expected_m - seasons_m
    if missing_m:
        warnings.append(f"M regular season: missing seasons {sorted(missing_m)}")

    # Women: compact results from 1998
    reg_w = dl.load_regular_season('W')
    seasons_w = set(reg_w['Season'].unique())
    expected_w = set(range(config.FIRST_COMPACT_SEASON_W, config.CURRENT_SEASON))
    missing_w = expected_w - seasons_w
    if missing_w:
        warnings.append(f"W regular season: missing seasons {sorted(missing_w)}")

    # Every tourney season has seeds
    for gender in ('M', 'W'):
        tourney = dl.load_tourney_results(gender)
        seeds = dl.load_tourney_seeds(gender)
        tourney_seasons = set(tourney['Season'].unique())
        seed_seasons = set(seeds['Season'].unique())
        missing_seeds = tourney_seasons - seed_seasons
        if missing_seeds:
            warnings.append(f"{gender} tourney seasons missing seeds: {missing_seeds}")

    # Detailed results available for 2003+ (M) and 2010+ (W)
    for gender, first_season in [('M', config.FIRST_DETAILED_SEASON_M),
                                   ('W', config.FIRST_DETAILED_SEASON_W)]:
        det = dl.load_regular_season(gender, detailed=True)
        min_det_season = det['Season'].min()
        if min_det_season > first_season:
            warnings.append(
                f"{gender} detailed results: earliest season {min_det_season} > expected {first_season}"
            )

    # Sample submission IDs parseable
    sub = dl.load_sample_submission(stage=1)
    all_team_ids_m = set(dl.load_teams('M')['TeamID'])
    all_team_ids_w = set(dl.load_teams('W')['TeamID'])
    all_team_ids = all_team_ids_m | all_team_ids_w
    for row_id in sub['ID'].head(100):
        parts = row_id.split('_')
        assert len(parts) == 3, f"Bad ID format: {row_id}"
        team_a, team_b = int(parts[1]), int(parts[2])
        if team_a not in all_team_ids:
            warnings.append(f"Sample submission: unknown TeamID {team_a}")
        if team_b not in all_team_ids:
            warnings.append(f"Sample submission: unknown TeamID {team_b}")

    return warnings


def check_cross_gender() -> list[str]:
    """Check cross-gender consistency constraints.

    Returns:
        List of warning strings.
    """
    warnings: list[str] = []

    m_ids = set(dl.load_teams('M')['TeamID'])
    w_ids = set(dl.load_teams('W')['TeamID'])

    overlap = m_ids & w_ids
    if overlap:
        warnings.append(f"M and W team IDs overlap: {overlap}")

    # Sample submission has both M and W pairs
    sub = dl.load_sample_submission(stage=1)
    def get_team_a(row_id: str) -> int:
        return int(row_id.split('_')[1])

    team_a_ids = sub['ID'].apply(get_team_a)
    has_m = team_a_ids.between(1000, 1999).any()
    has_w = team_a_ids.between(3000, 3999).any()
    if not has_m:
        warnings.append("Sample submission: no M (1xxx) team pairs found")
    if not has_w:
        warnings.append("Sample submission: no W (3xxx) team pairs found")

    # M and W should have same column names for equivalent files
    reg_m = dl.load_regular_season('M')
    reg_w = dl.load_regular_season('W')
    if list(reg_m.columns) != list(reg_w.columns):
        warnings.append(
            f"M/W regular season compact columns differ: {reg_m.columns.tolist()} vs {reg_w.columns.tolist()}"
        )

    return warnings


def run_all_checks() -> dict[str, list[str]]:
    """Run all data quality checks and return results.

    Returns:
        Dict mapping check name to list of warnings (empty = passed).
    """
    return {
        'referential_integrity': check_referential_integrity(),
        'temporal_consistency': check_temporal_consistency(),
        'score_consistency': check_score_consistency(),
        'seed_consistency': check_seed_consistency(),
        'completeness': check_completeness(),
        'cross_gender': check_cross_gender(),
    }


def generate_report(output_path: Optional[Path] = None) -> str:
    """Run all checks and produce a human-readable report.

    Args:
        output_path: If provided, write report to this file.

    Returns:
        Report as string.
    """
    results = run_all_checks()
    lines = ["Data Quality Report", "=" * 60]

    total_warnings = 0
    for check_name, warnings in results.items():
        status = "PASS" if not warnings else f"WARN ({len(warnings)} issues)"
        lines.append(f"\n[{status}] {check_name.replace('_', ' ').title()}")
        for w in warnings:
            lines.append(f"  - {w}")
            total_warnings += 1

    lines.append(f"\n{'=' * 60}")
    lines.append(f"Total warnings: {total_warnings}")
    if total_warnings == 0:
        lines.append("All checks passed.")

    report = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)

    return report
