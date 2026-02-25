"""Massey Ordinals processing (Men only).

Provides pre-tournament rankings and team ranking differentials
for use as features in matchup prediction.
"""
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

from src import config
from src import data_loader as dl

# Cache the full Massey DataFrame once — it's 5.8M rows
_massey_cache: Optional[pd.DataFrame] = None


def _get_massey() -> pd.DataFrame:
    """Return the Massey ordinals DataFrame, loading and caching once."""
    global _massey_cache
    if _massey_cache is None:
        _massey_cache = dl.load_massey_ordinals()
    return _massey_cache


def get_available_systems(min_seasons: int = 10) -> list[str]:
    """Return Massey systems with at least min_seasons of data.

    Args:
        min_seasons: Minimum number of seasons a system must cover.

    Returns:
        List of SystemName strings, sorted by season coverage (descending).
    """
    massey = _get_massey()
    coverage = massey.groupby('SystemName')['Season'].nunique()
    systems = coverage[coverage >= min_seasons].sort_values(ascending=False)
    return systems.index.tolist()


def get_team_rankings(
    season: int,
    team_id: int,
    day: int = 133,
    systems: Optional[list[str]] = None,
) -> dict[str, float]:
    """Get a team's latest ranking before a given day for each system.

    Args:
        season: Season year (e.g. 2024).
        team_id: Team ID.
        day: Only use rankings published on or before this DayNum.
        systems: If provided, return only these systems. Otherwise all available.

    Returns:
        Dict {system_name: ordinal_rank} — NaN for missing system/team combos.
    """
    massey = _get_massey()

    # Filter to this season, team, and day
    mask = (
        (massey['Season'] == season)
        & (massey['TeamID'] == team_id)
        & (massey['RankingDayNum'] <= day)
    )
    team_df = massey[mask]

    if len(team_df) == 0:
        if systems:
            return {s: float('nan') for s in systems}
        return {}

    # For each system, take the latest ranking
    latest = (
        team_df.sort_values('RankingDayNum')
        .groupby('SystemName')['OrdinalRank']
        .last()
    )

    if systems is None:
        return latest.to_dict()

    return {s: float(latest.get(s, float('nan'))) for s in systems}


def get_ranking_differential(
    season: int,
    team_a_id: int,
    team_b_id: int,
    day: int = 133,
    systems: Optional[list[str]] = None,
) -> dict[str, float]:
    """Get ranking differentials between two teams for each Massey system.

    Differential = rank_a - rank_b.
    Lower rank number = better team (ordinal ranking).
    So negative diff means team_a is ranked better (lower number).

    Args:
        season: Season year.
        team_a_id: First team (conventionally the lower TeamID).
        team_b_id: Second team.
        day: Only use rankings on or before this day.
        systems: Systems to include. Defaults to TOP_MASSEY_SYSTEMS.

    Returns:
        Dict {system_name + '_diff': rank_a - rank_b}
    """
    if systems is None:
        systems = config.TOP_MASSEY_SYSTEMS

    ranks_a = get_team_rankings(season, team_a_id, day, systems)
    ranks_b = get_team_rankings(season, team_b_id, day, systems)

    return {
        f'{sys}_rank_diff': ranks_a[sys] - ranks_b[sys]
        for sys in systems
    }


def get_season_system_index(season: int, day: int = 133) -> pd.DataFrame:
    """Return a pivot table of all team rankings for a season.

    Useful for bulk feature extraction — avoids row-by-row filtering.

    Args:
        season: Season year.
        day: Only use rankings on or before this DayNum.

    Returns:
        DataFrame with TeamID as index, SystemName as columns, OrdinalRank as values.
        Each cell is the latest ranking for that team/system pair before `day`.
    """
    massey = _get_massey()
    mask = (massey['Season'] == season) & (massey['RankingDayNum'] <= day)
    season_df = massey[mask].copy()

    if len(season_df) == 0:
        return pd.DataFrame()

    # Keep latest ranking per team+system
    latest = (
        season_df.sort_values('RankingDayNum')
        .groupby(['TeamID', 'SystemName'])['OrdinalRank']
        .last()
        .reset_index()
    )

    pivot = latest.pivot(index='TeamID', columns='SystemName', values='OrdinalRank')
    pivot.columns.name = None
    return pivot


def clear_cache() -> None:
    """Clear the Massey DataFrame cache."""
    global _massey_cache
    _massey_cache = None
