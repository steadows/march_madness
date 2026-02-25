"""Load all competition CSVs. One function per dataset, caching, typed DataFrames."""
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src import config

_cache: dict = {}


def _cache_key(*parts) -> str:
    return "_".join(str(p) for p in parts)


def _csv(filename: str) -> Path:
    return config.DATA_DIR / filename


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

def load_teams(gender: str = 'M') -> pd.DataFrame:
    """Load team master list.

    Returns:
        DataFrame with columns: TeamID, TeamName, FirstD1Season, LastD1Season
    """
    key = _cache_key('teams', gender)
    if key not in _cache:
        df = pd.read_csv(_csv(f'{gender}Teams.csv'), dtype={'TeamID': 'int32'})
        _cache[key] = df
    return _cache[key]


# ---------------------------------------------------------------------------
# Regular Season
# ---------------------------------------------------------------------------

def load_regular_season(gender: str = 'M', detailed: bool = False) -> pd.DataFrame:
    """Load regular season game results.

    Args:
        gender: 'M' or 'W'
        detailed: If True, load box score version; if False, compact version.

    Returns:
        DataFrame with game results. Compact has 8 cols; detailed has 34 cols.
    """
    detail_str = 'Detailed' if detailed else 'Compact'
    key = _cache_key('regular', gender, detail_str)
    if key not in _cache:
        df = pd.read_csv(
            _csv(f'{gender}RegularSeason{detail_str}Results.csv'),
            dtype={'Season': 'int16', 'WTeamID': 'int32', 'LTeamID': 'int32'},
        )
        _cache[key] = df
    return _cache[key]


# ---------------------------------------------------------------------------
# Tournament Results
# ---------------------------------------------------------------------------

def load_tourney_results(gender: str = 'M', detailed: bool = False) -> pd.DataFrame:
    """Load NCAA tournament game results.

    Args:
        gender: 'M' or 'W'
        detailed: If True, load box score version; if False, compact version.

    Returns:
        DataFrame with tournament game results.
    """
    detail_str = 'Detailed' if detailed else 'Compact'
    key = _cache_key('tourney', gender, detail_str)
    if key not in _cache:
        df = pd.read_csv(
            _csv(f'{gender}NCAATourney{detail_str}Results.csv'),
            dtype={'Season': 'int16', 'WTeamID': 'int32', 'LTeamID': 'int32'},
        )
        _cache[key] = df
    return _cache[key]


# ---------------------------------------------------------------------------
# Tournament Seeds
# ---------------------------------------------------------------------------

def load_tourney_seeds(gender: str = 'M') -> pd.DataFrame:
    """Load tournament seeds with parsed numeric seed.

    Returns:
        DataFrame with columns: Season, Seed (raw string), TeamID, SeedNum (int 1-16)
    """
    key = _cache_key('seeds', gender)
    if key not in _cache:
        df = pd.read_csv(
            _csv(f'{gender}NCAATourneySeeds.csv'),
            dtype={'Season': 'int16', 'TeamID': 'int32'},
        )
        df['SeedNum'] = df['Seed'].apply(parse_seed)
        _cache[key] = df
    return _cache[key]


def parse_seed(seed_str: str) -> int:
    """Extract numeric seed from string like 'W01', 'X16a', 'Z11b'.

    Args:
        seed_str: Raw seed string from competition data.

    Returns:
        Integer seed number (1-16).
    """
    return int(seed_str[1:3])


# ---------------------------------------------------------------------------
# Tournament Slots
# ---------------------------------------------------------------------------

def load_tourney_slots(gender: str = 'M') -> pd.DataFrame:
    """Load tournament bracket slot structure.

    Returns:
        DataFrame with columns: Season, Slot, StrongSeed, WeakSeed
    """
    key = _cache_key('slots', gender)
    if key not in _cache:
        df = pd.read_csv(
            _csv(f'{gender}NCAATourneySlots.csv'),
            dtype={'Season': 'int16'},
        )
        _cache[key] = df
    return _cache[key]


# ---------------------------------------------------------------------------
# Massey Ordinals (Men only)
# ---------------------------------------------------------------------------

def load_massey_ordinals() -> pd.DataFrame:
    """Load Massey Ordinals rankings (Men only, ~5.8M rows).

    Uses memory-efficient dtypes. Only available for M.

    Returns:
        DataFrame with columns: Season, RankingDayNum, SystemName, TeamID, OrdinalRank
    """
    key = 'massey'
    if key not in _cache:
        dtypes = {
            'Season': 'int16',
            'RankingDayNum': 'int16',
            'SystemName': 'category',
            'TeamID': 'int32',
            'OrdinalRank': 'int16',
        }
        df = pd.read_csv(_csv('MMasseyOrdinals.csv'), dtype=dtypes)
        _cache[key] = df
    return _cache[key]


# ---------------------------------------------------------------------------
# Conferences
# ---------------------------------------------------------------------------

def load_conferences(gender: str = 'M') -> pd.DataFrame:
    """Load team conference assignments per season.

    Returns:
        DataFrame with columns: Season, TeamID, ConfAbbrev
    """
    key = _cache_key('conferences', gender)
    if key not in _cache:
        df = pd.read_csv(
            _csv(f'{gender}TeamConferences.csv'),
            dtype={'Season': 'int16', 'TeamID': 'int32'},
        )
        _cache[key] = df
    return _cache[key]


def load_conference_list() -> pd.DataFrame:
    """Load master conference list (gender-neutral).

    Returns:
        DataFrame with conference abbreviations and names.
    """
    key = 'conference_list'
    if key not in _cache:
        _cache[key] = pd.read_csv(_csv('Conferences.csv'))
    return _cache[key]


# ---------------------------------------------------------------------------
# Coaches (Men only)
# ---------------------------------------------------------------------------

def load_coaches() -> pd.DataFrame:
    """Load team coaching records (Men only).

    Returns:
        DataFrame with columns: Season, TeamID, FirstDayNum, LastDayNum, CoachName
    """
    key = 'coaches'
    if key not in _cache:
        df = pd.read_csv(
            _csv('MTeamCoaches.csv'),
            dtype={'Season': 'int16', 'TeamID': 'int32'},
        )
        _cache[key] = df
    return _cache[key]


# ---------------------------------------------------------------------------
# Cities
# ---------------------------------------------------------------------------

def load_cities() -> pd.DataFrame:
    """Load city reference table (gender-neutral).

    Returns:
        DataFrame with columns: CityID, City, State
    """
    key = 'cities'
    if key not in _cache:
        _cache[key] = pd.read_csv(_csv('Cities.csv'), dtype={'CityID': 'int32'})
    return _cache[key]


def load_game_cities(gender: str = 'M') -> pd.DataFrame:
    """Load game location data.

    Returns:
        DataFrame with columns: Season, DayNum, WTeamID, LTeamID, CRType, CityID
    """
    key = _cache_key('game_cities', gender)
    if key not in _cache:
        df = pd.read_csv(
            _csv(f'{gender}GameCities.csv'),
            dtype={'Season': 'int16', 'WTeamID': 'int32', 'LTeamID': 'int32',
                   'CityID': 'int32'},
        )
        _cache[key] = df
    return _cache[key]


# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------

def load_seasons(gender: str = 'M') -> pd.DataFrame:
    """Load season metadata (DayZero, region assignments).

    Returns:
        DataFrame with columns: Season, DayZero, RegionW, RegionX, RegionY, RegionZ
    """
    key = _cache_key('seasons', gender)
    if key not in _cache:
        df = pd.read_csv(_csv(f'{gender}Seasons.csv'), dtype={'Season': 'int16'})
        _cache[key] = df
    return _cache[key]


# ---------------------------------------------------------------------------
# Conference Tournament
# ---------------------------------------------------------------------------

def load_conference_tourney(gender: str = 'M') -> pd.DataFrame:
    """Load conference tournament game results.

    Returns:
        DataFrame with columns: Season, ConfAbbrev, DayNum, WTeamID, LTeamID
    """
    key = _cache_key('conf_tourney', gender)
    if key not in _cache:
        df = pd.read_csv(
            _csv(f'{gender}ConferenceTourneyGames.csv'),
            dtype={'Season': 'int16', 'WTeamID': 'int32', 'LTeamID': 'int32'},
        )
        _cache[key] = df
    return _cache[key]


# ---------------------------------------------------------------------------
# Secondary Tournament (NIT, CBI, etc.)
# ---------------------------------------------------------------------------

def load_secondary_tourney(gender: str = 'M') -> pd.DataFrame:
    """Load secondary tournament (NIT/CBI) game results.

    Returns:
        DataFrame with game results including SecondaryTourney column (M only).
    """
    key = _cache_key('secondary_tourney', gender)
    if key not in _cache:
        df = pd.read_csv(
            _csv(f'{gender}SecondaryTourneyCompactResults.csv'),
            dtype={'Season': 'int16', 'WTeamID': 'int32', 'LTeamID': 'int32'},
        )
        _cache[key] = df
    return _cache[key]


# ---------------------------------------------------------------------------
# Sample Submission
# ---------------------------------------------------------------------------

def load_sample_submission(stage: int = 1) -> pd.DataFrame:
    """Load sample submission file.

    Args:
        stage: 1 or 2.

    Returns:
        DataFrame with columns: ID, Pred. ID format: Season_TeamA_TeamB.
    """
    key = _cache_key('submission', stage)
    if key not in _cache:
        df = pd.read_csv(_csv(f'SampleSubmissionStage{stage}.csv'))
        _cache[key] = df
    return _cache[key]


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def clear_cache() -> None:
    """Clear the in-memory data cache."""
    _cache.clear()
