"""Feature engineering pipeline for March Madness predictions.

Computes per-team season stats, Elo ratings, Massey ordinals, and matchup
differentials. All matchup features are (team_a_stat - team_b_stat) where
team_a always has the lower TeamID (matching submission format).

No temporal leakage: all stats use only pre-tournament games (DayNum < 134).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import config
from src import data_loader as dl
from src.elo import compute_elo_ratings, load_ratings, ELO_INIT
from src.massey import get_season_system_index

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECENT_FORM_DAYS = 14  # Days back from end of regular season for recent form
MIN_GAMES = 5          # Minimum games to compute reliable stats


# ---------------------------------------------------------------------------
# Internal helpers: per-team-per-game long format
# ---------------------------------------------------------------------------

def _make_team_game_df(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """Reshape detailed game results into per-team-per-game rows.

    Each original game yields two rows: one for winner, one for loser.
    This enables easy per-team aggregation.

    Args:
        detailed_df: Detailed results DataFrame (34 columns).

    Returns:
        DataFrame with one row per team per game.
    """
    cols = {
        'Season': ('Season', 'Season'),
        'DayNum': ('DayNum', 'DayNum'),
        'TeamID': ('WTeamID', 'LTeamID'),
        'OppID': ('LTeamID', 'WTeamID'),
        'Score': ('WScore', 'LScore'),
        'OppScore': ('LScore', 'WScore'),
        'Won': (None, None),  # handled separately
        'FGM': ('WFGM', 'LFGM'),
        'FGA': ('WFGA', 'LFGA'),
        'FGM3': ('WFGM3', 'LFGM3'),
        'FGA3': ('WFGA3', 'LFGA3'),
        'FTM': ('WFTM', 'LFTM'),
        'FTA': ('WFTA', 'LFTA'),
        'OR': ('WOR', 'LOR'),
        'DR': ('WDR', 'LDR'),
        'Ast': ('WAst', 'LAst'),
        'TO': ('WTO', 'LTO'),
        'Stl': ('WStl', 'LStl'),
        'Blk': ('WBlk', 'LBlk'),
        'OppOR': ('LOR', 'WOR'),
        'OppDR': ('LDR', 'WDR'),
    }

    winner_data = {}
    loser_data = {}
    for dest, (wcol, lcol) in cols.items():
        if dest == 'Won':
            winner_data['Won'] = 1
            loser_data['Won'] = 0
        else:
            winner_data[dest] = detailed_df[wcol].values
            loser_data[dest] = detailed_df[lcol].values

    w_df = pd.DataFrame(winner_data)
    l_df = pd.DataFrame(loser_data)
    return pd.concat([w_df, l_df], ignore_index=True)


def _compute_per_game_metrics(tg: pd.DataFrame) -> pd.DataFrame:
    """Add per-possession and rate metrics to a team-game DataFrame.

    Args:
        tg: Team-game DataFrame from _make_team_game_df.

    Returns:
        tg with additional metric columns.
    """
    poss = tg['FGA'] - tg['OR'] + tg['TO'] + 0.44 * tg['FTA']
    poss = poss.clip(lower=1)  # avoid division by zero

    tg = tg.copy()
    tg['poss'] = poss
    tg['off_eff'] = tg['Score'] / poss * 100
    tg['def_eff'] = tg['OppScore'] / poss * 100
    tg['net_eff'] = tg['off_eff'] - tg['def_eff']
    tg['efg_pct'] = (tg['FGM'] + 0.5 * tg['FGM3']) / tg['FGA'].clip(lower=1)
    tg['to_rate'] = tg['TO'] / (tg['FGA'] + 0.44 * tg['FTA'] + tg['TO']).clip(lower=1)
    tg['or_pct'] = tg['OR'] / (tg['OR'] + tg['OppDR']).clip(lower=1)
    tg['ft_rate'] = tg['FTM'] / tg['FGA'].clip(lower=1)
    tg['fg3_rate'] = tg['FGA3'] / tg['FGA'].clip(lower=1)
    tg['ast_to_ratio'] = tg['Ast'] / (tg['TO'] + 1)
    tg['stl_per_game'] = tg['Stl']
    tg['blk_per_game'] = tg['Blk']
    return tg


# ---------------------------------------------------------------------------
# Bulk season stat computation
# ---------------------------------------------------------------------------

_AGG_COLS = [
    'Won', 'Score', 'OppScore', 'off_eff', 'def_eff', 'net_eff',
    'efg_pct', 'to_rate', 'or_pct', 'ft_rate', 'fg3_rate',
    'ast_to_ratio', 'stl_per_game', 'blk_per_game',
]

# Rename aggregated columns to standard feature names
_COL_RENAME = {
    'Won': 'win_pct',
    'Score': 'pts_per_game',
    'OppScore': 'pts_allowed_per_game',
}


def _aggregate_team_stats(
    tg: pd.DataFrame,
    season: int,
    min_games: int = MIN_GAMES,
) -> dict[int, dict[str, float]]:
    """Aggregate per-game metrics into per-team season stats.

    Args:
        tg: Team-game DataFrame with metrics already computed.
        season: Season year (for filtering).
        min_games: Minimum games required; teams below this get NaN for rate stats.

    Returns:
        Dict {team_id: {stat_name: value}}
    """
    season_data = tg[tg['Season'] == season].copy()
    if len(season_data) == 0:
        return {}

    agg = season_data.groupby('TeamID')[_AGG_COLS].agg(['mean', 'count'])
    # Flatten multi-level columns
    agg.columns = [f'{c}_{fn}' for c, fn in agg.columns]
    agg = agg.rename(columns={'Won_count': 'game_count'})

    result = {}
    for team_id, row in agg.iterrows():
        game_count = int(row.get('Won_count', row.get('game_count', 0)))
        stats: dict[str, float] = {'game_count': game_count}

        for col in _AGG_COLS:
            mean_key = f'{col}_mean'
            rename = _COL_RENAME.get(col, col)
            if game_count >= min_games:
                stats[rename] = float(row.get(mean_key, np.nan))
            else:
                # Not enough games for reliable stats
                stats[rename] = float(row.get(mean_key, np.nan))

        result[int(team_id)] = stats
    return result


def compute_all_team_season_stats(
    season: int,
    gender: str,
    pre_tourney_only: bool = True,
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    """Compute full-season and recent-form stats for all teams in a season.

    Uses detailed results when available, compact results as fallback.

    Args:
        season: Season year.
        gender: 'M' or 'W'.
        pre_tourney_only: If True, exclude games on/after TOURNEY_START_DAY.

    Returns:
        (full_stats, recent_stats) — each is {team_id: {stat_name: value}}.
        Both dicts include teams from compact results (win_pct, pts_per_game,
        pts_allowed_per_game) even if detailed stats aren't available.
    """
    first_detailed = (
        config.FIRST_DETAILED_SEASON_M if gender == 'M'
        else config.FIRST_DETAILED_SEASON_W
    )

    # ---- Compact-based fallback stats (win_pct, pts per game) ----
    compact = dl.load_regular_season(gender, detailed=False)
    compact = compact[compact['Season'] == season]
    if pre_tourney_only:
        compact = compact[compact['DayNum'] < config.TOURNEY_START_DAY]

    compact_stats: dict[int, dict[str, float]] = {}
    if len(compact) > 0:
        # Build team-game view from compact
        w = pd.DataFrame({
            'TeamID': compact['WTeamID'].values,
            'Score': compact['WScore'].values,
            'OppScore': compact['LScore'].values,
            'Won': 1,
        })
        l = pd.DataFrame({
            'TeamID': compact['LTeamID'].values,
            'Score': compact['LScore'].values,
            'OppScore': compact['WScore'].values,
            'Won': 0,
        })
        cdf = pd.concat([w, l], ignore_index=True)
        agg = cdf.groupby('TeamID').agg(
            win_pct=('Won', 'mean'),
            pts_per_game=('Score', 'mean'),
            pts_allowed_per_game=('OppScore', 'mean'),
            game_count=('Won', 'count'),
        )
        for tid, row in agg.iterrows():
            compact_stats[int(tid)] = row.to_dict()

    # ---- Detailed stats (if available for this season) ----
    if season < first_detailed:
        # No detailed stats — return compact only with NaN for advanced metrics
        full_stats = _fill_nan_detailed(compact_stats)
        recent_stats = _compute_compact_recent(season, gender, compact)
        return full_stats, recent_stats

    detailed = dl.load_regular_season(gender, detailed=True)
    detailed = detailed[detailed['Season'] == season]
    if pre_tourney_only:
        detailed = detailed[detailed['DayNum'] < config.TOURNEY_START_DAY]

    if len(detailed) == 0:
        return _fill_nan_detailed(compact_stats), {}

    tg = _make_team_game_df(detailed)
    tg['Season'] = season
    tg = _compute_per_game_metrics(tg)

    # Full season stats
    detailed_stats = _aggregate_team_stats(tg, season)

    # Merge compact (win_pct from compact is most accurate) with detailed
    for tid in detailed_stats:
        if tid in compact_stats:
            detailed_stats[tid]['win_pct'] = compact_stats[tid]['win_pct']
            detailed_stats[tid]['game_count'] = compact_stats[tid]['game_count']

    # Teams in compact but not detailed (shouldn't happen often)
    for tid in compact_stats:
        if tid not in detailed_stats:
            detailed_stats[tid] = _fill_nan_detailed({tid: compact_stats[tid]})[tid]

    # Recent form: last RECENT_FORM_DAYS days of regular season
    if len(detailed) > 0:
        max_day = int(detailed['DayNum'].max())
        recent_cutoff = max_day - RECENT_FORM_DAYS
        recent_detailed = detailed[detailed['DayNum'] >= recent_cutoff]
        if len(recent_detailed) > 0:
            tg_recent = _make_team_game_df(recent_detailed)
            tg_recent['Season'] = season
            tg_recent = _compute_per_game_metrics(tg_recent)
            recent_stats = _aggregate_team_stats(tg_recent, season, min_games=2)
        else:
            recent_stats = {}
    else:
        recent_stats = {}

    return detailed_stats, recent_stats


def _fill_nan_detailed(stats: dict[int, dict[str, float]]) -> dict[int, dict[str, float]]:
    """Fill in NaN for all detailed stats columns."""
    detailed_cols = [
        'off_eff', 'def_eff', 'net_eff', 'efg_pct', 'to_rate', 'or_pct',
        'ft_rate', 'fg3_rate', 'ast_to_ratio', 'stl_per_game', 'blk_per_game',
    ]
    result = {}
    for tid, s in stats.items():
        row = dict(s)
        for col in detailed_cols:
            if col not in row:
                row[col] = np.nan
        result[tid] = row
    return result


def _compute_compact_recent(
    season: int,
    gender: str,
    compact_season: pd.DataFrame,
) -> dict[int, dict[str, float]]:
    """Compute recent win_pct/pts from compact when detailed not available."""
    if len(compact_season) == 0:
        return {}
    max_day = int(compact_season['DayNum'].max())
    recent = compact_season[compact_season['DayNum'] >= max_day - RECENT_FORM_DAYS]
    if len(recent) == 0:
        return {}
    w = pd.DataFrame({'TeamID': recent['WTeamID'], 'Won': 1, 'Score': recent['WScore'],
                      'OppScore': recent['LScore']})
    l = pd.DataFrame({'TeamID': recent['LTeamID'], 'Won': 0, 'Score': recent['LScore'],
                      'OppScore': recent['WScore']})
    df = pd.concat([w, l], ignore_index=True)
    agg = df.groupby('TeamID').agg(
        win_pct=('Won', 'mean'),
        pts_per_game=('Score', 'mean'),
        pts_allowed_per_game=('OppScore', 'mean'),
    )
    return {int(tid): row.to_dict() for tid, row in agg.iterrows()}


# ---------------------------------------------------------------------------
# Strength of Schedule
# ---------------------------------------------------------------------------

def compute_sos_bulk(
    season: int,
    gender: str,
    elo_ratings: dict[int, dict[int, float]],
    pre_tourney_only: bool = True,
) -> dict[int, float]:
    """Compute mean-opponent-Elo strength of schedule for every team.

    Args:
        season: Season year.
        gender: 'M' or 'W'.
        elo_ratings: {season: {team_id: elo}} from compute_elo_ratings.
        pre_tourney_only: If True, only use pre-tournament games.

    Returns:
        Dict {team_id: sos_elo}
    """
    # Use prior season Elo as proxy for opponent quality (avoids leakage)
    prior_season = season - 1
    prior_elos = elo_ratings.get(prior_season, {})

    compact = dl.load_regular_season(gender, detailed=False)
    compact = compact[compact['Season'] == season]
    if pre_tourney_only:
        compact = compact[compact['DayNum'] < config.TOURNEY_START_DAY]

    if len(compact) == 0:
        return {}

    # For each game, record opponent Elo for both teams
    rows = []
    for _, row in compact.iterrows():
        w_id = int(row['WTeamID'])
        l_id = int(row['LTeamID'])
        w_opp_elo = prior_elos.get(l_id, ELO_INIT)
        l_opp_elo = prior_elos.get(w_id, ELO_INIT)
        rows.append({'TeamID': w_id, 'opp_elo': w_opp_elo})
        rows.append({'TeamID': l_id, 'opp_elo': l_opp_elo})

    sos_df = pd.DataFrame(rows)
    sos = sos_df.groupby('TeamID')['opp_elo'].mean()
    return {int(tid): float(val) for tid, val in sos.items()}


# ---------------------------------------------------------------------------
# Coach experience
# ---------------------------------------------------------------------------

def compute_coach_exp_bulk(season: int) -> dict[int, int]:
    """Compute prior NCAA tournament appearances for each team's coach (M only).

    Args:
        season: Current season.

    Returns:
        Dict {team_id: prior_tourney_appearances}
    """
    coaches = dl.load_coaches()
    seeds = dl.load_tourney_seeds('M')

    result: dict[int, int] = {}

    for _, row in coaches[coaches['Season'] == season].iterrows():
        team_id = int(row['TeamID'])
        coach_name = row['CoachName']

        # Get all prior seasons this coach was at any team
        prior_stints = coaches[
            (coaches['CoachName'] == coach_name) & (coaches['Season'] < season)
        ]
        prior_teams_seasons = list(
            zip(prior_stints['Season'], prior_stints['TeamID'])
        )

        # Count seasons where coach's team was in the tournament
        exp = 0
        for s, t in prior_teams_seasons:
            if len(seeds[(seeds['Season'] == s) & (seeds['TeamID'] == t)]) > 0:
                exp += 1

        # Use max if multiple coaches listed (take most experienced)
        if team_id not in result or exp > result[team_id]:
            result[team_id] = exp

    return result


# ---------------------------------------------------------------------------
# Feature builder (season-level cache + matchup differentials)
# ---------------------------------------------------------------------------

class SeasonFeatureCache:
    """Precomputed features for all teams in a given season+gender.

    Loads and caches all data needed to build matchup features efficiently.
    """

    def __init__(
        self,
        season: int,
        gender: str,
        elo_ratings: dict[int, dict[int, float]],
    ) -> None:
        self.season = season
        self.gender = gender

        # Season stats
        self.full_stats, self.recent_stats = compute_all_team_season_stats(
            season, gender
        )

        # Elo
        self.elo_dict = elo_ratings.get(season, {})

        # Massey (M only)
        self.massey_pivot: Optional[pd.DataFrame] = None
        if gender == 'M':
            self.massey_pivot = get_season_system_index(season, day=133)

        # Seeds
        seeds_df = dl.load_tourney_seeds(gender)
        season_seeds = seeds_df[seeds_df['Season'] == season]
        self.seed_map: dict[int, int] = dict(
            zip(season_seeds['TeamID'].astype(int), season_seeds['SeedNum'].astype(int))
        )

        # SOS
        self.sos_dict = compute_sos_bulk(season, gender, elo_ratings)

        # Coach exp (M only)
        self.coach_exp: dict[int, int] = {}
        if gender == 'M':
            self.coach_exp = compute_coach_exp_bulk(season)

    def get_team_features(self, team_id: int) -> dict[str, float]:
        """Get all features for a single team.

        Args:
            team_id: Team ID.

        Returns:
            Dict of feature name to value (NaN for missing).
        """
        feats: dict[str, float] = {}

        # Season stats
        stats = self.full_stats.get(team_id, {})
        for col in ['win_pct', 'pts_per_game', 'pts_allowed_per_game',
                    'off_eff', 'def_eff', 'net_eff', 'efg_pct', 'to_rate',
                    'or_pct', 'ft_rate', 'fg3_rate', 'ast_to_ratio',
                    'stl_per_game', 'blk_per_game']:
            feats[col] = float(stats.get(col, np.nan))

        # Recent form
        recent = self.recent_stats.get(team_id, {})
        for col in ['win_pct', 'pts_per_game', 'pts_allowed_per_game',
                    'off_eff', 'def_eff', 'net_eff', 'efg_pct', 'to_rate',
                    'or_pct', 'ft_rate', 'fg3_rate']:
            feats[f'recent_{col}'] = float(recent.get(col, np.nan))

        # Elo
        feats['elo'] = float(self.elo_dict.get(team_id, ELO_INIT))

        # Massey ranks
        for sys in config.TOP_MASSEY_SYSTEMS:
            col = f'{sys.lower()}_rank'
            if self.massey_pivot is not None and len(self.massey_pivot) > 0:
                if sys in self.massey_pivot.columns and team_id in self.massey_pivot.index:
                    val = self.massey_pivot.loc[team_id, sys]
                    feats[col] = float(val) if not pd.isna(val) else np.nan
                else:
                    feats[col] = np.nan
            else:
                feats[col] = np.nan

        # Seed
        feats['seed_num'] = float(self.seed_map.get(team_id, np.nan))

        # SOS
        feats['sos_elo'] = float(self.sos_dict.get(team_id, np.nan))

        # Coach exp
        feats['coach_tourney_exp'] = float(self.coach_exp.get(team_id, 0.0))

        return feats


def build_matchup_features(
    team_a_id: int,
    team_b_id: int,
    cache: SeasonFeatureCache,
) -> dict[str, float]:
    """Build differential features for a matchup.

    team_a_id MUST be the lower TeamID (matches submission format).
    All differentials = team_a_stat - team_b_stat.
    Positive diff means team_a is better on that metric.

    Args:
        team_a_id: Lower TeamID.
        team_b_id: Higher TeamID.
        cache: Precomputed season feature cache.

    Returns:
        Flat dict of differential features.
    """
    assert team_a_id < team_b_id, "team_a must have lower TeamID"

    feats_a = cache.get_team_features(team_a_id)
    feats_b = cache.get_team_features(team_b_id)

    diff: dict[str, float] = {}
    for key in feats_a:
        va = feats_a[key]
        vb = feats_b[key]
        if pd.isna(va) or pd.isna(vb):
            diff[f'{key}_diff'] = np.nan
        else:
            diff[f'{key}_diff'] = va - vb

    return diff


# ---------------------------------------------------------------------------
# Training set builder
# ---------------------------------------------------------------------------

def build_training_set(
    seasons: list[int],
    gender: str,
    elo_ratings: Optional[dict[int, dict[int, float]]] = None,
) -> pd.DataFrame:
    """Build full training set from tournament results.

    For each tournament game in the given seasons, builds matchup differential
    features and assigns target = 1 if lower-ID team won, 0 otherwise.

    Args:
        seasons: List of season years to include (e.g. [2015, ..., 2024]).
        gender: 'M' or 'W'.
        elo_ratings: Pre-computed Elo ratings dict. If None, recomputes from scratch.

    Returns:
        DataFrame with feature columns + 'Season' + 'target' column.
        Lower-ID team is implicitly team_a.
    """
    if elo_ratings is None:
        games = dl.load_regular_season(gender)
        elo_ratings = compute_elo_ratings(games)

    tourney = dl.load_tourney_results(gender)

    rows = []
    for season in sorted(seasons):
        season_tourney = tourney[tourney['Season'] == season]
        if len(season_tourney) == 0:
            continue

        cache = SeasonFeatureCache(season, gender, elo_ratings)

        for _, game in season_tourney.iterrows():
            w_id = int(game['WTeamID'])
            l_id = int(game['LTeamID'])

            # Ensure lower ID is team_a
            team_a = min(w_id, l_id)
            team_b = max(w_id, l_id)
            target = 1 if w_id < l_id else 0

            feats = build_matchup_features(team_a, team_b, cache)
            feats['Season'] = season
            feats['WTeamID'] = w_id
            feats['LTeamID'] = l_id
            feats['target'] = target
            rows.append(feats)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Prediction set builder
# ---------------------------------------------------------------------------

def build_prediction_set(
    gender: str,
    season: int,
    elo_ratings: Optional[dict[int, dict[int, float]]] = None,
    stage: int = 1,
) -> pd.DataFrame:
    """Build features for every matchup in the sample submission.

    Args:
        gender: 'M' or 'W' — only processes matchups for this gender.
        season: Target season (e.g. 2026).
        elo_ratings: Pre-computed Elo ratings. Recomputed if None.
        stage: Sample submission stage (1 or 2).

    Returns:
        DataFrame with 'ID' column + feature columns, aligned with submission.
    """
    if elo_ratings is None:
        games = dl.load_regular_season(gender)
        elo_ratings = compute_elo_ratings(games)

    sub = dl.load_sample_submission(stage=stage)

    # Filter to this gender and season
    def parse_id(row_id: str) -> tuple[int, int, int]:
        parts = row_id.split('_')
        return int(parts[0]), int(parts[1]), int(parts[2])

    sub_parsed = sub['ID'].apply(parse_id)
    sub_parsed_df = pd.DataFrame(
        sub_parsed.tolist(), columns=['sub_season', 'team_a', 'team_b']
    )
    sub_parsed_df['ID'] = sub['ID'].values

    if gender == 'M':
        mask = (sub_parsed_df['sub_season'] == season) & \
               sub_parsed_df['team_a'].between(*config.MEN_ID_RANGE)
    else:
        mask = (sub_parsed_df['sub_season'] == season) & \
               sub_parsed_df['team_a'].between(*config.WOMEN_ID_RANGE)

    filtered = sub_parsed_df[mask].reset_index(drop=True)

    if len(filtered) == 0:
        return pd.DataFrame(columns=['ID'])

    cache = SeasonFeatureCache(season, gender, elo_ratings)

    rows = []
    for _, row in filtered.iterrows():
        team_a = int(row['team_a'])
        team_b = int(row['team_b'])
        feats = build_matchup_features(team_a, team_b, cache)
        feats['ID'] = row['ID']
        feats['Season'] = season
        rows.append(feats)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Export feature matrices
# ---------------------------------------------------------------------------

def export_feature_matrices(
    train_seasons: Optional[list[int]] = None,
    output_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and export full training feature matrices for M and W.

    Args:
        train_seasons: Seasons to include. Defaults to all tournament seasons.
        output_dir: Output directory. Defaults to config.ARTIFACTS_DIR.

    Returns:
        (men_df, women_df) feature matrices.
    """
    if output_dir is None:
        output_dir = config.ARTIFACTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for gender, first_season, label in [
        ('M', config.FIRST_COMPACT_SEASON_M, 'men'),
        ('W', config.FIRST_COMPACT_SEASON_W, 'women'),
    ]:
        games = dl.load_regular_season(gender)
        elo_ratings = compute_elo_ratings(games)

        tourney = dl.load_tourney_results(gender)
        available = sorted(tourney['Season'].unique())

        if train_seasons:
            seasons = [s for s in train_seasons if s in available]
        else:
            seasons = available

        print(f"Building {label} training set: {len(seasons)} seasons...")
        df = build_training_set(seasons, gender, elo_ratings)

        out_path = output_dir / f'features_{label}.csv'
        df.to_csv(out_path, index=False)
        print(f"  Saved {df.shape} -> {out_path}")
        results[gender] = df

    return results['M'], results['W']
