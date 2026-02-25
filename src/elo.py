"""Elo rating system for NCAA basketball.

Ratings initialized at 1500, updated after every game chronologically.
Home court advantage applied. Margin of victory scaling used.
Season-to-season: regress 75% toward mean (1500).
"""
import json
import math
from pathlib import Path
from typing import Optional

import pandas as pd

from src import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELO_INIT = 1500.0
ELO_K = 20.0          # Base K-factor
HCA = 100.0           # Home court advantage in Elo points
REGRESS_FACTOR = 0.75  # 75% own rating + 25% mean on season reset


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def elo_to_win_prob(elo_diff: float) -> float:
    """Convert Elo difference to win probability using logistic function.

    Args:
        elo_diff: elo_a - elo_b (positive = team_a favored)

    Returns:
        Probability that team_a wins, in [0, 1].
    """
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def _mov_multiplier(margin: int, elo_diff_winner: float) -> float:
    """FiveThirtyEight-style margin-of-victory multiplier.

    Accounts for the fact that bigger wins provide more information,
    but diminishing returns and autocorrelation corrections apply.

    Args:
        margin: Winning margin (positive integer).
        elo_diff_winner: Pre-game Elo advantage of the winning team.

    Returns:
        MOV multiplier (always >= 1.0).
    """
    # Autocorrelation correction: penalize teams that were already heavily favored
    autocorr = 2.2 / (elo_diff_winner * 0.001 + 2.2)
    return math.log(abs(margin) + 1) * autocorr


def _update_elos(
    winner_elo: float,
    loser_elo: float,
    margin: int,
    wloc: str,
) -> tuple[float, float]:
    """Compute new Elos after a game.

    Args:
        winner_elo: Pre-game Elo of winner.
        loser_elo: Pre-game Elo of loser.
        margin: Score difference (winner - loser), must be >= 1.
        wloc: Game location from winner's perspective: 'H', 'A', or 'N'.

    Returns:
        (new_winner_elo, new_loser_elo)
    """
    # Apply home court adjustment to winner's effective Elo
    if wloc == 'H':
        adj_winner = winner_elo + HCA
    elif wloc == 'A':
        adj_winner = winner_elo - HCA
    else:
        adj_winner = winner_elo

    expected_winner = elo_to_win_prob(adj_winner - loser_elo)
    elo_diff_winner = adj_winner - loser_elo
    k = ELO_K * _mov_multiplier(margin, elo_diff_winner)

    delta = k * (1.0 - expected_winner)
    return winner_elo + delta, loser_elo - delta


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_elo_ratings(
    games_df: pd.DataFrame,
    carry_over: Optional[dict[int, float]] = None,
) -> dict[int, dict[int, float]]:
    """Compute Elo ratings season-by-season from game results.

    Games within each season are processed chronologically (by DayNum).
    Between seasons, ratings regress toward the mean.

    Args:
        games_df: DataFrame with columns Season, DayNum, WTeamID, LTeamID,
                  WScore, LScore, WLoc.
        carry_over: Optional initial Elo dict {team_id: elo} to seed from.

    Returns:
        Dict {season: {team_id: elo_at_end_of_season}}
        The elo at end of season is AFTER playing all regular season games
        but BEFORE tournament games (tournament games not typically included).
    """
    # Sort by season then day
    games_df = games_df.sort_values(['Season', 'DayNum']).reset_index(drop=True)

    current_elos: dict[int, float] = dict(carry_over) if carry_over else {}
    season_ratings: dict[int, dict[int, float]] = {}

    seasons = sorted(games_df['Season'].unique())

    for season in seasons:
        season_games = games_df[games_df['Season'] == season]

        # Initialize any new teams
        all_teams = set(season_games['WTeamID']) | set(season_games['LTeamID'])
        for tid in all_teams:
            if tid not in current_elos:
                current_elos[tid] = ELO_INIT

        # Process games in order
        for _, row in season_games.iterrows():
            w_id = int(row['WTeamID'])
            l_id = int(row['LTeamID'])
            margin = int(row['WScore']) - int(row['LScore'])
            wloc = str(row['WLoc'])

            new_w, new_l = _update_elos(
                current_elos[w_id], current_elos[l_id], margin, wloc
            )
            current_elos[w_id] = new_w
            current_elos[l_id] = new_l

        # Snapshot end-of-season ratings (before regression)
        season_ratings[season] = {tid: elo for tid, elo in current_elos.items()}

        # Regress to mean for next season
        for tid in list(current_elos.keys()):
            current_elos[tid] = REGRESS_FACTOR * current_elos[tid] + (1 - REGRESS_FACTOR) * ELO_INIT

    return season_ratings


def get_pre_tourney_elo(
    season_ratings: dict[int, dict[int, float]],
    season: int,
    team_id: int,
) -> float:
    """Get a team's Elo rating at end of regular season (pre-tournament).

    Args:
        season_ratings: Output of compute_elo_ratings.
        season: Season year.
        team_id: Team ID.

    Returns:
        Elo rating, or ELO_INIT if not found.
    """
    return season_ratings.get(season, {}).get(team_id, ELO_INIT)


def save_ratings(
    season_ratings: dict[int, dict[int, float]],
    path: Path,
) -> None:
    """Serialize Elo ratings to JSON.

    Args:
        season_ratings: Output of compute_elo_ratings.
        path: Output file path.
    """
    # JSON requires string keys
    serializable = {
        str(season): {str(tid): elo for tid, elo in teams.items()}
        for season, teams in season_ratings.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(serializable, f)


def load_ratings(path: Path) -> dict[int, dict[int, float]]:
    """Load Elo ratings from JSON.

    Args:
        path: JSON file written by save_ratings.

    Returns:
        Dict {season: {team_id: elo}}
    """
    with open(path) as f:
        raw = json.load(f)
    return {
        int(season): {int(tid): elo for tid, elo in teams.items()}
        for season, teams in raw.items()
    }
