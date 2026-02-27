"""Barttorvik (T-Rank) data loading and team name mapping.

Loads Barttorvik advanced team ratings (adjoe, adjde, barthag, etc.)
and maps team names to Kaggle TeamIDs via MTeamSpellings/WTeamSpellings.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
BARTTORVIK_DIR = DATA_DIR / "barttorvik"

# 9 features we extract from Barttorvik
BARTTORVIK_FEATURES = [
    "adjoe",
    "adjde",
    "barthag",
    "adjt",
    "wab",
    "elite_sos",
    "qual_o",
    "qual_d",
    "qual_barthag",
]

# Barttorvik CSV column name -> our snake_case name
_COLUMN_RENAME = {
    "adjoe": "adjoe",
    "adjde": "adjde",
    "barthag": "barthag",
    "adjt": "adjt",
    "WAB": "wab",
    "elite.SOS": "elite_sos",
    "Qual.O": "qual_o",
    "Qual.D": "qual_d",
    "Qual.Barthag": "qual_barthag",
}

# Conference abbreviations that appear as team names in some years.
# These are conference summary rows, not actual teams.
_CONF_NAMES = {
    "A10", "ACC", "AE", "ASun", "Amer", "B10", "B12", "BE", "BSky", "BSth",
    "BW", "CAA", "CUSA", "Horz", "Ivy", "MAAC", "MAC", "MEAC", "MVC", "MWC",
    "NEC", "OVC", "P10", "P12", "Pat", "SB", "SC", "SEC", "SWAC", "Slnd",
    "Sum", "WAC", "WCC", "GWC", "Ind", "ind",
}

# Manual overrides for Barttorvik names that don't match any spelling variant.
# Barttorvik name (lowercase) -> TeamID
_MANUAL_OVERRIDES_M = {
    "arkansas pine bluff": 1115,
    "bethune cookman": 1126,
    "illinois chicago": 1227,
    "louisiana monroe": 1419,
    "queens": 1474,
    "saint francis": 1384,  # Saint Francis PA (NEC)
    "southeast missouri st.": 1369,
    "tarleton st.": 1470,
    "tennessee martin": 1404,
    "texas a&m corpus chris": 1394,
    "ut rio grande valley": 1410,
    "winston salem st.": 1445,
}

_MANUAL_OVERRIDES_W = {
    "arkansas pine bluff": 3115,
    "bethune cookman": 3126,
    "illinois chicago": 3227,
    "louisiana monroe": 3419,
    "queens": 3474,
    "saint francis": 3384,  # Saint Francis PA (NEC)
    "southeast missouri st.": 3369,
    "tarleton st.": 3470,
    "tennessee martin": 3404,
    "texas a&m corpus chris": 3394,
    "ut rio grande valley": 3410,
    "winston salem st.": 3445,
}


def _load_spellings(gender: str) -> dict[str, int]:
    """Load team spellings file and build lowercase name -> TeamID lookup."""
    prefix = "M" if gender == "M" else "W"
    path = DATA_DIR / f"{prefix}TeamSpellings.csv"
    df = pd.read_csv(path, encoding="latin-1")
    lookup = {}
    for _, row in df.iterrows():
        key = str(row["TeamNameSpelling"]).strip().lower()
        lookup[key] = int(row["TeamID"])
    return lookup


def build_name_mapping(gender: str) -> dict[str, int]:
    """Build Barttorvik team name -> Kaggle TeamID mapping.

    Strategy:
    1. Lowercase direct lookup in spellings file
    2. Try removing periods (e.g. "St." -> "St")
    3. Try replacing "St." with "State"
    4. Fall back to manual overrides

    Args:
        gender: "M" or "W"

    Returns:
        Dict mapping lowercase Barttorvik team name to TeamID.
    """
    spell_lookup = _load_spellings(gender)
    overrides = _MANUAL_OVERRIDES_M if gender == "M" else _MANUAL_OVERRIDES_W

    # Collect all unique Barttorvik team names for this gender.
    # Must apply the same column fix as load_barttorvik_ratings() for
    # pre-2023 files (44 cols, shifted alignment).
    all_names = set()
    if gender == "M":
        year_range = range(2008, 2027)
        file_pattern = "barttorvik_ratings_{}.csv"
    else:
        year_range = range(2021, 2027)
        file_pattern = "barttorvik_w_ratings_{}.csv"

    for yr in year_range:
        path = BARTTORVIK_DIR / file_pattern.format(yr)
        if path.exists():
            df = pd.read_csv(path)
            if len(df.columns) == 44:
                df = pd.read_csv(path, names=_COLS_44, header=0)
            all_names.update(df["team"].unique())

    # Filter out conference summary rows
    team_names = all_names - _CONF_NAMES

    mapping = {}
    unmatched = []

    for name in sorted(team_names):
        key = name.strip().lower()

        # Strategy 1: direct lookup
        if key in spell_lookup:
            mapping[key] = spell_lookup[key]
            continue

        # Strategy 2: remove periods
        no_dots = key.replace(".", "")
        if no_dots in spell_lookup:
            mapping[key] = spell_lookup[no_dots]
            continue

        # Strategy 3: "St." -> "State"
        st_expanded = key.replace("st.", "state")
        if st_expanded in spell_lookup:
            mapping[key] = spell_lookup[st_expanded]
            continue

        # Strategy 4: manual overrides
        if key in overrides:
            mapping[key] = overrides[key]
            continue

        unmatched.append(name)

    return mapping, unmatched


# The correct 45-column header (from 2023+ files).
# Pre-2023 files have 44 columns: the numeric 'rank' column is absent and
# everything is shifted left by 1.  Reassigning header to _COLS_45[1:]
# fixes the alignment (the old 'Fun.Rk..adjt' column actually contains adjt).
_COLS_45 = [
    "rank", "team", "conf", "record", "adjoe", "oe.Rank", "adjde", "de.Rank",
    "barthag", "rank.1", "proj..W", "Proj..L", "Pro.Con.W", "Pro.Con.L",
    "Con.Rec.", "sos", "ncsos", "consos", "Proj..SOS", "Proj..Noncon.SOS",
    "Proj..Con.SOS", "elite.SOS", "elite.noncon.SOS", "Opp.OE", "Opp.DE",
    "Opp.Proj..OE", "Opp.Proj.DE", "Con.Adj.OE", "Con.Adj.DE", "Qual.O",
    "Qual.D", "Qual.Barthag", "Qual.Games", "FUN", "ConPF", "ConPA",
    "ConPoss", "ConOE", "ConDE", "ConSOSRemain", "Conf.Win.", "WAB",
    "WAB.Rk", "Fun.Rk", "adjt",
]
_COLS_44 = _COLS_45[1:]  # Same but without leading 'rank'


def load_barttorvik_ratings(gender: str) -> pd.DataFrame:
    """Load all Barttorvik ratings for a gender, mapped to Kaggle TeamIDs.

    Reads all yearly CSVs, fixes column alignment for pre-2023 files,
    filters conference summary rows, maps team names to TeamIDs, renames
    columns to snake_case, and validates the result.

    Args:
        gender: "M" or "W"

    Returns:
        DataFrame with columns: Season, TeamID, adjoe, adjde, barthag,
        adjt, wab, elite_sos, qual_o, qual_d, qual_barthag
    """
    mapping, unmatched = build_name_mapping(gender)
    if unmatched:
        raise ValueError(f"Unmatched Barttorvik teams: {unmatched}")

    frames = []
    if gender == "M":
        year_range = range(2008, 2027)
        file_pattern = "barttorvik_ratings_{}.csv"
    else:
        year_range = range(2021, 2027)
        file_pattern = "barttorvik_w_ratings_{}.csv"

    for yr in year_range:
        path = BARTTORVIK_DIR / file_pattern.format(yr)
        if not path.exists():
            continue

        # Detect column count and fix alignment for pre-2023 files.
        # Pre-2023: 44 cols, numeric rank is absent, columns shifted left by 1.
        # 2023+: 45 cols, proper rank column present.
        df = pd.read_csv(path)
        if len(df.columns) == 44:
            df = pd.read_csv(path, names=_COLS_44, header=0)
        # else: 45-col files parse correctly as-is

        # Filter out conference summary rows
        df = df[~df["team"].isin(_CONF_NAMES)].copy()

        # Map team name to TeamID
        df["TeamID"] = df["team"].str.strip().str.lower().map(mapping)

        # Flag any teams that didn't map (shouldn't happen after validation)
        unmapped = df[df["TeamID"].isna()]["team"].unique()
        if len(unmapped) > 0:
            raise ValueError(f"Unmapped teams in {path.name}: {list(unmapped)}")

        df["TeamID"] = df["TeamID"].astype("int32")
        df["Season"] = np.int16(yr)

        # Select and rename only the columns we need
        cols_to_keep = ["Season", "TeamID"] + list(_COLUMN_RENAME.keys())
        df = df[cols_to_keep].rename(columns=_COLUMN_RENAME)

        frames.append(df)

    result = pd.concat(frames, ignore_index=True)

    # Validate no duplicate (Season, TeamID) rows
    dupes = result.duplicated(subset=["Season", "TeamID"], keep=False)
    if dupes.any():
        dupe_rows = result[dupes][["Season", "TeamID"]].drop_duplicates()
        raise ValueError(f"Duplicate (Season, TeamID) rows:\n{dupe_rows}")

    # Ensure feature columns are float64
    for col in BARTTORVIK_FEATURES:
        result[col] = result[col].astype("float64")

    return result


def get_unmatched_report(gender: str) -> str:
    """Return a human-readable report of unmatched teams."""
    mapping, unmatched = build_name_mapping(gender)
    lines = [
        f"{'M' if gender == 'M' else 'W'} Barttorvik Name Mapping Report",
        f"  Matched: {len(mapping)} teams",
        f"  Unmatched: {len(unmatched)} teams",
    ]
    if unmatched:
        lines.append("  Unmatched names:")
        for name in unmatched:
            lines.append(f"    - {name}")
    return "\n".join(lines)
