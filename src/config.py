"""Project-wide constants, paths, and configuration."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# Season boundaries
FIRST_COMPACT_SEASON_M = 1985
FIRST_COMPACT_SEASON_W = 1998
FIRST_DETAILED_SEASON_M = 2003
FIRST_DETAILED_SEASON_W = 2010
CURRENT_SEASON = 2026
TOURNEY_START_DAY = 134  # approximate first day of NCAA tournament

# Team ID ranges (no overlap between genders)
MEN_ID_RANGE = (1100, 1999)
WOMEN_ID_RANGE = (3100, 3999)

# Massey systems to prioritize (by predictive value)
TOP_MASSEY_SYSTEMS = ['POM', 'SAG', 'MOR', 'WOL', 'DOL', 'COL', 'RPI', 'AP', 'USA']

# Prediction clipping bounds — mandatory for valid submission
CLIP_LOW = 0.05
CLIP_HIGH = 0.95

# Gender file prefix mapping
GENDER_PREFIX = {'M': 'M', 'W': 'W'}
