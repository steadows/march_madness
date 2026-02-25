# Skill: Data Engineering

## When to Use
Reference this skill when loading data, building features, or transforming DataFrames.

## Data Loading Patterns

### Caching
All data loading functions should cache results to avoid re-reading CSVs:
```python
_cache = {}

def load_teams(gender='M'):
    key = f'teams_{gender}'
    if key not in _cache:
        prefix = gender
        _cache[key] = pd.read_csv(config.DATA_DIR / f'{prefix}Teams.csv')
    return _cache[key]
```

### Gender Parameter
Use `gender='M'` (default) or `gender='W'`. Map to file prefix:
```python
# M → MRegularSeasonCompactResults.csv
# W → WRegularSeasonCompactResults.csv
```

### Large Files
`MMasseyOrdinals.csv` is 5.8M rows. Load efficiently:
```python
# Specify dtypes to reduce memory
dtypes = {'Season': 'int16', 'RankingDayNum': 'int16', 'SystemName': 'category', 'TeamID': 'int16', 'OrdinalRank': 'int16'}
df = pd.read_csv(path, dtype=dtypes)
```

## Feature Engineering Patterns

### Per-Possession Stats
Raw stats are misleading because teams play at different tempos. Always normalize:
```python
possessions = FGA - OR + TO + 0.44 * FTA
off_efficiency = Score / possessions * 100  # points per 100 possessions
```

### Differential Features
ALL matchup features must be computed as differentials:
```python
# TeamA = lower ID (matches submission format)
feature_diff = team_a_stat - team_b_stat
```
This means: positive diff = TeamA is better on that metric.

### Seed Parsing
Seeds are strings like "W01", "X16a", "Z11b". Extract numeric seed:
```python
def parse_seed(seed_str: str) -> int:
    """Extract numeric seed from string like 'W01' or 'X16a'."""
    return int(seed_str[1:3])
```
The letter prefix is the region (W, X, Y, Z). Trailing letters (a, b) indicate play-in games.

### Temporal Features
When computing season stats, respect time ordering:
- Regular season stats: aggregate all games up to day 132 (before tournament)
- "Recent form": stats from last 14 days of regular season only
- NEVER include tournament games in features used to predict tournament games

### Handling Missing Data
- Detailed stats only available since 2003 (M) and 2010 (W)
- Some teams have few games in a season — use minimum game threshold (e.g., 10 games)
- Massey Ordinals: not all systems rank all teams. Use NaN for missing, then handle in model.
- For GBMs: NaN is handled natively. For logistic regression: impute with median.

## Key Datasets Hierarchy (by predictive value)

1. **MMasseyOrdinals** — Rankings from 60+ systems. POM, SAG, MOR are most valuable.
2. **MNCAATourneySeeds** — Seed differential is a top-3 feature every year.
3. **MRegularSeasonDetailedResults** — Box scores for computing efficiency metrics.
4. **MRegularSeasonCompactResults** — For Elo ratings and basic win/loss records.
5. **MTeamConferences / Conferences** — Conference strength features.
6. **MTeamCoaches** — Coach tournament experience.
7. **Cities / MGameCities** — Travel distance (minor impact).
