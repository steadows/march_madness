# Skill: External Data Sources

## When to Use
Load this skill during Phase 5 when looking to add external data for model improvement.

## Rules
- External data MUST be publicly available and free (competition rules)
- Always document the source in CLAUDE.md Key Decisions
- Check if the data is already available in Massey Ordinals before fetching externally
- Test that external data actually improves Brier Score via CV before committing to it

## Available in Massey Ordinals (Already Have)
These systems in MMasseyOrdinals.csv are proxies for paid/premium sources:
- **POM** — KenPom efficiency ratings (gold standard)
- **SAG** — Sagarin ratings
- **MOR** — Jeff Massey's own ratings
- **WOL** — Wolfe ratings
- **DOL** — Dolphin ratings
- **COL** — Colley Matrix
- **RPI** — Rating Percentage Index (old NCAA metric)
- **AP** — Associated Press poll
- **USA** — USA Today coaches poll

**Check what's available before fetching anything external.**

## BartTorvik / T-Rank (Free)
- URL: https://barttorvik.com
- Has: adjusted efficiency, tempo, four factors, player-level data
- Access: `toRvik` R package or scrape CSV exports
- Python approach:
```python
# BartTorvik makes CSV downloads available
# Check: https://barttorvik.com/trank.php?year=2026&conyes=1&csv=1
import pandas as pd
url = "https://barttorvik.com/trank.php?year=2026&conyes=1&csv=1"
torvik_df = pd.read_csv(url)
```

## KenPom (Paid but Proxied)
- Already available as POM in Massey Ordinals (rank only, not raw efficiency)
- Full data ($25/year): kenpom.com — has raw adjusted efficiency margins
- Only worth buying if POM ranks aren't sufficient

## ESPN BPI
- Available at ESPN website, changes daily
- Harder to programmatically access
- Lower priority than Massey-available systems

## Vegas Lines / Spreads
- Strong predictive signal for CURRENT season games
- Sources: covers.com, vegasinsider.com, oddsshark.com
- Challenge: historical lines are harder to find for free
- If available, the pre-tournament line for each team (e.g., "20-1 to win championship") is very informative

## Integration Pattern
```python
def merge_external_data(features_df, external_df, on=['Season', 'TeamID']):
    """Merge external features, handling missing data gracefully."""
    merged = features_df.merge(external_df, on=on, how='left')
    # External data will have NaN for teams/seasons not covered
    # GBMs handle NaN natively — no need to impute for XGBoost/LightGBM
    return merged
```

## Priority Order (by expected Brier improvement)
1. More Massey systems (already have the data — just use more of them)
2. BartTorvik efficiency stats (free, easy to fetch)
3. Vegas lines for current season (if you can find free historical data)
4. Everything else is marginal
