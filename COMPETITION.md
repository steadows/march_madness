# March Machine Learning Mania 2026

> **Kaggle Competition**: [Overview](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/overview) | [Data](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data)
>
> **Objective**: Predict the probability that Team A beats Team B for every possible matchup in both the Men's and Women's 2026 NCAA basketball tournaments.

---

## Competition Details

### Evaluation Metric: Brier Score (MSE)

**This is NOT Log Loss** (which was used in older editions). The 2026 competition uses **Brier Score**, calculated as the mean of squared differences between predicted probabilities and actual outcomes (0 or 1). **Lower is better.**

```
Brier Score = (1/N) * SUM( (predicted_prob - actual_outcome)^2 )
```

Key implication: Brier Score is more forgiving of confident wrong predictions than Log Loss. A prediction of 0.99 for a loss costs `(0.99 - 0)^2 = 0.98` under Brier vs `-ln(0.01) = 4.6` under Log Loss. Still, calibration matters.

### Submission Format

CSV with two columns: `ID,Pred`

- **ID**: `{Season}_{TeamID1}_{TeamID2}` — lower TeamID always first
- **Pred**: Probability that the lower-ID team wins
- Covers **every possible team pair** (not just tournament-selected teams)
- Men's and Women's teams are in a **single unified file** (TeamIDs don't overlap: M=1xxx, W=3xxx)

Example:
```csv
ID,Pred
2026_1101_1102,0.5
2026_1101_1103,0.65
2026_3101_3102,0.5
```

- **Stage 1 sample**: ~519K rows (historical seasons 2022-2025, all team pairs)
- **Stage 2 sample**: ~132K rows (2026 season, all team pairs)

### Timeline

| Date | Event |
|------|-------|
| Feb 19, 2026 | Competition opens |
| **Mar 19, 2026, 4:00 PM UTC** | **Final submission deadline** |
| Mar 19 – Apr 6, 2026 | Tournament plays out; leaderboard updates as results come in |

- Pre-tournament leaderboard reflects scores from 2021-2025 historical data only
- Once 2026 games begin, Kaggle rescores against actual results
- Kaggle will release updated data at least once before the deadline

### Prizes — $50,000 Total

| Place | Prize |
|-------|-------|
| 1st | $10,000 |
| 2nd | $8,000 |
| 3rd | $7,000 |
| 4th–8th | $5,000 each |

### Rules

- **Team size**: Max 5 members
- **Submissions**: Max 5/day; must manually select **2 final submissions** for scoring
- **External data**: Allowed if publicly available and free (or reasonably priced)
- **Automated ML tools**: Allowed with appropriate license
- **Age**: Must be 18+
- Winners must provide reproducible code + methodology write-up

---

## Dataset Reference

All files prefixed `M` = Men's, `W` = Women's. Structure is identical between the two.

### Core Game Results

| File | Rows | Description |
|------|------|-------------|
| `MRegularSeasonCompactResults.csv` | ~160K | Every regular season game since 1985. Columns: `Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, NumOT` |
| `MRegularSeasonDetailedResults.csv` | ~100K | Same but with box score stats (since 2003). Adds: `WFGM, WFGA, WFGM3, WFGA3, WFTM, WFTA, WOR, WDR, WAst, WTO, WStl, WBlk, WPF` (and same for L) |
| `MNCAATourneyCompactResults.csv` | ~2.4K | Every NCAA tournament game since 1985 (compact) |
| `MNCAATourneyDetailedResults.csv` | ~1.2K | NCAA tournament games with box scores (since 2003) |

**Box score stat columns** (detailed results):
- `FGM/FGA` — Field goals made/attempted
- `FGM3/FGA3` — 3-pointers made/attempted
- `FTM/FTA` — Free throws made/attempted
- `OR/DR` — Offensive/defensive rebounds
- `Ast` — Assists
- `TO` — Turnovers
- `Stl` — Steals
- `Blk` — Blocks
- `PF` — Personal fouls

### Tournament Structure

| File | Description |
|------|-------------|
| `MNCAATourneySeeds.csv` | Seed assignments per team per season. Format: `W01`, `X16a` (region + seed + optional play-in indicator) |
| `MNCAATourneySlots.csv` | Bracket structure mapping. Columns: `Season, Slot, StrongSeed, WeakSeed` |
| `MNCAATourneySeedRoundSlots.csv` | Maps seeds to game rounds and day ranges |

### Team & Conference Info

| File | Description |
|------|-------------|
| `MTeams.csv` | 381 teams. Columns: `TeamID, TeamName, FirstD1Season, LastD1Season` |
| `MTeamConferences.csv` | Conference membership per team per season |
| `Conferences.csv` | Conference abbreviation → full name lookup |
| `MTeamCoaches.csv` | Coach assignments per team per season (with day ranges for mid-season changes) |
| `MTeamSpellings.csv` | Alternate team name spellings → TeamID mapping |

### Ratings & Rankings

| File | Rows | Description |
|------|------|-------------|
| `MMasseyOrdinals.csv` | **~5.8M** | Rankings from dozens of computer rating systems. Columns: `Season, RankingDayNum, SystemName, TeamID, OrdinalRank`. Updated throughout each season. Available since 2003. |

This is a **massive** and extremely valuable dataset. Contains systems like POM (KenPom proxy), SAG (Sagarin), MOR, WOL, DOL, and many more.

### Geography

| File | Description |
|------|-------------|
| `Cities.csv` | City ID → City, State lookup |
| `MGameCities.csv` | Maps games to cities (regular season + tournament). Useful for travel distance features. |

### Secondary Tournaments

| File | Description |
|------|-------------|
| `MSecondaryTourneyCompactResults.csv` | Results from NIT, CBI, CIT, and other non-NCAA tournaments |
| `MSecondaryTourneyTeams.csv` | Teams selected for secondary tournaments |
| `MConferenceTourneyGames.csv` | Conference tournament game results |

### Seasons

| File | Description |
|------|-------------|
| `MSeasons.csv` | Season metadata: `Season, DayZero, RegionW, RegionX, RegionY, RegionZ`. DayZero is the reference date for DayNum. |

---

## Best Architectures & Methods

### What Wins This Competition

Based on analysis of past winners and top solutions:

**Gradient Boosted Trees dominate.** XGBoost, LightGBM, and CatBoost are the backbone of nearly every winning solution. They handle tabular data well, are robust to feature scale differences, and produce well-calibrated probabilities with proper tuning.

**Ensembles of diverse models** are essential for top finishes. Winners typically stack 20-70+ models including:
- Multiple GBM variants with different hyperparameters
- Elo/rating-based models
- Logistic regression baselines
- Neural networks for diversity

### Tier 1: Must-Have Approaches

#### 1. Gradient Boosted Trees (XGBoost / LightGBM / CatBoost)
- Primary workhorse model
- Train on team-pair features (differentials between team stats)
- Target: probability of lower-ID team winning
- Typical performance: 0.40-0.45 log loss range (historical metric)

#### 2. Custom Elo Rating System
- Build from scratch using game-by-game results
- Key tuning parameters: K-factor, home court advantage, margin of victory scaling
- Converts Elo differential to win probability via logistic function
- Strong standalone model AND critical feature for GBMs

#### 3. Massey Ordinals Integration
- **Single most valuable dataset** in the competition data
- Key systems to prioritize: **POM** (KenPom proxy), **SAG** (Sagarin), **MOR**, **WOL**, **DOL**
- Use ranks from the latest available day before the tournament
- Rank differentials between team pairs are extremely predictive

#### 4. Seed-Based Features
- Seed differential is one of the top predictive features
- Historical win rates by seed matchup (e.g., 1-seed vs 16-seed)
- Parse seed strings to extract region and numeric seed

### Tier 2: Strong Additions

#### 5. Efficiency Metrics (KenPom-Style "Four Factors")
Compute per-possession stats from the detailed box scores:
- **Effective FG%**: `(FGM + 0.5 * FGM3) / FGA`
- **Turnover Rate**: `TO / (FGA + 0.44 * FTA + TO)`
- **Offensive Rebound %**: `OR / (OR + opponent DR)`
- **Free Throw Rate**: `FTM / FGA`
- Calculate for both offense and defense, per-possession adjusted

#### 6. Strength of Schedule
- Average opponent rating/rank
- Record against tournament-quality teams
- Conference strength aggregation

#### 7. Momentum / Recent Form
- Weight recent games more heavily than early-season games
- Performance in last 10-14 games
- Conference tournament results

### Tier 3: Edge-Case Improvements

#### 8. Neural Networks
- Embedding-based models that learn team representations
- LSTMs/Transformers over game sequences (can capture momentum)
- Best used as diversity in ensembles, not standalone

#### 9. Travel Distance
- Using Cities data to compute distance between team location and game location
- Minor but measurable impact

#### 10. Coach Experience
- Tournament appearances, wins for the coach
- Years at current school

### Key Technical Strategies

#### Probability Calibration
- **Critical for Brier Score**: Clip predictions to [0.05, 0.95] as a safety floor/ceiling
- Use Platt scaling or isotonic regression for post-hoc calibration
- Brier Score is less punishing than Log Loss for extreme predictions, but calibration still matters

#### Cross-Validation Strategy
- **Expanding window CV by season** (walk-forward): Train on seasons 1-N, validate on season N+1
- Never leak future season data into training
- Typical: train on 2003-2023, validate on 2024-2025, predict 2026

#### Men's vs Women's
- **Train separate models** for men's and women's tournaments
- Different dynamics, different data availability (women's data starts later)
- Combine predictions into single submission file

#### Feature Engineering Pattern
All features should be computed as **team-pair differentials**:
```
feature_diff = TeamA_stat - TeamB_stat
```
Where TeamA = lower ID team, TeamB = higher ID team (matching submission format).

#### Common Pitfalls
- **Not clipping probabilities** — even one 0.0 or 1.0 prediction can be devastating
- **Data leakage** — accidentally including current-season tournament results in training features
- **Overfitting to Stage 1** — the historical leaderboard doesn't predict tournament performance
- **Ignoring women's data** — it's half the submission; don't treat it as an afterthought
- **Using raw stats instead of per-possession** — tempo-free metrics are far more predictive

### Recommended External Data Sources

| Source | Access | Value |
|--------|--------|-------|
| KenPom (kenpom.com) | Paid (~$25/yr) but proxied via POM in Massey Ordinals | Gold standard for efficiency metrics |
| BartTorvik / T-Rank | Free (barttorvik.com) | Excellent advanced stats, `toRvik` R package available |
| Sagarin ratings | Free | Long historical record, available as SAG in Massey Ordinals |
| ESPN BPI | Free | Basketball Power Index |
| NET Rankings | Free (NCAA official) | Used for tournament selection |
| Vegas lines / spreads | Various | Strong predictive signal for current season |
| FiveThirtyEight / Silver Bulletin | Free | Composite Elo-based ratings |

### Recommended Model Stack

```
Level 0 (Base Models):
├── XGBoost (2-3 configs)
├── LightGBM (2-3 configs)
├── CatBoost (1-2 configs)
├── Custom Elo model
├── Logistic Regression (seed + Massey features)
├── Ridge Regression
└── Optional: Small neural net with team embeddings

Level 1 (Meta-Learner):
├── Logistic Regression or Ridge on Level 0 outputs
└── Or simple weighted average with optimized weights

Final:
└── Clip to [0.05, 0.95], calibrate
```

---

## Quick Start Checklist

- [ ] EDA: Explore all datasets, understand distributions and time ranges
- [ ] Build baseline: Seed-based logistic regression
- [ ] Build Elo system from compact results
- [ ] Extract top Massey Ordinal systems and create rank differential features
- [ ] Compute efficiency metrics from detailed box scores
- [ ] Engineer full feature set (differentials for all team-pair features)
- [ ] Train GBM models with season-based expanding window CV
- [ ] Train separate M and W models
- [ ] Build ensemble / stacking pipeline
- [ ] Calibrate probabilities and clip
- [ ] Generate submission file in correct format
- [ ] Validate submission against sample file structure
