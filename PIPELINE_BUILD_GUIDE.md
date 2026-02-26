# March Madness ML Pipeline — Agentic Build Guide

> **GSD: Get Shit Done. Every section has checkboxes. Check them off. Move on.**

---

## Scope: What You Build vs. What the Human Does

```
┌──────────────────────────────────────────────────────────────┐
│  AGENT BUILDS (Phases 0-3):                                  │
│  ✓ Project scaffolding & environment                         │
│  ✓ Data loading layer with validation                        │
│  ✓ Feature engineering pipeline                              │
│  ✓ Elo system, Massey processing                             │
│  ✓ Cross-validation framework                                │
│  ✓ Data quality tests (comprehensive)                        │
│  ✓ Feature matrix export (clean, validated, ready to train)  │
│  ✓ Submission generation scaffolding                         │
│  ✓ One baseline model to prove the pipeline works end-to-end │
├──────────────────────────────────────────────────────────────┤
│  HUMAN DOES (Phase 4+):                                      │
│  → Model training & experimentation                          │
│  → Hyperparameter tuning                                     │
│  → Ensemble optimization                                     │
│  → Final submission selection                                │
└──────────────────────────────────────────────────────────────┘
```

**Your #1 job: deliver CLEAN, VALIDATED data that the human can trust blindly when training models. Zero surprises. Zero data bugs.**

---

## Operating Rules

1. Read `CLAUDE.md` first. Always. It's your memory.
2. Run `bash skills.sh` to see available skills. Load ONLY what you need for current phase.
3. One phase per session. If context gets long (30+ tool calls), checkpoint and stop.
4. Write code to disk. Don't draft in chat. Don't re-read files you just wrote.
5. Redirect verbose output: `cmd > artifacts/log.txt 2>&1 && tail -20 artifacts/log.txt`
6. Every phase has a **GSD Checklist**. Every box must be checked before moving on.
7. Update `CLAUDE.md` at end of every phase. This is non-optional.

## Conda Environment — ALWAYS USE THIS

```bash
# Python binary (use this for ALL python commands):
/opt/anaconda3/envs/march_madness/bin/python

# Pip (if you need to install anything):
/opt/anaconda3/envs/march_madness/bin/pip install <package>

# Pytest:
/opt/anaconda3/envs/march_madness/bin/python -m pytest tests/ -q

# NEVER use bare `python3` or `pip` — those hit the base env, not ours.
```

---

## Project Structure

```
march_madness/
├── .claude/
│   ├── settings.local.json
│   └── skills/                  # Agent skill files (9 skills)
├── data/                        # Raw competition CSVs (35 files, DO NOT MODIFY)
├── src/
│   ├── __init__.py
│   ├── config.py                # Paths, constants, feature lists
│   ├── data_loader.py           # Load all competition CSVs
│   ├── data_validator.py        # Data quality checks and assertions
│   ├── elo.py                   # Custom Elo rating system
│   ├── massey.py                # Massey Ordinals processing
│   ├── feature_engineering.py   # Full feature computation pipeline
│   ├── cv.py                    # Cross-validation framework
│   ├── models.py                # Model training (baseline only, human extends)
│   ├── calibration.py           # Probability calibration & clipping
│   ├── submission.py            # Submission file generation
│   └── utils.py                 # Shared utilities
├── tests/
│   ├── test_data_loader.py      # Data loading tests
│   ├── test_data_quality.py     # Comprehensive data quality suite
│   ├── test_elo.py              # Elo system tests
│   ├── test_features.py         # Feature engineering tests
│   ├── test_pipeline.py         # End-to-end pipeline integration test
│   └── test_submission.py       # Submission format validation
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   └── 02_feature_analysis.ipynb # Feature distributions and correlations
├── artifacts/
│   ├── features_men.csv         # Final clean feature matrix (M)
│   ├── features_women.csv       # Final clean feature matrix (W)
│   ├── feature_columns.json     # Feature column names and descriptions
│   ├── elo_ratings.json         # Precomputed Elo ratings
│   └── data_quality_report.txt  # Quality report from validator
├── submissions/
├── CLAUDE.md                    # Agent operating manual (auto-loaded)
├── COMPETITION.md               # Competition details & methods research
├── PIPELINE_BUILD_GUIDE.md      # This file
├── skills.sh                    # Skill index and loader
└── requirements.txt
```

---

## Phase 0: Environment Setup

**Session budget: ~15 tool calls**
**Load skill:** `bash skills.sh project-conventions`

### GSD Checklist
- [ ] Create directories: `src/`, `tests/`, `notebooks/`, `artifacts/`, `submissions/`
- [ ] Create `requirements.txt` (see below)
- [ ] Run `pip install -r requirements.txt` — all installs succeed
- [ ] Create `src/__init__.py` (empty)
- [ ] Create `src/config.py` with all constants (see below)
- [ ] Verify: `python3 -c "import pandas, numpy, sklearn, xgboost, lightgbm, catboost; print('OK')"`
- [ ] Verify: `python3 -c "from src import config; print(config.DATA_DIR)"` resolves
- [ ] Verify: `ls data/*.csv | wc -l` returns 35
- [ ] Init git repo if not already: `git init && git add -A && git commit -m "phase 0: project setup"`
- [ ] Update `CLAUDE.md`: mark Phase 0 complete, list installed packages

### requirements.txt
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
catboost>=1.2
scipy>=1.11
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
pytest>=7.0
optuna>=3.0
```

### src/config.py Must Include
```python
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
TOURNEY_START_DAY = 134  # approximate

# Team ID ranges (no overlap)
MEN_ID_RANGE = (1100, 1999)
WOMEN_ID_RANGE = (3100, 3999)

# Massey systems to prioritize (by predictive value)
TOP_MASSEY_SYSTEMS = ['POM', 'SAG', 'MOR', 'WOL', 'DOL', 'COL', 'RPI', 'AP', 'USA']

# Prediction clipping bounds
CLIP_LOW = 0.05
CLIP_HIGH = 0.95
```

---

## Phase 1: Data Loading & Validation

**Session budget: ~35 tool calls**
**Load skill:** `bash skills.sh data-engineering`

> **This phase is about TRUST. When it's done, every DataFrame the pipeline touches has been loaded, typed, validated, and tested. No silent errors.**

### GSD Checklist — Data Loader
- [ ] Create `src/data_loader.py` with ALL load functions (see list below)
- [ ] Every function: caches results, accepts `gender` param, returns typed DataFrame
- [ ] Massey ordinals loaded with memory-efficient dtypes
- [ ] Create `tests/test_data_loader.py`
- [ ] Test: every load function returns DataFrame with expected columns
- [ ] Test: row counts within expected ranges (not empty, not absurdly large)
- [ ] Test: key columns (Season, TeamID) have no NaN
- [ ] Test: seed strings parse correctly ("W01"→1, "X16a"→16, "Z11b"→11)
- [ ] `pytest tests/test_data_loader.py -q` — ALL PASS

### Load Functions Required
```
load_teams(gender='M')                    → TeamID, TeamName, FirstD1Season, LastD1Season
load_regular_season(gender='M', detailed=False) → game results (compact or detailed)
load_tourney_results(gender='M', detailed=False) → tournament results
load_tourney_seeds(gender='M')            → Season, Seed, TeamID
load_tourney_slots(gender='M')            → Season, Slot, StrongSeed, WeakSeed
load_massey_ordinals()                    → Season, RankingDayNum, SystemName, TeamID, OrdinalRank
load_conferences(gender='M')             → Season, TeamID, ConfAbbrev
load_coaches()                           → Season, TeamID, FirstDayNum, LastDayNum, CoachName (M only)
load_cities()                            → CityID, City, State
load_game_cities(gender='M')             → Season, DayNum, game→city mapping
load_seasons(gender='M')                 → Season, DayZero, Regions
load_conference_tourney(gender='M')      → conference tourney results
load_secondary_tourney(gender='M')       → NIT/CBI results
load_sample_submission(stage=1)          → ID, Pred
```

### GSD Checklist — Data Quality Validator
- [ ] Create `src/data_validator.py` — standalone validation module
- [ ] Create `tests/test_data_quality.py` — comprehensive data quality test suite
- [ ] **Test suite covers ALL of the following:**

#### Referential Integrity Tests
- [ ] Every TeamID in game results exists in MTeams/WTeams
- [ ] Every TeamID in tourney seeds exists in MTeams/WTeams
- [ ] Every TeamID in conferences exists in MTeams/WTeams
- [ ] Every conference abbreviation in team_conferences exists in Conferences.csv
- [ ] Every CityID in game_cities exists in Cities.csv

#### Temporal Consistency Tests
- [ ] No game has DayNum < 0 or DayNum > 200
- [ ] Tournament games have DayNum >= 132 (approximately)
- [ ] Regular season games have DayNum < 134
- [ ] Massey ordinals RankingDayNum is within valid range per season
- [ ] No team plays a game before their FirstD1Season or after LastD1Season
- [ ] Coach date ranges don't overlap for the same team/season

#### Score & Stats Consistency Tests
- [ ] Winner score > Loser score in all game results (except NumOT edge cases — investigate)
- [ ] WScore and LScore are positive integers in realistic range (30-150)
- [ ] In detailed results: FGM <= FGA (can't make more than attempted)
- [ ] In detailed results: FGM3 <= FGA3
- [ ] In detailed results: FTM <= FTA
- [ ] In detailed results: FGM3 <= FGM (3-pointers are subset of all field goals)
- [ ] In detailed results: OR + DR > 0 (every game has rebounds)
- [ ] Score reconstructs from box score: `Score ≈ 2*(FGM-FGM3) + 3*FGM3 + FTM` (exact)

#### Seed Consistency Tests
- [ ] Seeds are in valid format: 1 letter (W/X/Y/Z) + 2 digits + optional letter (a/b)
- [ ] Numeric seeds range 1-16
- [ ] Each region in each season has exactly seeds 1-16 (with play-in adjustments)
- [ ] Every team in tourney results has a seed for that season

#### Completeness Tests
- [ ] No season in the expected range is missing from regular season results
- [ ] Every season with tourney results also has tourney seeds
- [ ] Detailed results available for seasons 2003+ (M) and 2010+ (W)
- [ ] Sample submission IDs are parseable and all TeamIDs exist

#### Cross-Gender Consistency Tests
- [ ] M and W team IDs don't overlap
- [ ] Sample submission contains both M (1xxx) and W (3xxx) team pairs
- [ ] M and W have same file structure (same column names per equivalent file)

### GSD Checklist — EDA
- [ ] Create `notebooks/01_eda.ipynb`
- [ ] Document: shape and season range of every dataset
- [ ] Document: score distributions (histogram), margin of victory
- [ ] Document: seed vs tourney win rate (1-seeds win X%, 16-seeds win Y%)
- [ ] Document: Massey system coverage (which systems have most seasons/teams)
- [ ] Document: missing data map (which seasons lack detailed stats)
- [ ] Document: M vs W differences (data start dates, number of teams, game counts)
- [ ] Save summary findings to `artifacts/data_quality_report.txt`

### GSD Checklist — Phase Wrap
- [ ] `pytest tests/test_data_loader.py tests/test_data_quality.py -q` — ALL PASS
- [ ] `artifacts/data_quality_report.txt` exists and documents findings
- [ ] `git add -A && git commit -m "phase 1: data loading and validation"`
- [ ] Update `CLAUDE.md`: mark Phase 1 complete, log any data issues found

---

## Phase 2: Feature Engineering

**Session budget: ~40 tool calls. If needed, split into 2a (Elo + Massey) and 2b (full features).**
**Load skill:** `bash skills.sh data-engineering`

> **This is the highest-value phase. Every feature must be conceptually correct, temporally valid (no leakage), and tested.**

### GSD Checklist — Elo System (2A)
- [ ] Create `src/elo.py`
- [ ] Elo initializes all teams at 1500
- [ ] Games processed chronologically within each season (sorted by DayNum)
- [ ] Home court advantage applied (WLoc: H/A/N)
- [ ] Margin of victory scaling (bigger wins = bigger Elo change)
- [ ] Season reset: regress to mean (0.75 * elo + 0.25 * 1500)
- [ ] Function: `compute_elo_ratings(games_df) -> {season: {team_id: elo}}`
- [ ] Function: `elo_to_win_prob(elo_diff) -> float` (logistic)
- [ ] Works for BOTH M and W data (pass gender-specific games)
- [ ] Create `tests/test_elo.py`
- [ ] Test: top teams (Duke, Kansas, UConn) have Elo > 1600 in recent seasons
- [ ] Test: bottom teams have Elo < 1400
- [ ] Test: mean Elo across all teams ≈ 1500 (conservation)
- [ ] Test: `elo_to_win_prob(0) ≈ 0.5`
- [ ] Test: `elo_to_win_prob(400) > 0.9`
- [ ] Save ratings: `json.dump(ratings, open('artifacts/elo_ratings.json', 'w'))`
- [ ] `pytest tests/test_elo.py -q` — ALL PASS

### GSD Checklist — Massey Ordinals (2A)
- [ ] Create `src/massey.py`
- [ ] Load efficiently with optimized dtypes (int16, category)
- [ ] Function: `get_available_systems(min_seasons=10) -> list[str]`
- [ ] Function: `get_team_rankings(season, team_id, day=133) -> dict`
  - Returns latest ranking BEFORE the given day for each system
- [ ] Function: `get_ranking_differential(season, team_a, team_b) -> dict`
  - Returns `{system_diff: rank_a - rank_b}` for each system
- [ ] Validate: rankings are positive integers (ordinal ranks, 1 = best)
- [ ] Validate: POM, SAG, MOR all present and have good coverage (2003+)
- [ ] Handle missing: return NaN for system/team combos that don't exist
- [ ] Test: 1-seed teams should have top-10 rankings in most systems
- [ ] `pytest` tests for massey pass

### GSD Checklist — Full Feature Pipeline (2B)
- [ ] Create `src/feature_engineering.py`

#### Per-Team Season Stats (from detailed results)
- [ ] `compute_team_season_stats(season, team_id, gender) -> dict`
- [ ] Win count, loss count, win percentage
- [ ] Points per game (off), points allowed per game (def)
- [ ] Possessions estimate: `FGA - OR + TO + 0.44 * FTA`
- [ ] Offensive efficiency: `Score / Poss * 100`
- [ ] Defensive efficiency: `OppScore / Poss * 100`
- [ ] Net efficiency: `Off - Def`
- [ ] Effective FG%: `(FGM + 0.5 * FGM3) / FGA`
- [ ] Turnover rate: `TO / (FGA + 0.44 * FTA + TO)`
- [ ] Offensive rebound %: `OR / (OR + Opp_DR)`
- [ ] Free throw rate: `FTM / FGA`
- [ ] 3-point rate: `FGA3 / FGA`
- [ ] Assist/turnover ratio
- [ ] Steal rate, block rate
- [ ] All computed for BOTH offense and defense side
- [ ] **ONLY uses games before tournament (DayNum < TOURNEY_START_DAY)**
- [ ] Falls back gracefully for teams with no detailed stats (returns NaN)

#### Per-Team Derived Stats
- [ ] Elo rating (from `src/elo.py`)
- [ ] Massey ordinal ranks (from `src/massey.py`)
- [ ] Seed (parsed numeric from seed string, NaN if not in tourney)
- [ ] Coach tournament experience (count of prior tourney appearances)
- [ ] Strength of schedule (mean Elo of opponents faced)
- [ ] Recent form: stats from last 14 DayNum days of regular season

#### Matchup Feature Builder
- [ ] `build_matchup_features(season, team_a_id, team_b_id) -> dict`
- [ ] team_a ALWAYS has lower ID (matching submission format)
- [ ] ALL features computed as differentials: `team_a_stat - team_b_stat`
- [ ] Returns flat dict ready for DataFrame row

#### Training Set Builder
- [ ] `build_training_set(seasons, gender) -> pd.DataFrame`
- [ ] For each tournament game in given seasons: build features
- [ ] Target: `1 if lower-ID team won, 0 if lower-ID team lost`
- [ ] Returns clean DataFrame: features + target column
- [ ] Drops rows where critical features are all NaN

#### Prediction Set Builder
- [ ] `build_prediction_set(sample_submission_path, gender) -> pd.DataFrame`
- [ ] Reads sample submission, filters to relevant gender
- [ ] Builds features for every matchup in submission
- [ ] Returns DataFrame aligned with submission IDs

### GSD Checklist — Feature Validation
- [ ] Create `tests/test_features.py`
- [ ] Test: differential symmetry — `feature(A,B) = -feature(B,A)` for all differential cols
- [ ] Test: seed differential for 1v16 matchup ≈ -15
- [ ] Test: no temporal leakage — features only use pre-tournament data
- [ ] Test: no features are constant (zero variance)
- [ ] Test: NaN rate per column < 20% (log any higher)
- [ ] Test: training set target is roughly balanced (45-55% ones)
- [ ] Test: feature count matches expected (log exact count)
- [ ] Test: feature matrix reproducible — running twice gives identical results

### GSD Checklist — Feature Matrix Export
- [ ] Export: `artifacts/features_men.csv` — full training set for M
- [ ] Export: `artifacts/features_women.csv` — full training set for W
- [ ] Export: `artifacts/feature_columns.json` — column names + descriptions
- [ ] Log: print feature matrix shapes, NaN summary, target distribution
- [ ] **These CSVs are the handoff to the human for model training**

### GSD Checklist — Quick Sanity Model
- [ ] Train ONE XGBoost on `features_men.csv` with default params, 1 CV fold
- [ ] Brier Score < 0.23 (proves pipeline is working)
- [ ] Print top 10 feature importances
- [ ] If seed_diff and elo_diff aren't in top 5: STOP, something is wrong
- [ ] Save importance chart to `artifacts/feature_importance.png`

### GSD Checklist — Phase Wrap
- [ ] `pytest tests/ -q` — ALL tests pass (data loader + quality + elo + features)
- [ ] Feature matrices exported to `artifacts/`
- [ ] `git add -A && git commit -m "phase 2: feature engineering complete"`
- [ ] Update `CLAUDE.md`:
  - Mark Phase 2 complete
  - Log all feature column names in Feature Columns section
  - Log XGBoost sanity Brier Score
  - Log any Key Decisions (which Massey systems used, etc.)

---

## Phase 3: CV Framework, Baseline Model & Submission Scaffolding

**Session budget: ~35 tool calls**
**Load skills:** `bash skills.sh model-training` and `bash skills.sh evaluation-validation`

> **Build just enough model infrastructure for the human to take over. One working baseline, one working submission generator, solid CV framework.**

### GSD Checklist — Cross-Validation
- [ ] Create `src/cv.py`
- [ ] `expanding_window_cv(df, season_col='Season', min_train_end=2019) -> list[tuple]`
- [ ] Folds: train up to N, validate on N+1 (2020, 2021, 2022, 2023, 2024)
- [ ] Never leaks future seasons into training
- [ ] Returns (train_indices, val_indices) tuples
- [ ] `evaluate_brier(y_true, y_pred) -> float` helper
- [ ] Test: fold count matches expected
- [ ] Test: no index overlap between train and val
- [ ] Test: val seasons are strictly after train seasons

### GSD Checklist — Baseline Model
- [ ] Create `src/models.py`
- [ ] `train_logistic_baseline(X_train, y_train) -> model` (seed_diff only)
- [ ] `train_xgboost(X_train, y_train, X_val, y_val, params=None) -> model`
- [ ] Both return sklearn-compatible `.predict_proba()` interface
- [ ] Run logistic baseline through full CV → log Brier per fold + mean
- [ ] Run XGBoost through full CV → log Brier per fold + mean
- [ ] **Logistic Brier < 0.25** (mandatory sanity check)
- [ ] **XGBoost Brier < 0.22** (proves features work)
- [ ] Save baseline models to `artifacts/`
- [ ] Log all scores to console AND to `artifacts/baseline_results.txt`

### GSD Checklist — Calibration
- [ ] Create `src/calibration.py`
- [ ] `clip_predictions(preds, low=0.05, high=0.95) -> np.array`
- [ ] `calibrate_platt(preds, y_true) -> np.array`
- [ ] `calibrate_isotonic(preds, y_true) -> np.array`
- [ ] `reliability_diagram(preds, y_true, save_path) -> None`
- [ ] Generate and save reliability diagram for XGBoost baseline

### GSD Checklist — Submission Generator
- [ ] Create `src/submission.py`
- [ ] `generate_submission(predict_fn, stage=1, output_path=None) -> pd.DataFrame`
- [ ] Reads correct sample submission (stage 1 or 2)
- [ ] Parses IDs → (season, team_a, team_b)
- [ ] Routes to M or W model based on TeamID range
- [ ] Clips all predictions to [0.05, 0.95]
- [ ] Writes CSV
- [ ] Create `tests/test_submission.py`
- [ ] Test: output row count matches sample submission exactly
- [ ] Test: output IDs match sample IDs in exact order
- [ ] Test: no NaN in Pred column
- [ ] Test: all Pred values in [0.05, 0.95]
- [ ] Test: Pred has reasonable spread (std > 0.05)
- [ ] Test: both M and W matchups present in output
- [ ] Generate a Stage 1 submission with XGBoost baseline: `submissions/baseline_v1.csv`

### GSD Checklist — Integration Test
- [ ] Create `tests/test_pipeline.py`
- [ ] Test: full pipeline end-to-end for a single season (e.g., 2024)
  - Load data → compute features → train model → predict → generate submission rows
- [ ] Test: pipeline produces valid predictions for both M and W
- [ ] Test: pipeline handles teams with missing data gracefully (NaN features, not crash)

### GSD Checklist — Phase Wrap
- [ ] `pytest tests/ -q` — ALL tests pass (every test file)
- [ ] `artifacts/baseline_results.txt` has Brier scores
- [ ] `submissions/baseline_v1.csv` is a valid submission file
- [ ] `git add -A && git commit -m "phase 3: baseline models and submission pipeline"`
- [ ] Update `CLAUDE.md`:
  - Mark Phase 3 complete
  - Log baseline Brier Scores in the score table
  - Update status: "Pipeline ready for human model training"

---

## Phase 3 Deliverables Summary — What the Human Gets

When Phase 3 is complete, the human should be able to:

```python
# 1. Load clean feature matrices
import pandas as pd
men_features = pd.read_csv('artifacts/features_men.csv')
women_features = pd.read_csv('artifacts/features_women.csv')

# 2. Use the CV framework
from src.cv import expanding_window_cv, evaluate_brier

# 3. Train any model they want
model = SomeModel()
for train_idx, val_idx in expanding_window_cv(men_features):
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    preds = model.predict_proba(X.iloc[val_idx])[:, 1]
    print(f"Brier: {evaluate_brier(y.iloc[val_idx], preds)}")

# 4. Generate a submission
from src.submission import generate_submission
generate_submission(predict_fn=my_predict, stage=2, output_path='submissions/my_sub.csv')
```

**The data pipelines are vetted. The features are validated. The tests prove it. The human just plugs in models.**

---

---

## Phase 4: Ensemble Pipeline

**Session budget: ~35 tool calls**
**Load skill:** `bash skills.sh ensemble-submission`

> **Goal: build a proper multi-model ensemble that beats the XGBoost-only baseline. Target: M Brier < 0.200 in CV.**

### Context: What Exists Already
Before writing any code, read these files to understand what's already built:
- `src/models.py` — `train_logistic_baseline()`, `train_xgboost()`, `run_cv_baseline()` already exist
- `src/cv.py` — `expanding_window_cv()`, `evaluate_brier()`, `cv_evaluate()` already exist
- `src/calibration.py` — `clip_predictions()`, `calibrate_platt()`, `reliability_diagram()` already exist
- `src/submission.py` — `generate_submission()`, `make_predict_fn()` already exist
- `artifacts/features_men.csv` — 2585×42 training matrix, ready to use
- `artifacts/features_women.csv` — 1717×42 training matrix, ready to use
- `submissions/baseline_v1.csv` — Stage 1 baseline submission already generated

**DO NOT rewrite existing code. Add to it.**

### Baseline Scores to Beat (from Phase 3 CV, 5 folds 2020-2024)
| Model | M Brier | W Brier |
|-------|---------|---------|
| Logistic (seed_diff only) | 0.213 | 0.153 |
| XGBoost (all features) | 0.212 | 0.157 |

---

### GSD Checklist — Model Wrappers

Add these functions to `src/models.py` (do NOT create a new file):

- [ ] `train_lightgbm(X_train, y_train, X_val=None, y_val=None, params=None) -> model`
  - Default params: `objective='binary'`, `metric='binary_logloss'`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_samples=20`, `reg_alpha=0.1`, `reg_lambda=1.0`, `n_estimators=500`, `verbose=-1`
  - Early stopping on val set if provided (`callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)]`)
  - Returns fitted LGBMClassifier with `.predict_proba()` interface

- [ ] `train_catboost(X_train, y_train, X_val=None, y_val=None, params=None) -> model`
  - Default params: `iterations=500`, `depth=6`, `learning_rate=0.05`, `l2_leaf_reg=3.0`, `eval_metric='Logloss'`, `verbose=0`, `random_state=42`
  - Early stopping on val set if provided
  - Returns fitted CatBoostClassifier

- [ ] `train_ridge(X_train, y_train) -> Pipeline`
  - `StandardScaler + Ridge(alpha=1.0)` wrapped in sklearn Pipeline
  - Use `predict_proba` via sigmoid transform: `1 / (1 + np.exp(-model.decision_function(X)))`
  - Actually: use `LogisticRegression(C=0.1, penalty='l2')` — same effect, native predict_proba
  - Returns Pipeline with `.predict_proba()` interface

- [ ] Verify: all three new wrappers produce probabilities in (0, 1)
- [ ] Verify: all three accept NaN features (XGBoost/LGB/CatBoost handle natively; Ridge needs `.fillna(median)` first)

---

### GSD Checklist — Ensemble Module

Create `src/ensemble.py`:

- [ ] `run_all_models_cv(df, feature_cols, target_col='target', gender='M') -> dict`
  - Runs ALL 5 models through `expanding_window_cv(df, min_train_end=2019)`
  - Models: logistic (seed_diff only), ridge (all features), XGBoost, LightGBM, CatBoost
  - For each fold: trains all 5 models, collects predictions on val set
  - Returns dict: `{model_name: {'fold_briers': [...], 'mean_brier': float, 'oof_preds': array}}`
  - Also stores out-of-fold (OOF) predictions for stacking
  - Logs per-fold Brier for each model

- [ ] `simple_average_ensemble(oof_preds_dict) -> np.ndarray`
  - Takes dict of `{model_name: oof_pred_array}`, returns simple mean
  - Evaluate: `evaluate_brier(y_true, simple_avg)` — should beat best single model

- [ ] `optimize_ensemble_weights(oof_preds_dict, y_true) -> dict`
  - Uses `scipy.optimize.minimize` to find weights that minimize Brier on OOF predictions
  - Constraint: weights sum to 1.0, all weights >= 0
  - Returns `{model_name: weight}`
  - Log the optimal weights — they reveal which models are most valuable

- [ ] `weighted_ensemble(oof_preds_dict, weights) -> np.ndarray`
  - Applies given weights and returns weighted average prediction

- [ ] Run full ensemble CV for BOTH M and W
- [ ] Log all results to `artifacts/ensemble_results.txt`
- [ ] **Simple average ensemble Brier < 0.210 for M** (must beat single XGBoost)
- [ ] **Weighted ensemble Brier < 0.205 for M** (target, not mandatory to proceed)

---

### GSD Checklist — Recency Weighting

- [ ] Add `compute_sample_weights(seasons: list[int]) -> np.ndarray` to `src/ensemble.py`
  - Linear decay: most recent season weight=1.0, oldest=0.3
  - `weights = np.linspace(0.3, 1.0, len(seasons))` mapped per row
- [ ] Re-run XGBoost CV with sample weights — log whether it improves Brier
- [ ] If improved: add `sample_weight` param to `train_xgboost()` and `train_lightgbm()`

---

### GSD Checklist — Tests

Create `tests/test_ensemble.py`:

- [ ] Test: `train_lightgbm` returns model with `.predict_proba()` that outputs (N, 2) array
- [ ] Test: `train_catboost` same
- [ ] Test: `train_ridge` (logistic L2) same
- [ ] Test: all three models' predictions are in (0, 1)
- [ ] Test: `simple_average_ensemble` output shape matches input
- [ ] Test: `optimize_ensemble_weights` weights sum to 1.0 (within 1e-6)
- [ ] Test: weighted ensemble Brier <= best single model Brier on same OOF data
- [ ] Test: `compute_sample_weights` — most recent season has highest weight
- [ ] `pytest tests/test_ensemble.py -q` — ALL PASS
- [ ] `pytest tests/ -q` — ALL existing tests still pass

---

### GSD Checklist — Final Submission

- [ ] Train final ensemble on ALL available data (no held-out val — use all seasons):
  - M: train on 2003-2025, W: train on 2010-2025
  - Use optimal weights from OOF optimization
- [ ] Generate Stage 1 submission: `submissions/ensemble_v1.csv`
- [ ] Verify `ensemble_v1.csv`: 519,144 rows, no NaN, Pred in [0.05, 0.95], std > 0.05
- [ ] Generate Stage 2 submission: `submissions/ensemble_2026.csv`
  - This is the real competition submission (2026 season)
- [ ] Save `artifacts/ensemble_weights.json` — the optimal weights per model per gender
- [ ] Save `artifacts/ensemble_results.txt` — all CV Brier scores

---

### GSD Checklist — Phase Wrap

- [ ] `pytest tests/ -q` — ALL tests pass
- [ ] `artifacts/ensemble_results.txt` documents all model Brier scores
- [ ] `submissions/ensemble_v1.csv` exists and is valid (519,144 rows)
- [ ] `submissions/ensemble_2026.csv` exists and is valid (132,133 rows)
- [ ] `git add -A && git commit -m "phase 4: ensemble pipeline complete"`
- [ ] Update `CLAUDE.md`:
  - Mark Phase 4 complete
  - Log ensemble Brier scores in score table
  - Log optimal ensemble weights in Key Decisions

---

### Expected File Changes Summary

```
src/models.py          — ADD: train_lightgbm(), train_catboost(), train_ridge()
src/ensemble.py        — CREATE NEW: run_all_models_cv(), simple/weighted ensemble, weight optimizer
tests/test_ensemble.py — CREATE NEW: tests for all new model wrappers + ensemble logic
artifacts/ensemble_results.txt   — NEW: all CV scores
artifacts/ensemble_weights.json  — NEW: optimal model weights
submissions/ensemble_v1.csv      — NEW: Stage 1 ensemble submission
submissions/ensemble_2026.csv    — NEW: Stage 2 (2026) ensemble submission
```

---

## Phase 5: Hyperparameter Tuning — EOA + Ax/BoTorch

**Load skill:** `bash skills.sh hyperparameter-tuning`

> **Goal: tune individual model HPs and ensemble weights using two independent optimizers. Track all trials in TensorBoard HParams. Target: M Brier < 0.195.**

### Design Decisions
1. **EOA vs Ax: Independent (compare)** — run both on the same tasks, pick the winner per model
2. **Search scope: Independent per model** — XGBoost (~7 dims), LightGBM (~7 dims), CatBoost (~3 dims)
3. **Weight tuning: Sequential** — first tune model HPs, then optimize ensemble weights with tuned models

### Infrastructure Already Built
- `src/tuning.py` — search spaces, `evaluate_params()`, `evaluate_ensemble_weights()`, TensorBoard HParams logging
- `src/tuning_eoa.py` — `tune_model_eoa()`, `tune_ensemble_weights_eoa()` via mealpy `EOA.OriginalEOA`
- `src/tuning_ax.py` — `tune_model_ax()`, `tune_ensemble_weights_ax()` via Ax `AxClient` + BoTorch GP
- `scripts/run_tuning.py` — end-to-end orchestration: tune all models × both methods × both genders
- `tests/test_tuning.py` — 12 tests covering search spaces, evaluation, EOA, Ax

### GSD Checklist — Run Tuning

- [ ] `python scripts/run_tuning.py` — run full tuning pipeline (est. 1-2 hours)
- [ ] TensorBoard logs in `runs/` — verify with `tensorboard --logdir runs/`
- [ ] `artifacts/tuning_results.json` — complete comparison of EOA vs Ax per model/gender
- [ ] `artifacts/tuned_params.json` — best params per model per gender

### GSD Checklist — Tuned Submission

- [ ] Update `scripts/generate_ensemble_submission.py` to load tuned params
- [ ] Generate `submissions/ensemble_tuned_v1.csv` (Stage 1, 519,144 rows)
- [ ] Generate `submissions/ensemble_tuned_2026.csv` (Stage 2)
- [ ] Validate both submissions

### GSD Checklist — Phase Wrap

- [ ] `pytest tests/ -q` — ALL tests pass
- [ ] TensorBoard HParams dashboard viewable
- [ ] Tuned ensemble Brier < 0.195 for M
- [ ] `git add -A && git commit -m "phase 5: hyperparameter tuning complete"`
- [ ] Update `CLAUDE.md`: scores, phase status, key decisions

---

## Escalation Protocol

### 🔴 STOP — Escalate Immediately
- Any test in `test_data_quality.py` fails and you can't explain why
- Brier Score > 0.25 on baseline (pipeline is broken)
- Score reconstruction test fails (box scores don't add up)
- Feature matrix has > 20% NaN in any column
- Sample submission IDs don't parse correctly
- Required package won't install

### 🟡 PAUSE — Ask Before Proceeding
- You want to drop a dataset or feature due to quality concerns
- You found a data anomaly (e.g., duplicate games, impossible scores)
- M and W pipelines diverge significantly in structure
- You're unsure if a feature introduces temporal leakage

### 🟢 PROCEED — Log and Continue
- Minor NaN in non-critical columns (< 5%)
- A few teams have very few games (< 10 per season)
- Some Massey systems have sparse coverage (use what's available)

---

## Context Window Survival Guide

**If you're reading this mid-session and context feels long:**

1. How many tool calls have you made? If > 30, checkpoint.
2. Are you in the middle of a sub-task? Finish it, test it.
3. Update `CLAUDE.md` with exactly where you are.
4. Commit: `git add src/ tests/ CLAUDE.md && git commit -m "checkpoint"`
5. Tell the user: "Checkpointing. Next session should start at [specific task]."

**Prevention:**
- Write code to files directly. Never show code in chat and then also write it.
- `pytest -q` not `-v`. One line per test, not paragraphs.
- `cmd > file.txt 2>&1 && tail -10 file.txt` for anything verbose.
- Don't read entire source files. Use grep or read specific line ranges.
- Don't load multiple skills. One at a time.
