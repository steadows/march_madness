# March Madness 2026 — Agent Operating Manual

> **AUTO-LOADED every session. This is your brain between sessions. READ IT. UPDATE IT. TRUST IT.**

---

## 🎯 Mission

Kaggle competition: March Machine Learning Mania 2026. Predict NCAA tournament game probabilities.
- **Metric**: Brier Score (lower = better, coin flip = 0.25, good model = 0.18-0.20)
- **Deadline**: March 19, 2026 4:00 PM UTC
- **Submission**: CSV with win probabilities for every possible team pair (M + W combined)
- **Prize**: $50K total, $10K for 1st

---

## 🚦 Session Protocol — READ THIS FIRST EVERY TIME

### Step 1: Orient
1. Read this file (you're doing it)
2. Check **Current Status** below — what phase are you on?
3. Read your current phase in `PIPELINE_BUILD_GUIDE.md`
4. Load ONLY the skills you need: `bash skills.sh` to see the index

### Step 2: Work
- **One phase per session.** Do not try to do everything at once.
- Write code directly to disk. Do NOT draft code in conversation — write it with the Write tool.
- Run `pytest tests/ -q` (quiet). Only `-v` when debugging a specific failure.
- Pipe verbose outputs to files: `python3 script.py > artifacts/output.log 2>&1`
- If a command output exceeds ~50 lines, redirect to a file and read only what you need.

### Step 3: Context Window Management — CRITICAL
**You WILL hit the context limit if you are not disciplined. Follow these rules:**

- ✅ **DO**: Write code to files immediately (don't show it in chat first)
- ✅ **DO**: Use `pytest -q` (1 line per test, not full traces)
- ✅ **DO**: Redirect long output: `cmd > file.txt 2>&1`, then `tail -20 file.txt`
- ✅ **DO**: Summarize findings in 2-3 bullet points, not full analysis paragraphs
- ✅ **DO**: Update CLAUDE.md status and end session cleanly when a phase is done
- ❌ **DON'T**: Cat/read entire large files — read specific line ranges
- ❌ **DON'T**: Re-read src files you just wrote — trust your own output
- ❌ **DON'T**: Load multiple skill files at once — one at a time
- ❌ **DON'T**: Run EDA or exploration that prints huge DataFrames — use `.head()`, `.shape`, `.describe()`
- ❌ **DON'T**: Ask the user open-ended questions — check if the answer is in this file or the build guide first

**IF YOU FEEL THE CONTEXT GETTING LONG** (you've done 30+ tool calls):
1. STOP what you're doing
2. Run verification checks on what you've completed so far
3. Update CLAUDE.md with your progress
4. Commit your code: `git add src/ tests/ && git commit -m "phase N: progress checkpoint"`
5. Tell the user: "Phase N partially complete. Committing and suggesting a new session to continue."

### Step 4: End of Phase
When a phase's verification checks ALL pass:
1. ✅ Update the **Phase Progress** checkboxes below
2. ✅ Update **Current Phase** status line
3. ✅ Log any **Key Decisions** made during this phase
4. ✅ Update **Best Brier Scores** if applicable
5. ✅ Log any **Known Issues** discovered
6. ✅ Commit: `git add -A && git commit -m "phase N: <description>"`
7. ✅ Tell the user what was accomplished and what's next

### Skill Loading Protocol
**DO NOT read all skills at once.** Use `bash skills.sh` to see the index, then load what you need:
```bash
bash skills.sh                              # See all available skills + phase mapping
bash skills.sh model-training               # Load a custom (project-specific) skill
bash skills.sh c/python-testing-patterns     # Load a community skill (c/ prefix)
```

Two types of skills:
- **Custom skills** (~1K tokens each) — project-specific, written for this competition
- **Community skills** (~3-6K tokens each, `c/` prefix) — general Python engineering patterns from skills.sh

**Phase → Skill map (load 1 custom + 1 community MAX per session):**
| Phase | Custom Skill | Community Skill |
|-------|-------------|----------------|
| 0: Setup | `project-conventions` | — |
| 1: Data | `data-engineering` | `c/data-quality-frameworks` or `c/python-testing-patterns` |
| 2: Features | `data-engineering` | `c/python-performance-optimization` or `c/python-type-safety` |
| 3: Training | `model-training` or `evaluation-validation` | `c/test-driven-development` |
| **4: Ensemble** | **`ensemble-submission`** | **—** |
| 5: Iteration | `hyperparameter-tuning` or `external-data` | — |
| Debugging | `debugging` | `c/python-error-handling` |
| Commits | — | `c/git-commit` |

---

## 📊 Current Status

### Phase Progress
- [x] Phase 0: Environment Setup
- [x] Phase 1: Data Loading & EDA
- [x] Phase 2: Feature Engineering (Elo, Massey, Four Factors, Differentials)
- [x] Phase 3: Model Training & CV (XGBoost, LightGBM, CatBoost, Logistic baseline)
- [x] Phase 4: Ensemble Pipeline (LightGBM + CatBoost + Ridge + weighted ensemble)
- [ ] Phase 5: Iteration & Improvement

### 🔵 Current Phase: Phase 5 — Hyperparameter Tuning (IN PROGRESS)
<!-- Tuning complete. Bug fixed: ensemble CV now uses tuned params. -->
<!-- Next: run scripts/generate_tuned_submission.py to get final tuned scores + submissions -->
<!-- Then: update Brier scores table, commit, and close Phase 5 -->

### 📈 Best Brier Scores
| Model | Men | Women | CV Folds | Notes |
|-------|-----|-------|----------|-------|
| Seed-only logistic | 0.202 | 0.147 | 5 folds (2020-2024) | Floor baseline |
| Ridge (L2 logistic) | 0.202 | 0.143 | 5 folds (2020-2024) | All 38 features |
| XGBoost (default) | 0.197 | 0.144 | 5 folds (2020-2024) | All 38 features |
| LightGBM (default) | 0.200 | 0.145 | 5 folds (2020-2024) | All 38 features |
| CatBoost (default) | 0.197 | 0.140 | 5 folds (2020-2024) | All 38 features |
| Weighted Ensemble (default) | 0.195 | 0.139 | 5 folds (2020-2024) | Pre-tuning best |
| XGBoost (tuned) | 0.184 | 0.132 | 5 folds (2020-2024) | EOA-tuned HPs |
| LightGBM (tuned) | 0.186 | 0.133 | 5 folds (2020-2024) | EOA-tuned HPs |
| CatBoost (tuned) | 0.186 | 0.130 | 5 folds (2020-2024) | EOA-tuned HPs |
| Simple Avg (tuned) | 0.186 | 0.131 | 5 folds (2020-2024) | Tuned models |
| Weighted Ensemble (tuned, Ax weights) | **0.1799** | — | 5 folds (2020-2024) | **BEST M** — Ax won weights (vs scipy 0.1800, EOA 0.1802) |
| Weighted Ensemble (tuned, scipy weights) | — | **0.1263** | 5 folds (2020-2024) | **BEST W** — scipy won weights (tied Ax 0.1263, EOA 0.1266) |
| **Kaggle Stage 1** | — | — | Leaderboard | **0.00396** (prev 0.03627) — ensemble_tuned_v1.csv |

### 🔑 Key Decisions
<!-- Log decisions so future sessions don't re-debate. Format: "DECISION: <what> — <why>" -->
- DECISION: Use full path `/opt/anaconda3/envs/march_madness/bin/python` for all Python commands — base env is different
- DECISION: WTeams.csv has only TeamID+TeamName (no FirstD1Season/LastD1Season) — W teams load differently than M
- DECISION: Seed 16 win rate ~25.8% in tournament is correct — inflated by First Four 16v16 play-in games
- DATA: Massey has 196 systems (not "60+" as docs say), top systems by coverage: AP, DOL, USA, WLK, POM, MOR, COL (all 24 seasons)
- DECISION: seed_num_diff ranks low in XGBoost importance (rank 38) despite -0.48 correlation with target — expected collinearity artifact (POM/MOR/ELO all encode quality). Feature is correct, not a bug.
- DECISION: coach_tourney_exp_diff is M-only; always 0 for W data (coaches file is M only). OK for GBMs, excluded from W variance checks.
- DECISION: Training on all seasons (including pre-2003 M, pre-2010 W) gives 31-41% NaN but modern-only (2003+/2010+) is 3.3% NaN. GBMs handle NaN natively.
- DECISION: M ensemble weights: XGBoost 50%, CatBoost 33%, Logistic 13%, LightGBM 4%, Ridge 0%. Ridge zeroed out by optimizer.
- DECISION: W ensemble weights: CatBoost 58%, Ridge 22%, Logistic 16%, LightGBM 4%, XGBoost 0%. XGBoost zeroed out for W.
- DECISION: Phase 5 uses two independent optimizers: EOA (mealpy) + Ax/BoTorch. Compare independently, pick winner per model.
- DECISION: Tune models independently (~7 dims each), not jointly (~17 dims). Sequential: HPs first, then ensemble weights.
- DECISION: All trials tracked in TensorBoard HParams (runs/ directory) for visual comparison.
- DECISION: EOA early stopping patience=15 epochs — M-XGBoost converged at epoch 15, no improvement through epoch 39+.
- TUNING RESULT: M-XGBoost EOA best (Brier 0.1826): max_depth=9, learning_rate=0.242, colsample_bytree=0.527, min_child_weight=9, reg_alpha=0.833, reg_lambda=0.506, subsample=0.821. Pattern: deeper trees + higher LR + more regularization + less feature sampling vs defaults.
- BUG FIX: `run_all_models_cv()` was missing `model_params` kwarg — ensemble weight optimization was using default (untuned) model predictions. Fixed by adding `model_params` arg; new `scripts/generate_tuned_submission.py` loads tuned params and re-runs everything properly.
- DECISION: First EOA run M-XGBoost (Brier 0.1826, max_depth=9) beat second run (0.1854, max_depth=4). Manually inserted first run's params into tuned_params.json.
- TUNING RESULT: Weight optimization 3-way comparison — EOA lost both (M: 0.1802, W: 0.1266). Ax won M (0.1799), scipy won W (0.1263). All within 0.0003. Weight blending is smooth/low-dim, favors BO/Nelder-Mead over population search. EOA's strength is rugged HP tuning, not weight mixing.

### ⚠️ Known Issues / Blockers
<!-- Format: "ISSUE: <what> — SEVERITY: high/medium/low — STATUS: open/resolved" -->
- ISSUE: WTeams.csv missing FirstD1Season/LastD1Season columns (only TeamID+TeamName) — SEVERITY: low — STATUS: resolved (tests adjusted, W teams still load fine)

### 📋 Feature Columns
<!-- After Phase 2, list ALL final feature column names here. Phase 3 reads this list instead of re-deriving. -->
38 differential features (all end in `_diff`), see `artifacts/feature_columns.json` for full list.
Key features: `win_pct`, `pts_per_game`, `pts_allowed_per_game`, `off_eff`, `def_eff`, `net_eff`,
`efg_pct`, `to_rate`, `or_pct`, `ft_rate`, `fg3_rate`, `ast_to_ratio`, `stl_per_game`, `blk_per_game`,
`recent_win_pct`, `recent_off_eff`, `recent_def_eff`, `recent_net_eff`, `recent_efg_pct`,
`recent_to_rate`, `recent_or_pct`, `recent_ft_rate`, `recent_fg3_rate`,
`recent_pts_per_game`, `recent_pts_allowed_per_game`,
`elo`, `pom_rank`, `sag_rank`, `mor_rank`, `wol_rank`, `dol_rank`, `col_rank`, `rpi_rank`, `ap_rank`, `usa_rank`,
`seed_num`, `sos_elo`, `coach_tourney_exp`
Feature matrix files: `artifacts/features_men.csv` (2585×42), `artifacts/features_women.csv` (1717×42)
To get feature cols: `[c for c in df.columns if c.endswith('_diff')]`

---

## 📌 Quick Reference (Don't re-look-up these facts)

### Data Facts
- 35 CSVs in `data/`. Prefix `M` = men, `W` = women. Identical structure.
- Detailed box scores: M since 2003, W since 2010
- Compact results: M since 1985, W since 1998
- Massey Ordinals: 5.8M rows, 60+ ranking systems, M only
- TeamIDs: Men = 1xxx, Women = 3xxx (no overlap)
- Tournament starts ~Day 134. Regular season before that.

### Submission Format
```
ID,Pred
2026_1101_1102,0.5
```
- `{Season}_{LowerTeamID}_{HigherTeamID}` — lower ID always first
- `Pred` = probability the lower-ID team wins
- **CLIP to [0.05, 0.95]** — non-negotiable
- Stage 1: ~519K rows (seasons 2022-2025). Stage 2: ~132K rows (2026)

### Key Files
```
PIPELINE_BUILD_GUIDE.md  — Phase-by-phase build spec with verification checks
COMPETITION.md           — Full competition details, data dictionary, methods research
skills.sh                — Skill index and loader (bash skills.sh <name>)
.claude/skills/*.md      — Individual skill files
```

### Conda Environment
```bash
# Environment: march_madness (Python 3.11.14)
# Location: /opt/anaconda3/envs/march_madness/

# Activate:
conda activate march_madness

# Run Python:
/opt/anaconda3/envs/march_madness/bin/python

# Run pytest:
/opt/anaconda3/envs/march_madness/bin/python -m pytest tests/ -q

# IMPORTANT: Always use the full path or activate the env first.
# Never use bare `python3` — that hits the base anaconda env.
```

### Installed Packages
- pandas 3.0.1, numpy 2.4.2, scikit-learn 1.8.0
- xgboost 3.2.0, lightgbm 4.6.0, catboost 1.2.10
- scipy 1.17.1, optuna 4.7.0, pytest 9.0.2
- matplotlib 3.10.8, seaborn 0.13.2, jupyter
