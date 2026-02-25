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
| 4: Ensemble | `ensemble-submission` | — |
| 5: Iteration | `hyperparameter-tuning` or `external-data` | — |
| Debugging | `debugging` | `c/python-error-handling` |
| Commits | — | `c/git-commit` |

---

## 📊 Current Status

### Phase Progress
- [x] Phase 0: Environment Setup
- [x] Phase 1: Data Loading & EDA
- [ ] Phase 2: Feature Engineering (Elo, Massey, Four Factors, Differentials)
- [ ] Phase 3: Model Training & CV (XGBoost, LightGBM, CatBoost, Logistic baseline)
- [ ] Phase 4: Ensemble & Submission Generation
- [ ] Phase 5: Iteration & Improvement

### 🔵 Current Phase: Phase 2 — Feature Engineering (NOT STARTED)
<!-- AGENT: Update this line EVERY session. Examples: -->
<!-- "Phase 2 — Elo system done, Massey processing in progress" -->
<!-- "Phase 3 — XGBoost trained (Brier 0.19), starting LightGBM" -->

### 📈 Best Brier Scores
| Model | Men | Women | CV Folds | Notes |
|-------|-----|-------|----------|-------|
| Seed-only logistic | — | — | — | Floor baseline |
| Elo-only logistic | — | — | — | |
| XGBoost | — | — | — | |
| LightGBM | — | — | — | |
| CatBoost | — | — | — | |
| Simple Avg Ensemble | — | — | — | |
| Weighted Ensemble | — | — | — | **BEST** |

### 🔑 Key Decisions
<!-- Log decisions so future sessions don't re-debate. Format: "DECISION: <what> — <why>" -->
- DECISION: Use full path `/opt/anaconda3/envs/march_madness/bin/python` for all Python commands — base env is different
- DECISION: WTeams.csv has only TeamID+TeamName (no FirstD1Season/LastD1Season) — W teams load differently than M
- DECISION: Seed 16 win rate ~25.8% in tournament is correct — inflated by First Four 16v16 play-in games
- DATA: Massey has 196 systems (not "60+" as docs say), top systems by coverage: AP, DOL, USA, WLK, POM, MOR, COL (all 24 seasons)

### ⚠️ Known Issues / Blockers
<!-- Format: "ISSUE: <what> — SEVERITY: high/medium/low — STATUS: open/resolved" -->
- ISSUE: WTeams.csv missing FirstD1Season/LastD1Season columns (only TeamID+TeamName) — SEVERITY: low — STATUS: resolved (tests adjusted, W teams still load fine)

### 📋 Feature Columns
<!-- After Phase 2, list ALL final feature column names here. Phase 3 reads this list instead of re-deriving. -->

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
