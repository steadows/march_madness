# Skill: Debugging

## When to Use
Load this skill when tests fail, code errors out, or results look wrong.

## Triage Protocol
When something breaks, follow this order:

### 1. Read the Error (Don't Guess)
```bash
# Run the failing test in isolation with full output
pytest tests/test_specific.py::test_function -v --tb=short
```
- `--tb=short` gives you the traceback without flooding context
- Never `--tb=long` unless short didn't give enough info

### 2. Common Error Patterns

**KeyError on DataFrame column:**
- Wrong column name. Check: `df.columns.tolist()`
- Gender mismatch: M files have different prefixes than W
- Season filter excluded all data: check `df.shape` after filtering

**Shape mismatch / dimension errors:**
- Features built for M but applied to W (different season ranges)
- NaN rows not dropped before model training
- Submission IDs don't match sample — check parsing

**Import errors:**
- Package not installed: `pip install <package>`
- Circular import: check if `src/a.py` imports from `src/b.py` which imports from `src/a.py`
- Relative imports: use `from src.module import func` or set PYTHONPATH

**Model training errors:**
- XGBoost NaN in features: `print(df.isna().sum())` to find them
- CatBoost categorical issues: ensure categorical features are strings, not ints
- Memory error on Massey: use `dtype` optimization (see data-engineering skill)

**Brier Score looks wrong:**
- Score > 0.25: model is worse than coin flip. Check label encoding (is 1 = correct team?)
- Score ≈ 0.25: model isn't learning. Check features aren't all NaN or constant
- Score < 0.10: too good, you have data leakage. Check temporal ordering.

### 3. Data Leakage Detection
The #1 silent killer. Check for:
- Are you using tournament results as features for predicting that same tournament?
- Are you using Season X+1 data to predict Season X?
- Is the Massey ordinal ranking day AFTER the tournament started?
- Are you training on the same data you're validating on?

**Quick leakage test:**
```python
# Train on 2003-2022, predict 2023. Score should be 0.18-0.22.
# If it's < 0.12, something is leaking.
```

### 4. Diagnostic One-Liners
```python
# Feature matrix health check
python3 -c "
import pandas as pd
df = pd.read_csv('artifacts/features.csv')
print(f'Shape: {df.shape}')
print(f'NaN per col:\n{df.isna().sum()[df.isna().sum() > 0]}')
print(f'Target dist:\n{df[\"target\"].value_counts()}')
print(f'Constant cols: {[c for c in df.columns if df[c].nunique() <= 1]}')
"
```

### 5. When to Escalate to Human
- Error persists after 3 fix attempts on the same issue
- You suspect data corruption (file won't parse, unexpected values)
- Test passes but output is nonsensical (Brier = 0.0, all predictions identical)
- You need to install a system-level dependency (not pip)
- External data source is down or changed format
