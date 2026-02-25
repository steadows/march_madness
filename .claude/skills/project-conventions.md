# Skill: Project Conventions

## When to Use
Reference this skill when creating any new file, function, or module in the project.

## Code Style
- Python 3.10+ (use type hints everywhere)
- Use `pathlib.Path` for all file paths, never string concatenation
- Docstrings on every public function (Google style)
- Functions should do ONE thing. If a function exceeds 50 lines, break it up.
- No global mutable state. Use `src/config.py` for constants only.

## Naming Conventions
- Files: `snake_case.py`
- Functions: `snake_case()`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- DataFrames: suffix with `_df` (e.g., `games_df`, `features_df`)
- Feature columns: descriptive, like `seed_diff`, `elo_diff`, `off_efficiency_diff`

## File Organization
```
src/config.py    → All constants, paths, feature lists
src/data_loader.py → Data I/O only, no transformations
src/feature_engineering.py → All feature computation
src/models.py    → Model training and prediction
src/cv.py        → Cross-validation logic
src/ensemble.py  → Ensembling methods
src/calibration.py → Probability calibration
src/submission.py → Submission file generation
src/utils.py     → Shared helpers
```

## Data Flow
```
Raw CSVs → data_loader → DataFrames → feature_engineering → Feature Matrix → models → Predictions → ensemble → calibration → submission
```

## Gender Convention
- Men's data: files prefixed with `M`, TeamIDs in 1000-1999 range
- Women's data: files prefixed with `W`, TeamIDs in 3000-3999 range
- Functions that load gender-specific data take `gender='M'` or `gender='W'`
- Models are trained SEPARATELY for M and W — never mix them

## Testing
- Every `src/*.py` module gets a corresponding `tests/test_*.py`
- Tests must be runnable with `pytest tests/ -v`
- Use `pytest.approx()` for floating point comparisons
- Test with small data slices for speed (e.g., single season)

## Commit Convention
- Commit after each completed phase
- Message format: `phase N: brief description of what was done`
- Example: `phase 2: add elo rating system and massey ordinal features`
