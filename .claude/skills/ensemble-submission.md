# Skill: Ensemble & Submission

## When to Use
Reference this skill when combining model predictions, optimizing ensemble weights, or generating the submission file.

## Ensemble Methods (in order of simplicity)

### 1. Simple Average
```python
ensemble_pred = np.mean([xgb_pred, lgb_pred, cat_pred], axis=0)
```
Simple, robust, often hard to beat. Start here.

### 2. Optimized Weighted Average
```python
from scipy.optimize import minimize

def brier_loss(weights, preds_list, y_true):
    weights = np.array(weights)
    weights = weights / weights.sum()  # normalize
    blended = sum(w * p for w, p in zip(weights, preds_list))
    return np.mean((blended - y_true) ** 2)

result = minimize(
    brier_loss,
    x0=[1/n_models] * n_models,  # equal weights start
    args=(preds_list, y_true),
    method='Nelder-Mead',
    bounds=[(0, 1)] * n_models,
)
optimal_weights = result.x / result.x.sum()
```

Optimize on the LAST validation fold only (not the full training set).

### 3. Stacking
```python
# Level 0: Get out-of-fold predictions from each base model
# For each CV fold, train base models and predict on the held-out fold
# Collect all OOF predictions → these become features for Level 1

# Level 1: Train a simple model on OOF predictions
from sklearn.linear_model import LogisticRegression
meta_model = LogisticRegression(C=1.0)
meta_model.fit(oof_predictions, y_train)
```

**Rules for stacking:**
- Level 1 model should be SIMPLE (logistic regression or ridge)
- Use OOF predictions only — never in-sample predictions
- Stacking with < 500 training samples is risky (tournament data is small!)

## Submission File Format

### Structure
```csv
ID,Pred
2026_1101_1102,0.5
2026_1101_1103,0.65
2026_3101_3102,0.48
```

### ID Format
`{Season}_{TeamID_lower}_{TeamID_higher}`
- Lower TeamID always comes first
- Season is always the current year for Stage 2 (2026)
- Stage 1 includes historical seasons for leaderboard scoring

### Generation Protocol
```python
def generate_submission(sample_path, predict_fn, output_path):
    """
    Args:
        sample_path: Path to SampleSubmission CSV
        predict_fn: function(season, team_a, team_b) -> float
        output_path: Where to write the submission
    """
    sample = pd.read_csv(sample_path)
    predictions = []

    for _, row in sample.iterrows():
        parts = row['ID'].split('_')
        season = int(parts[0])
        team_a = int(parts[1])  # lower ID
        team_b = int(parts[2])  # higher ID

        pred = predict_fn(season, team_a, team_b)
        pred = np.clip(pred, 0.05, 0.95)
        predictions.append(pred)

    sample['Pred'] = predictions
    sample.to_csv(output_path, index=False)
```

### Validation Checks (MUST pass before upload)
```python
def validate_submission(submission_path, sample_path):
    sub = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_path)

    # 1. Same number of rows
    assert len(sub) == len(sample), f"Row count mismatch: {len(sub)} vs {len(sample)}"

    # 2. Same IDs in same order
    assert (sub['ID'] == sample['ID']).all(), "ID mismatch"

    # 3. All predictions are valid floats
    assert sub['Pred'].notna().all(), "NaN predictions found"

    # 4. All predictions in valid range
    assert (sub['Pred'] >= 0.0).all() and (sub['Pred'] <= 1.0).all(), "Predictions out of [0,1]"

    # 5. Predictions clipped
    assert (sub['Pred'] >= 0.05).all() and (sub['Pred'] <= 0.95).all(), "Predictions not clipped to [0.05, 0.95]"

    # 6. Sanity: predictions have spread
    assert sub['Pred'].std() > 0.05, f"Predictions too concentrated: std={sub['Pred'].std():.4f}"

    print(f"✓ Submission valid: {len(sub)} rows, mean={sub['Pred'].mean():.4f}, std={sub['Pred'].std():.4f}")
```

### Stage 1 vs Stage 2
- **Stage 1**: Covers historical seasons (2022-2025). Used for pre-tournament leaderboard. Submit against `SampleSubmissionStage1.csv`.
- **Stage 2**: Covers 2026 only. This is the REAL competition. Submit against `SampleSubmissionStage2.csv` (released closer to tournament).
- Build pipeline to handle BOTH — just change which sample file you read.

### Teams Without Data
Some team pairs in the submission file may include teams you have minimal data for. Fallback strategy:
- If no features available: predict 0.5 (uninformed prior)
- If only seeds available: use seed-based logistic regression
- If Elo available but no detailed stats: use Elo-based prediction
- Layer your fallbacks from most-informed to least-informed
