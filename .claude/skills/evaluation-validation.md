# Skill: Evaluation & Validation

## When to Use
Reference this skill when setting up cross-validation, evaluating models, calibrating probabilities, or comparing model performance.

## Competition Metric: Brier Score

```python
Brier = mean((predicted - actual)^2)
```

- Range: [0, 1]. Lower is better.
- Perfect predictions = 0.0
- Always predicting 0.5 = 0.25 (coin flip baseline)
- A good model: 0.18-0.20
- A great model: 0.16-0.18

**Key difference from Log Loss**: Brier Score is bounded and less punishing on extreme misses.
- Log Loss of predicting 0.99 for a loss: -ln(0.01) = 4.6
- Brier of predicting 0.99 for a loss: (0.99)^2 = 0.98

Still clip predictions, but you have more room to be confident.

## Cross-Validation: Expanding Window

**This is the ONLY valid CV strategy for this competition.**

```python
def expanding_window_cv(df, target_col='target', season_col='Season', min_train_end=2019):
    """
    Yields (train_idx, val_idx) tuples.
    Train on all seasons up to N, validate on season N+1.
    """
    seasons = sorted(df[season_col].unique())
    for val_season in seasons:
        if val_season <= min_train_end:
            continue
        train_mask = df[season_col] < val_season
        val_mask = df[season_col] == val_season
        if train_mask.sum() > 0 and val_mask.sum() > 0:
            yield df[train_mask].index, df[val_mask].index
```

**Why not random K-fold?**
- Tournament dynamics change over time (new 3pt line, shot clock changes)
- Random splits leak future information into training
- Expanding window simulates real prediction scenario

## Calibration

### Why Calibrate?
GBMs often produce poorly calibrated probabilities. A predicted 0.7 should mean the team wins ~70% of the time.

### Methods
```python
from sklearn.calibration import CalibratedClassifierCV

# Platt Scaling (sigmoid fit) — works well with small datasets
calibrated = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)

# Isotonic Regression — more flexible but needs more data
calibrated = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
```

### Probability Clipping — MANDATORY
```python
predictions = np.clip(predictions, 0.05, 0.95)
```
This protects against catastrophic Brier Score on upsets. An unclipped 0.99 prediction on a loss costs 0.98 Brier. A clipped 0.95 costs 0.90. Small difference on Brier, but adds up.

### Reliability Diagram
Always generate a reliability diagram to visually check calibration:
```python
from sklearn.calibration import calibration_curve

fraction_positive, mean_predicted = calibration_curve(y_true, y_pred, n_bins=10)
# Plot: x=mean_predicted, y=fraction_positive
# Perfect calibration = diagonal line
```

## Model Comparison Protocol

When comparing models:
1. Run ALL models through the SAME CV folds
2. Compare mean Brier Score across folds
3. Also compare per-fold to check consistency
4. A model that's 0.002 better on average but has high variance across folds may not be worth it
5. Statistical significance: use paired t-test on fold scores if needed

## Red Flags (Escalate if You See These)
- Any model Brier Score > 0.25 (worse than coin flip)
- Brier Score improves dramatically when you add a suspicious feature (likely leakage)
- Validation scores much better than expected (< 0.15 on this data = probably leaking)
- Predictions concentrated near 0.5 with std < 0.05 (model isn't learning)
- Huge variance between CV folds (unstable model)
