# Skill: Model Training

## When to Use
Reference this skill when training models, tuning hyperparameters, or evaluating model performance.

## Model Priority (by expected impact)

### Tier 1: Must-Build
1. **Logistic Regression** — seed_diff only baseline. If this scores Brier > 0.25, your data pipeline is broken.
2. **XGBoost** — primary workhorse
3. **LightGBM** — fast, often competitive with XGBoost
4. **CatBoost** — handles categoricals natively, good diversity in ensemble

### Tier 2: Should-Build
5. **Elo-based logistic model** — standalone Elo → probability
6. **Ridge Regression** — regularized linear model for ensemble diversity

### Tier 3: Nice-to-Have
7. **Neural Network** — team embeddings, feedforward
8. **Random Forest** — additional ensemble diversity

## XGBoost Configuration
```python
import xgboost as xgb

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',  # train with logloss even though comp uses Brier
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 1000,
    'early_stopping_rounds': 50,
    'verbosity': 0,
}
```

## LightGBM Configuration
```python
import lightgbm as lgb

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 1000,
    'verbose': -1,
}
```

## CatBoost Configuration
```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    eval_metric='Logloss',
    verbose=0,
    early_stopping_rounds=50,
)
```

## Training Protocol

### Always
1. Use the expanding window CV from `src/cv.py` — never random splits
2. Use early stopping on validation set within each fold
3. Save feature importances from every model
4. Report Brier Score (not Log Loss!) as the primary metric
5. Train M and W models completely separately

### Brier Score Computation
```python
from sklearn.metrics import brier_score_loss

# Lower is better. Perfect = 0.0, Coin flip = 0.25
brier = brier_score_loss(y_true, y_pred)
```

### Feature Importance Analysis
After training, check:
- Top features should be: seed_diff, elo_diff, massey system diffs
- If random/nonsensical features rank high, you may have data leakage
- Drop features with near-zero importance to reduce noise

## Common Mistakes to Avoid
- **Training on tournament data and testing on tournament data from same season** — this is leakage
- **Using current-season tournament results as features** — you're predicting BEFORE the tournament
- **Not using early stopping** — GBMs will overfit badly on small tournament datasets
- **Optimizing Log Loss when competition uses Brier** — they're related but not identical
- **Treating all seasons equally** — recent seasons are more predictive. Consider sample weighting.

## Sample Weighting (Optional Advanced)
Weight recent seasons more:
```python
# Linear decay: most recent season gets weight 1.0, oldest gets 0.3
weights = np.linspace(0.3, 1.0, n_seasons)
```
