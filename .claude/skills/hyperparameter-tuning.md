# Skill: Hyperparameter Tuning

## When to Use
Load this skill during Phase 5 (Iteration) or whenever you need to optimize model hyperparameters.

## Framework: Optuna
```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'n_estimators': 1000,  # fixed, use early stopping
    }

    # Use the SAME expanding window CV as Phase 3
    brier_scores = []
    for train_idx, val_idx in expanding_window_cv(df):
        model = xgb.XGBClassifier(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx],
                  eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                  verbose=False)
        preds = model.predict_proba(X.iloc[val_idx])[:, 1]
        preds = np.clip(preds, 0.05, 0.95)
        brier_scores.append(brier_score_loss(y.iloc[val_idx], preds))

    return np.mean(brier_scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, show_progress_bar=True)
```

## Search Spaces by Model

### XGBoost
| Parameter | Range | Scale |
|-----------|-------|-------|
| max_depth | 3-8 | int |
| learning_rate | 0.01-0.3 | log |
| subsample | 0.6-1.0 | uniform |
| colsample_bytree | 0.5-1.0 | uniform |
| min_child_weight | 1-10 | int |
| reg_alpha | 0.001-10 | log |
| reg_lambda | 0.001-10 | log |

### LightGBM
| Parameter | Range | Scale |
|-----------|-------|-------|
| max_depth | 3-8 | int |
| num_leaves | 15-63 | int |
| learning_rate | 0.01-0.3 | log |
| subsample | 0.6-1.0 | uniform |
| colsample_bytree | 0.5-1.0 | uniform |
| min_child_samples | 5-50 | int |
| reg_alpha | 0.001-10 | log |
| reg_lambda | 0.001-10 | log |

### CatBoost
| Parameter | Range | Scale |
|-----------|-------|-------|
| depth | 3-8 | int |
| learning_rate | 0.01-0.3 | log |
| l2_leaf_reg | 0.1-10 | log |
| bagging_temperature | 0-5 | uniform |
| random_strength | 0-5 | uniform |

## Rules
- ALWAYS use the same CV folds as Phase 3. Never random search with different splits.
- Run M and W tuning separately.
- 50-100 trials is usually enough. Diminishing returns after that.
- Save study results: `study.trials_dataframe().to_csv('artifacts/optuna_results.csv')`
- Log best params to CLAUDE.md Key Decisions section.
- Re-train final model with best params on ALL training data (no validation holdout).

## Quick Win: Ensemble Weight Tuning
Often higher ROI than individual model tuning:
```python
from scipy.optimize import minimize

def neg_brier(weights):
    w = np.array(weights) / sum(weights)
    blended = sum(w[i] * preds[i] for i in range(len(preds)))
    return brier_score_loss(y_val, np.clip(blended, 0.05, 0.95))

result = minimize(neg_brier, x0=[1]*n_models, method='Nelder-Mead')
```
