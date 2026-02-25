"""Run full ensemble CV for both M and W, save results and weights."""
import json
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

from src.ensemble import (
    run_all_models_cv,
    simple_average_ensemble,
    optimize_ensemble_weights,
    weighted_ensemble,
    log_ensemble_results,
)
from src.cv import evaluate_brier


def main():
    # Load feature matrices
    df_m = pd.read_csv('artifacts/features_men.csv')
    df_w = pd.read_csv('artifacts/features_women.csv')
    feature_cols = [c for c in df_m.columns if c.endswith('_diff')]

    print(f"Men: {df_m.shape}, Women: {df_w.shape}")
    print(f"Features: {len(feature_cols)}")
    print()

    # === Men ===
    print("=== Running Men's Ensemble CV ===")
    results_m = run_all_models_cv(df_m, feature_cols, gender='M')
    print()

    for name, data in results_m.items():
        print(f"  {name}: {data['mean_brier']:.4f} ± {data['std_brier']:.4f}")

    # Simple average
    oof_dict_m = {n: results_m[n]['oof_preds'] for n in results_m}
    y_true_m = results_m['logistic']['oof_targets']
    avg_m = simple_average_ensemble(oof_dict_m)
    print(f"  simple_avg: {evaluate_brier(y_true_m, avg_m):.4f}")

    # Optimize weights
    weights_m = optimize_ensemble_weights(oof_dict_m, y_true_m)
    wtd_m = weighted_ensemble(oof_dict_m, weights_m)
    print(f"  weighted: {evaluate_brier(y_true_m, wtd_m):.4f}")
    print(f"  weights: {weights_m}")
    print()

    # === Women ===
    print("=== Running Women's Ensemble CV ===")
    results_w = run_all_models_cv(df_w, feature_cols, gender='W')
    print()

    for name, data in results_w.items():
        print(f"  {name}: {data['mean_brier']:.4f} ± {data['std_brier']:.4f}")

    oof_dict_w = {n: results_w[n]['oof_preds'] for n in results_w}
    y_true_w = results_w['logistic']['oof_targets']
    avg_w = simple_average_ensemble(oof_dict_w)
    print(f"  simple_avg: {evaluate_brier(y_true_w, avg_w):.4f}")

    weights_w = optimize_ensemble_weights(oof_dict_w, y_true_w)
    wtd_w = weighted_ensemble(oof_dict_w, weights_w)
    print(f"  weighted: {evaluate_brier(y_true_w, wtd_w):.4f}")
    print(f"  weights: {weights_w}")
    print()

    # === Save results ===
    log_ensemble_results(results_m, results_w, weights_m, weights_w)

    # Save weights
    weights_out = {'M': weights_m, 'W': weights_w}
    with open('artifacts/ensemble_weights.json', 'w') as f:
        json.dump(weights_out, f, indent=2)
    print("Saved artifacts/ensemble_weights.json")
    print("Saved artifacts/ensemble_results.txt")


if __name__ == '__main__':
    main()
