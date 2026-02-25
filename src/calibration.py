"""Probability calibration and clipping utilities."""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from src import config


def clip_predictions(
    preds: np.ndarray,
    low: float = config.CLIP_LOW,
    high: float = config.CLIP_HIGH,
) -> np.ndarray:
    """Clip predictions to [low, high] — mandatory for valid submission.

    Args:
        preds: Array of predicted probabilities.
        low: Lower bound (default 0.05).
        high: Upper bound (default 0.95).

    Returns:
        Clipped array.
    """
    return np.clip(np.asarray(preds, dtype=float), low, high)


def calibrate_platt(
    preds: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """Platt scaling: fit logistic regression on raw predictions.

    Args:
        preds: Raw predicted probabilities (on held-out data).
        y_true: True binary labels.

    Returns:
        Calibrated probabilities.
    """
    preds = np.asarray(preds).reshape(-1, 1)
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(preds, y_true)
    return lr.predict_proba(preds)[:, 1]


def calibrate_isotonic(
    preds: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """Isotonic regression calibration.

    Note: Requires sufficient data (>= 100 samples) to be reliable.

    Args:
        preds: Raw predicted probabilities (on held-out data).
        y_true: True binary labels.

    Returns:
        Calibrated probabilities.
    """
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(preds, y_true)
    return iso.predict(preds)


def reliability_diagram(
    preds: np.ndarray,
    y_true: np.ndarray,
    save_path: Optional[Path] = None,
    n_bins: int = 10,
    title: str = 'Reliability Diagram',
) -> None:
    """Generate and optionally save a calibration reliability diagram.

    Args:
        preds: Predicted probabilities.
        y_true: True binary labels.
        save_path: If provided, save figure to this path.
        n_bins: Number of probability bins.
        title: Plot title.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fraction_of_positives = []
    mean_predicted = []

    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (preds >= low) & (preds < high)
        if mask.sum() > 0:
            fraction_of_positives.append(y_true[mask].mean())
            mean_predicted.append(preds[mask].mean())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(mean_predicted, fraction_of_positives, 's-', label='Model')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    axes[1].hist(preds, bins=20, edgecolor='black')
    axes[1].set_title('Prediction Distribution')
    axes[1].set_xlabel('Predicted probability')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
