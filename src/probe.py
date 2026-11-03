"""
src/probe.py

The linear-probe fit shared by the baseline (E1) and the random split (E5), so
both use the same procedure: class-weighted logistic regression, one fit per C,
keeping the best by validation macro-F1.
"""

import logging
import time

import numpy as np
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression

from src import metrics

logger = logging.getLogger(__name__)


def fit_and_select(
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    labels: list[int], probe: DictConfig,
) -> tuple[LogisticRegression, list[dict]]:
    """Fit one probe per C on train; return the best by val macro-F1, and the sweep."""
    sweep, best, best_f1 = [], None, -1.0
    for c in probe.C_grid:
        start = time.time()
        clf = LogisticRegression(
            C=float(c),
            class_weight=probe.class_weight,
            max_iter=probe.max_iter,
            random_state=probe.seed,
        )
        clf.fit(x_train, y_train)
        val_f1 = metrics.macro_f1(y_val, clf.predict(x_val), labels)
        sweep.append({
            "C": float(c),
            "val_macro_f1": val_f1,
            "n_iter": int(np.max(clf.n_iter_)),
            "converged": bool(np.max(clf.n_iter_) < probe.max_iter),
            "fit_seconds": time.time() - start,
        })
        logger.info(f"  C={c:<8} val macro-F1 {val_f1:.4f} ({sweep[-1]['fit_seconds']:.0f} s)")
        if val_f1 > best_f1:
            best, best_f1 = clf, val_f1
    return best, sweep
