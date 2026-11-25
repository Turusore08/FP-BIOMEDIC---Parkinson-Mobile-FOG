"""Shared post-processing and binary classification evaluation metrics for Mobile-FOG."""

from __future__ import annotations

import numpy as np


def apply_min_duration_filter(
    y_pred: np.ndarray,
    trial_ids: np.ndarray,
    min_run: int = 2,
) -> np.ndarray:
    """Remove isolated positive FoG predictions shorter than `min_run` windows per trial."""
    out = y_pred.copy()
    for tid in np.unique(trial_ids):
        idx = np.where(trial_ids == tid)[0]
        seq = out[idx]
        n = len(seq)
        i = 0
        while i < n:
            if seq[i] == 1:
                j = i
                while j < n and seq[j] == 1:
                    j += 1
                if (j - i) < min_run:
                    seq[i:j] = 0
                i = j
            else:
                i += 1
        out[idx] = seq
    return out


def compute_roc_pr_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]:
    """Compute exact ROC curve, ROC-AUC, PR curve, and Average Precision (PR-AUC)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    order = np.argsort(-y_score, kind="mergesort")
    y_s = y_true[order]
    sc_s = y_score[order]
    diff_idx = np.where(np.diff(sc_s) != 0)[0]
    thr_idx = np.r_[diff_idx, len(y_s) - 1]

    tps = np.cumsum(y_s == 1)[thr_idx]
    fps = np.cumsum(y_s == 0)[thr_idx]

    total_pos = max(int(tps[-1]), 1)
    total_neg = max(int(fps[-1]), 1)
    tpr = np.r_[0.0, tps / total_pos]
    fpr = np.r_[0.0, fps / total_neg]
    trap_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    roc_auc = float(trap_fn(tpr, fpr))

    prec = tps / np.maximum(tps + fps, 1)
    rec = tps / total_pos
    pr_auc = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    rec_plot = np.r_[0.0, rec]
    prec_plot = np.r_[1.0, prec]
    return fpr, tpr, roc_auc, rec_plot, prec_plot, pr_auc


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, object]:
    """Compute confusion matrix and binary/macro clinical metrics from hard predictions."""
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)

    tp = int(((y_p == 1) & (y_t == 1)).sum())
    tn = int(((y_p == 0) & (y_t == 0)).sum())
    fp = int(((y_p == 1) & (y_t == 0)).sum())
    fn = int(((y_p == 0) & (y_t == 1)).sum())

    prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = (2 * prec1 * rec1 / (prec1 + rec1)) if (prec1 + rec1) > 0 else 0.0

    prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = (2 * prec0 * rec0 / (prec0 + rec0)) if (prec0 + rec0) > 0 else 0.0

    n = max(len(y_t), 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "cm": np.array([[tn, fp], [fn, tp]], dtype=int),
        "macro_prec": 0.5 * (prec1 + prec0),
        "macro_rec": 0.5 * (rec1 + rec0),
        "spec": rec0,
        "sens": rec1,
        "macro_f1": 0.5 * (f1_1 + f1_0),
        "f1_fog": f1_1,
        "acc": (tp + tn) / n,
        "err_rate": (fp + fn) / n,
    }
