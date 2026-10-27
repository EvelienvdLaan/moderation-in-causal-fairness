"""Small overlap summaries."""
import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def outcome_model_diagnostics(observed, predicted, *, binary):
    """Summarize cross-fitted outcome-model predictions on held-out folds."""
    observed = np.asarray(observed)
    predicted = np.asarray(predicted, dtype=float)
    if binary:
        labels = predicted >= 0.5
        return {
            "outcome_f1": float(f1_score(observed, labels, zero_division=0)),
            "outcome_roc_auc": float(roc_auc_score(observed, predicted)),
            "outcome_log_loss": float(log_loss(observed, predicted, labels=[0, 1])),
            "outcome_brier_score": float(brier_score_loss(observed, predicted)),
        }
    mse = mean_squared_error(observed, predicted)
    return {
        "outcome_r2": float(r2_score(observed, predicted)),
        "outcome_rmse": float(np.sqrt(mse)),
        "outcome_mae": float(mean_absolute_error(observed, predicted)),
    }


def overlap_diagnostics(propensity_c, propensity_mc, weight, clip, observed=None):
    """Summarize nuisance probabilities and influence-function weights."""
    diagnostics = {
        "clip": float(clip),
        "propensity_c_min": float(np.min(propensity_c)),
        "propensity_c_max": float(np.max(propensity_c)),
        "propensity_mc_min": float(np.min(propensity_mc)),
        "propensity_mc_max": float(np.max(propensity_mc)),
        "fraction_propensity_c_clipped": float(np.mean((propensity_c <= clip) | (propensity_c >= 1-clip))),
        "fraction_propensity_mc_clipped": float(np.mean((propensity_mc <= clip) | (propensity_mc >= 1-clip))),
        "max_absolute_weight": float(np.max(np.abs(weight))),
        "absolute_weight_p95": float(np.quantile(np.abs(weight), .95)),
        "absolute_weight_p99": float(np.quantile(np.abs(weight), .99)),
        "effective_sample_size": float(np.sum(np.abs(weight)) ** 2 / np.sum(weight ** 2)) if np.any(weight) else 0.0,
    }
    if observed is not None:
        diagnostics.update({
            "propensity_c_roc_auc": float(roc_auc_score(observed, propensity_c)),
            "propensity_c_brier_score": float(brier_score_loss(observed, propensity_c)),
            "propensity_mc_roc_auc": float(roc_auc_score(observed, propensity_mc)),
            "propensity_mc_brier_score": float(brier_score_loss(observed, propensity_mc)),
        })
    return diagnostics
