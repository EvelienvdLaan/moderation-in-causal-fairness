"""Helpers for the binary simulation robustness assessment."""

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .estimation import estimate_reference_specific_effects


N_SPLITS = 5
DR_RANDOM_STATE = 4
PROPENSITY_CLIP = .01
EFFECTS = ('nde_m0', 'nde_m1', 'nie_a0', 'nie_a1', 'interaction')


def fixed_hgb_outcome():
    """Pilot-selected once, then frozen for every reported specification."""
    return HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=300, max_leaf_nodes=15,
        min_samples_leaf=20, max_depth=3, early_stopping=True,
        random_state=DR_RANDOM_STATE,
    )


def fixed_hgb_propensity():
    """Fixed nonlinear sensitivity for P(A | M, C) only."""
    return HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=200, max_leaf_nodes=15,
        min_samples_leaf=20, max_depth=3, early_stopping=True,
        random_state=DR_RANDOM_STATE,
    )


def summarize_result(specification, simulation, result):
    diagnostics = list(result['diagnostics'].values())
    dimensions = [
        dimension
        for diagnostic in diagnostics
        for dimension in diagnostic['mediator_dimensions_by_fold']
    ]
    row = {
        **specification,
        'input_dim': len(simulation['roles']['mediators']),
        'reduced_dim_min': min(dimensions),
        'reduced_dim_max': max(dimensions),
        'outcome_log_loss': np.mean([
            diagnostic['outcome_log_loss'] for diagnostic in diagnostics
        ]),
        'min_ess': min(diagnostic['effective_sample_size'] for diagnostic in diagnostics),
        'max_clipping': max(
            max(diagnostic['fraction_propensity_c_clipped'],
                diagnostic['fraction_propensity_mc_clipped'])
            for diagnostic in diagnostics
        ),
        'propensity_c_roc_auc': diagnostics[0]['propensity_c_roc_auc'],
        'propensity_c_brier_score': diagnostics[0]['propensity_c_brier_score'],
        'propensity_mc_roc_auc': diagnostics[0]['propensity_mc_roc_auc'],
        'propensity_mc_brier_score': diagnostics[0]['propensity_mc_brier_score'],
    }
    for effect in EFFECTS:
        estimate = result['effects'][effect]
        low, high = result['uncertainty'][effect]['confidence_interval']
        truth = simulation['truth']['effects'][effect]
        row.update({
            f'{effect}_estimate': estimate,
            f'{effect}_ci_low': low,
            f'{effect}_ci_high': high,
            f'{effect}_truth': truth,
            f'{effect}_error': estimate - truth,
        })
    row['mean_absolute_effect_error'] = np.mean([
        abs(row[f'{effect}_error']) for effect in EFFECTS
    ])
    return row


def run_specification(simulation, specification):
    roles = simulation['roles']
    options = specification['options']
    with warnings.catch_warnings():
        warnings.simplefilter('always', RuntimeWarning)
        result = estimate_reference_specific_effects(
            simulation['data'], roles['outcome'], roles['protected_attribute'],
            roles['mediators'], roles['confounders'], n_splits=N_SPLITS,
            random_state=DR_RANDOM_STATE, propensity_clip=PROPENSITY_CLIP,
            **options,
        )
    public = {key: value for key, value in specification.items() if key != 'options'}
    return summarize_result(public, simulation, result)


def estimate_ci(row, effect):
    return (
        f"{row[f'{effect}_estimate']:.3f} "
        f"[{row[f'{effect}_ci_low']:.3f}, {row[f'{effect}_ci_high']:.3f}]"
    )
