"""Reference-specific natural direct and indirect effects."""

from .effects import effects_from_nested_means
from .estimation import NestedMeanEstimate, estimate_nested_mean, estimate_reference_specific_effects
from .reporting import (
    diagnostic_table,
    effect_table,
    plot_binary_reference_curves,
    plot_binary_scale_comparison,
    plot_estimated_vs_true_effects,
    plot_feature_group_curves,
    simulation_result_table,
    plot_nonadditivity_mechanisms,
    plot_real_data_effects,
    robustness_metadata,
    robustness_row,
)
from .simulation_robustness import (
    DR_RANDOM_STATE,
    EFFECTS,
    N_SPLITS,
    estimate_ci,
    fixed_hgb_outcome,
    fixed_hgb_propensity,
    run_specification,
)

__all__ = [
    "NestedMeanEstimate",
    "effects_from_nested_means",
    "estimate_nested_mean",
    "estimate_reference_specific_effects",
    "effect_table",
    "diagnostic_table",
    "plot_binary_reference_curves",
    "plot_binary_scale_comparison",
    "plot_estimated_vs_true_effects",
    "plot_feature_group_curves",
    "simulation_result_table",
    "plot_nonadditivity_mechanisms",
    "plot_real_data_effects",
    "robustness_metadata",
    "robustness_row",
    "DR_RANDOM_STATE",
    "EFFECTS",
    "N_SPLITS",
    "estimate_ci",
    "fixed_hgb_outcome",
    "fixed_hgb_propensity",
    "run_specification",
]
