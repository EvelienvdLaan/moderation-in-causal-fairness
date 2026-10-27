"""Optional mitigation methods kept separate from causal-effect estimation."""

from .inprocessing import DoublyRobustInprocessor
from .reporting import (
    mitigation_compact_overview_table,
    mitigation_overview_table,
    plot_mitigation_summary_figure,
)
from .workflows import (
    effect_results_frame,
    estimate_prediction_effects,
    fit_path_specific_predictor,
    fit_mitigation_strategies,
    mitigation_specifications,
    mitigation_simulation_specifications,
    overlap_summary,
    performance_by_strategy,
    plot_strategy_effects,
    predict_path_specific_model,
    predictive_performance,
    prediction_error,
    run_mitigation_analysis,
    run_mitigation_simulation,
)

__all__ = [
    "DoublyRobustInprocessor",
    "effect_results_frame",
    "estimate_prediction_effects",
    "fit_path_specific_predictor",
    "fit_mitigation_strategies",
    "mitigation_compact_overview_table",
    "mitigation_overview_table",
    "mitigation_specifications",
    "mitigation_simulation_specifications",
    "overlap_summary",
    "performance_by_strategy",
    "plot_mitigation_summary_figure",
    "plot_strategy_effects",
    "predict_path_specific_model",
    "predictive_performance",
    "prediction_error",
    "run_mitigation_analysis",
    "run_mitigation_simulation",
]
