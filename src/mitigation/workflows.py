"""Functional helpers for fitting and evaluating mitigation strategies."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from estimate_effects import estimate_reference_specific_effects

from .inprocessing import DoublyRobustInprocessor


def fit_path_specific_predictor(
        train, *, outcome="Y", protected_attribute="A", mediators,
        confounders=(), outcome_type="binary", penalty_weights=None, **model_kwargs):
    """Fit one path-specific predictor and return an explicit result mapping."""
    estimator = DoublyRobustInprocessor(
        outcome_type=outcome_type,
        penalty_weights={} if penalty_weights is None else penalty_weights,
        **model_kwargs,
    ).fit(train, outcome, protected_attribute, mediators, confounders)
    return {
        "estimator": estimator,
        "training_surrogate": {
            "nested_means": dict(estimator.result_.nested_means),
            "effects": dict(estimator.result_.effects),
        },
        "optimization": {
            "converged": estimator.result_.converged,
            "message": estimator.result_.message,
            "objective": estimator.result_.objective,
            "prediction_loss": estimator.result_.prediction_loss,
        },
        "roles": {
            "outcome": outcome,
            "protected_attribute": protected_attribute,
            "mediators": tuple(mediators),
            "confounders": tuple(confounders),
        },
    }


def predict_path_specific_model(fitted, data):
    """Predict from a functional path-specific fit result."""
    estimator = fitted["estimator"] if isinstance(fitted, dict) else fitted
    return estimator.predict(data)


def prediction_error(observed, predicted, outcome_type):
    """Return log loss for binary outcomes or RMSE for continuous outcomes."""
    if outcome_type == "binary":
        return float(log_loss(observed, predicted, labels=[0, 1]))
    return float(mean_squared_error(observed, predicted) ** .5)


def predictive_performance(model, data, *, outcome="Y", protected_attribute="A",
                           outcome_type="binary"):
    """Report overall and group-specific predictive performance."""
    predicted = predict_path_specific_model(model, data)
    rows = []
    scopes = [("overall", np.ones(len(data), dtype=bool))]
    scopes += [(f"A={group}", data[protected_attribute].to_numpy() == group) for group in (0, 1)]
    for scope, mask in scopes:
        observed, values = data.loc[mask, outcome].to_numpy(), predicted[mask]
        row = {"scope": scope, "n": int(mask.sum())}
        if outcome_type == "binary":
            row.update({
                "accuracy": accuracy_score(observed, values >= .5),
                "auroc": roc_auc_score(observed, values) if len(np.unique(observed)) == 2 else np.nan,
                "log_loss": log_loss(observed, values, labels=[0, 1]),
            })
        else:
            row.update({
                "rmse": mean_squared_error(observed, values) ** .5,
                "mae": mean_absolute_error(observed, values),
                "r2": r2_score(observed, values),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def performance_by_strategy(models, test, *, outcome="Y", protected_attribute="A",
                            outcome_type="binary"):
    """Stack predictive metrics for fitted mitigation strategies."""
    frames = []
    for strategy, model in models.items():
        frame = predictive_performance(
            model, test, outcome=outcome, protected_attribute=protected_attribute,
            outcome_type=outcome_type,
        )
        frame.insert(0, "strategy", strategy)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def mitigation_specifications(group0="A=0", group1="A=1"):
    """Return standard direct- and indirect-effect penalty specifications."""
    direct = {
        "No mitigation": (),
        f"NDE at {group0} M(0)": ("nde_m0",),
        f"NDE at {group1} M(1)": ("nde_m1",),
        "Symmetric NDE": ("symmetric_nde",),
        "Both NDEs": ("nde_m0", "nde_m1"),
    }
    indirect = {
        "No mitigation": (),
        f"NIE while {group0}": ("nie_a0",),
        f"NIE while {group1}": ("nie_a1",),
        "Symmetric NIE": ("symmetric_nie",),
        "Both NIEs": ("nie_a0", "nie_a1"),
    }
    return direct, indirect


def mitigation_simulation_specifications():
    """Return the reference-specific strategies used in the simulation."""
    direct = {
        "No mitigation": (),
        "Penalize NDE at M(0)": ("nde_m0",),
        "Penalize NDE at M(1)": ("nde_m1",),
        "Penalize symmetric NDE": ("symmetric_nde",),
        "Penalize both NDEs": ("nde_m0", "nde_m1"),
    }
    indirect = {
        "No mitigation": (),
        "Penalize NIE at A=0": ("nie_a0",),
        "Penalize NIE at A=1": ("nie_a1",),
        "Penalize symmetric NIE": ("symmetric_nie",),
        "Penalize both NIEs": ("nie_a0", "nie_a1"),
    }
    return direct, indirect


def estimate_prediction_effects(model, data, outcome, protected_attribute, mediators,
                                confounders=(), **estimator_kwargs):
    """Estimate model-output effects with the shared doubly robust estimator."""
    evaluation = data.copy()
    prediction_column = "__model_prediction__"
    evaluation[prediction_column] = predict_path_specific_model(model, data)
    return estimate_reference_specific_effects(
        evaluation, prediction_column, protected_attribute, mediators, confounders,
        **estimator_kwargs,
    )


def fit_mitigation_strategies(train, test, *, outcome, protected_attribute,
                              mediators, confounders=(), specifications,
                              outcome_type="binary", penalty_strength=100,
                              estimator_kwargs=None, **model_kwargs):
    """Fit strategies and evaluate every effect with one DR method."""
    estimator_kwargs = {} if estimator_kwargs is None else dict(estimator_kwargs)
    models, rows, evaluations = {}, [], {}
    for label, targets in specifications.items():
        model = fit_path_specific_predictor(
            train, outcome=outcome, protected_attribute=protected_attribute,
            mediators=mediators, confounders=confounders,
            outcome_type=outcome_type,
            penalty_weights={target: penalty_strength for target in targets},
            **model_kwargs,
        )
        evaluation = estimate_prediction_effects(
            model, test, outcome, protected_attribute, mediators, confounders,
            **estimator_kwargs,
        )
        predicted = predict_path_specific_model(model, test)
        models[label] = model
        evaluations[label] = evaluation
        row = {
            "model": label,
            **evaluation["nested_means"],
            **evaluation["effects"],
            "prediction_error": prediction_error(test[outcome], predicted, outcome_type),
            "converged": model["optimization"]["converged"],
        }
        for effect, uncertainty in evaluation["uncertainty"].items():
            lower, upper = uncertainty["confidence_interval"]
            row.update({
                f"{effect}_se": uncertainty["standard_error"],
                f"{effect}_lower": lower,
                f"{effect}_upper": upper,
            })
        rows.append(row)
    return models, pd.DataFrame(rows).set_index("model"), evaluations


def run_mitigation_analysis(train, test, *, mediators, confounders=(),
                            direct_specifications, indirect_specifications,
                            outcome="Y", protected_attribute="A",
                            outcome_type="binary", penalty_strength=100,
                            estimator_kwargs=None, **model_kwargs):
    """Fit direct and indirect strategies and collect effects and metrics."""
    common = dict(
        outcome=outcome,
        protected_attribute=protected_attribute,
        mediators=mediators,
        confounders=confounders,
        outcome_type=outcome_type,
        penalty_strength=penalty_strength,
        estimator_kwargs=estimator_kwargs,
        **model_kwargs,
    )
    specifications = dict(direct_specifications)
    specifications.update(indirect_specifications)
    models, results, evaluations = fit_mitigation_strategies(
        train, test, specifications=specifications, **common
    )
    direct_results = results.loc[list(direct_specifications)].copy()
    indirect_results = results.loc[list(indirect_specifications)].copy()
    performance = performance_by_strategy(
        models, test, outcome=outcome, protected_attribute=protected_attribute,
        outcome_type=outcome_type,
    ).drop_duplicates()
    return {
        "direct": direct_results,
        "indirect": indirect_results,
        "performance": performance,
        "models": models,
        "held_out_evaluations": evaluations,
    }


def run_mitigation_simulation(train_dataset, test_dataset, *, penalty_strength=200,
                              estimator_kwargs=None, **model_kwargs):
    """Run the standard mitigation strategies on two dataset contracts."""
    train_roles, test_roles = train_dataset["roles"], test_dataset["roles"]
    if train_roles != test_roles:
        raise ValueError("Training and test variable roles must match")
    direct, indirect = mitigation_simulation_specifications()
    return run_mitigation_analysis(
        train_dataset["data"], test_dataset["data"],
        outcome=train_roles["outcome"],
        protected_attribute=train_roles["protected_attribute"],
        mediators=train_roles["mediators"],
        confounders=train_roles["confounders"],
        direct_specifications=direct,
        indirect_specifications=indirect,
        outcome_type="binary",
        penalty_strength=penalty_strength,
        estimator_kwargs=estimator_kwargs,
        **model_kwargs,
    )


def effect_results_frame(results, effects):
    """Return tidy held-out effects with standard errors and intervals."""
    rows = []
    for strategy, row in results.iterrows():
        for effect in effects:
            rows.append({
                "strategy": strategy,
                "effect": effect,
                "estimate": row[effect],
                "standard_error": row[f"{effect}_se"],
                "lower": row[f"{effect}_lower"],
                "upper": row[f"{effect}_upper"],
            })
    return pd.DataFrame(rows)


def overlap_summary(evaluations):
    """Summarize worst held-out overlap diagnostics by strategy."""
    rows = []
    for strategy, evaluation in evaluations.items():
        diagnostics = evaluation["diagnostics"]
        rows.append({
            "strategy": strategy,
            "max_propensity_c_clipped": max(
                value["fraction_propensity_c_clipped"] for value in diagnostics.values()
            ),
            "max_propensity_mc_clipped": max(
                value["fraction_propensity_mc_clipped"] for value in diagnostics.values()
            ),
            "max_absolute_weight": max(
                value["max_absolute_weight"] for value in diagnostics.values()
            ),
            "effective_sample_size": min(
                value["effective_sample_size"] for value in diagnostics.values()
            ),
        })
    return pd.DataFrame(rows)


def plot_strategy_effects(
    results, effects, performance, *,
    metric="log_loss", title=None, ylabel="Estimated effect"
):
    """Plot reference-specific effects above an aligned performance-cost strip.

    Loss metrics are expressed as percentage increase relative to the
    unmitigated model. For higher-is-better metrics, the strip instead reports
    the decrease in percentage points.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Keep only overall performance and match it to results.index.
    perf = (
        performance.loc[performance["scope"].eq("overall")]
        .set_index("strategy")[metric]
        .reindex(results.index)
    )
    if perf.isna().any():
        missing = perf.index[perf.isna()].tolist()
        raise ValueError(f"Missing {metric} for strategies: {missing}")

    x = np.arange(len(results))
    width = .36
    fig, (ax, ax_perf) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True,
        gridspec_kw={"height_ratios": (3.2, 1), "hspace": .08},
        constrained_layout=True,
    )

    for offset, effect in zip((-.5, .5), effects):
        values = results[effect].to_numpy()
        lower = results.get(f"{effect}_lower")
        upper = results.get(f"{effect}_upper")

        yerr = None
        if lower is not None and upper is not None:
            yerr = np.vstack([
                values - lower.to_numpy(),
                upper.to_numpy() - values,
            ])

        ax.bar(
            x + offset * width, values, width,
            yerr=yerr, capsize=3, label=effect
        )

    ax.axhline(0, color="black", linewidth=.8)
    ax.set(ylabel=ylabel, title=title)
    ax.grid(axis="y", color="#E6E6E6", linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    baseline = perf.loc["No mitigation"]
    if metric in {"log_loss", "rmse", "mae"}:
        performance_cost = 100 * (perf / baseline - 1)
        performance_label = "Predictive loss increase (%)"
    else:
        performance_cost = 100 * (baseline - perf)
        performance_label = f"{metric.upper()} decrease (percentage points)"
    colors = np.where(performance_cost.to_numpy() > 0, "#D55E00", "#009E73")
    ax_perf.axhline(0, color="black", linewidth=.8)
    ax_perf.plot(x, performance_cost, color="#777777", linewidth=1, zorder=1)
    ax_perf.scatter(x, performance_cost, color=colors, s=35, zorder=2)
    ax_perf.set_ylabel(performance_label)
    ax_perf.set_xticks(x, results.index, rotation=15, ha="right")
    ax_perf.grid(axis="y", color="#E6E6E6", linewidth=.7)
    ax_perf.set_axisbelow(True)
    ax_perf.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best")

    return fig

