"""Small table and plotting helpers shared by the notebooks."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


EFFECT_ORDER = ("nde_m0", "nde_m1", "nie_a0", "nie_a1", "interaction", "total_effect")
EFFECT_LABELS = {
    "nde_m0": "NDE at M(0)",
    "nde_m1": "NDE at M(1)",
    "nie_a0": "NIE while A=0",
    "nie_a1": "NIE while A=1",
    "interaction": "Interaction",
    "total_effect": "Total effect",
}
SIMULATION_ORDER = (
    "psi_00", "psi_01", "psi_10", "psi_11",
    "nde_m0", "nde_m1", "nie_a0", "nie_a1", "interaction", "total_effect",
)

_ROBUSTNESS_CHECK_NUMBER = {
    "primary": 1,
    "outcome_alternative": 2,
    "propensity_mc_hgb": 3,
    "pca60": 4,
    "pca90": 5,
    "pfi20": 6,
}
_ROBUSTNESS_DATASET_ID = {
    "Law": "LAW",
    "COMPAS": "COMPAS",
    "SCUT": "SCUT",
    "Reddit": "REDDIT",
}
_ROBUSTNESS_MODEL_LABELS = {
    "linear_with_interactions": "Linear + A interactions",
    "hgb_with_interactions": "HGB + A interactions",
    "hist_gradient_boosting": "HGB",
    "logistic": "Logistic",
}
_ROBUSTNESS_REDUCTION_LABELS = {
    "none": "None",
    "pca70": "PCA 70%",
    "pca60": "PCA 60%",
    "pca90": "PCA 90%",
    "pfi20": "PFI 20%",
}
_ROBUSTNESS_REDUCTION_CHECKS = frozenset({"pca60", "pca90", "pfi20"})


def plot_nonadditivity_mechanisms(m0=-2.5, m1=-0.5):
    colors = {
        0: "#0072B2",
        1: "#D55E00",
    }

    grid = np.linspace(m0 - 1.2, m1 + 1.2, 400)

    panels = [
        (
            "(a) Additive linear model",
            lambda a, m: 4 * a + 2 * m,
        ),
        (
            "(b) Linear model with interaction",
            lambda a, m: 4 * a + 2 * m + 1.5 * a * m,
        ),
        (
            "(c) Nonlinear link",
            lambda a, m: 1 / (1 + np.exp(-(4 * a + 2 * m))),
        ),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19, 6.5),
    )

    for ax, (title, response) in zip(axes, panels):
        curves = {
            a: response(a, grid)
            for a in (0, 1)
        }

        values = {
            (a, r): float(response(a, m))
            for a in (0, 1)
            for r, m in enumerate((m0, m1))
        }

        lower = min(curve.min() for curve in curves.values())
        upper = max(curve.max() for curve in curves.values())
        span = upper - lower

        # Main outcome curves.
        for a in (0, 1):
            ax.plot(
                grid,
                curves[a],
                color=colors[a],
                linewidth=3,
                zorder=2,
            )

            ax.scatter(
                [m0, m1],
                [values[a, 0], values[a, 1]],
                color=colors[a],
                s=55,
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
            )

        # Light reference lines for M(0) and M(1).
        for m in (m0, m1):
            ax.axvline(
                m,
                color="0.78",
                linestyle="--",
                linewidth=1.2,
                zorder=0,
            )

        # Direct effects.
        for r, m in enumerate((m0, m1)):
            y0 = values[0, r]
            y1 = values[1, r]

            ax.annotate(
                "",
                xy=(m, y1),
                xytext=(m, y0),
                arrowprops=dict(
                    arrowstyle="<->",
                    color="0.25",
                    linewidth=1.8,
                    shrinkA=6,
                    shrinkB=6,
                    mutation_scale=16,
                ),
                zorder=4,
            )

            offset = 10 if r == 0 else -10

            ax.annotate(
                rf"$\mathrm{{NDE}}_{{M({r})}}$",
                xy=(m, (y0 + y1) / 2),
                xytext=(offset, 0),
                textcoords="offset points",
                ha="left" if r == 0 else "right",
                va="center",
                fontsize=15,
                color="0.2",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85,
                    pad=1.5,
                ),
                zorder=6,
            )

        # Indirect effects:
        # blue on the right and orange on the left.
        nie_positions = {
            0: m1 + 0.8,
            1: m0 - 0.8,
        }

        for a in (0, 1):
            arrow_x = nie_positions[a]
            y0 = values[a, 0]
            y1 = values[a, 1]

            # Subtle horizontal construction lines.
            for m, y in ((m0, y0), (m1, y1)):
                ax.plot(
                    [m, arrow_x],
                    [y, y],
                    color=colors[a],
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.45,
                    zorder=1,
                )

            ax.annotate(
                "",
                xy=(arrow_x, y1),
                xytext=(arrow_x, y0),
                arrowprops=dict(
                    arrowstyle="<->",
                    color=colors[a],
                    linestyle=":",
                    linewidth=2,
                    mutation_scale=17,
                ),
                zorder=4,
            )

            label_offset = 11 if a == 0 else -11

            ax.annotate(
                rf"$\mathrm{{NIE}}_{{A={a}}}$",
                xy=(arrow_x, (y0 + y1) / 2),
                xytext=(label_offset, 0),
                textcoords="offset points",
                rotation=90,
                ha="center",
                va="center",
                color=colors[a],
                fontsize=15,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85,
                    pad=1.5,
                ),
                zorder=6,
            )

        ax.set_title(
            title,
            fontsize=18,
            pad=16,
        )

        ax.set_xticks([m0, m1])
        ax.set_xticklabels(
            [r"$M(0)$", r"$M(1)$"],
            fontsize=16,
        )

        ax.tick_params(
            axis="y",
            labelsize=14,
            length=4,
        )

        ax.tick_params(
            axis="x",
            length=0,
            pad=8,
        )

        ax.set_xlim(grid.min(), grid.max())
        ax.set_ylim(
            lower - 0.08 * span,
            upper + 0.08 * span,
        )

        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("0.35")
        ax.grid(False)

    # Shared axis labels avoid repeating the same text three times.
    fig.supxlabel(
        "Mediator $M$",
        fontsize=19,
        y=0.11,
    )

    fig.supylabel(
        "Outcome",
        fontsize=19,
        x=0.01,
    )

    # A small legend showing only the group colors.
    legend = [
        Line2D(
            [0], [0],
            color=colors[0],
            lw=3,
            label="$A=0$",
        ),
        Line2D(
            [0], [0],
            color=colors[1],
            lw=3,
            label="$A=1$",
        ),
        Line2D(
            [0], [0],
            color="0.25",
            lw=2,
            linestyle="-",
            label="Direct effect (NDE)",
        ),
        Line2D(
            [0], [0],
            color="0.4",
            lw=2,
            linestyle=":",
            label="Indirect effect (NIE)",
        ),
    ]

    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=15,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=2.5,
        columnspacing=2.2,
    )

    fig.tight_layout(
        rect=(0.025, 0.14, 1, 1),
        w_pad=2.5,
    )

    return fig, axes


def plot_feature_group_curves(
        frame, outcome, *, features, continuous_features=None, title,
        protected_attribute="A", group_labels=None, outcome_label="Mean outcome",
        n_bins=8, n_columns=3):
    """Plot descriptive outcome curves across features by protected group."""
    import matplotlib.pyplot as plt

    features = list(features)
    continuous_features = set(continuous_features or [])
    required = {protected_attribute, outcome, *features}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing curve-plot columns: {missing}")
    if not features:
        raise ValueError("features must contain at least one column")
    unknown_continuous = sorted(continuous_features - set(features))
    if unknown_continuous:
        raise ValueError(
            f"continuous_features not present in features: {unknown_continuous}"
        )
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if n_columns < 1:
        raise ValueError("n_columns must be at least 1")

    groups = list(pd.unique(frame[protected_attribute].dropna()))
    try:
        groups.sort()
    except TypeError:
        pass
    labels = dict(group_labels or {group: str(group) for group in groups})
    missing_labels = [group for group in groups if group not in labels]
    if missing_labels:
        raise ValueError(f"Missing labels for protected groups: {missing_labels}")
    n_rows = int(np.ceil(len(features) / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(5*n_columns, 4*n_rows),
        sharey=True, squeeze=False,
    )
    for ax, feature in zip(axes.flat, features):
        panel = frame[[protected_attribute, feature, outcome]].dropna().copy()
        if feature in continuous_features:
            panel["bin"] = pd.qcut(panel[feature], q=n_bins, duplicates="drop")
            grouped = panel.groupby([protected_attribute, "bin"], observed=True).agg(
                x=(feature, "mean"), y=(outcome, "mean"), n=(outcome, "size")
            ).reset_index()
        else:
            grouped = panel.groupby([protected_attribute, feature], observed=True).agg(
                y=(outcome, "mean"), n=(outcome, "size")
            ).reset_index().rename(columns={feature: "x"})
        for group_index, group in enumerate(groups):
            line = grouped[grouped[protected_attribute] == group].sort_values("x")
            ax.plot(
                line.x, line.y, marker="o", linewidth=1.6,
                color=f"C{group_index}", label=labels[group],
            )
        ax.set(xlabel=feature, title=feature)
        ax.grid(alpha=.2)
    for ax in axes.flat[len(features):]:
        fig.delaxes(ax)
    for ax in axes[:, 0]:
        if ax in fig.axes:
            ax.set_ylabel(outcome_label)
    axes.flat[0].legend()
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def effect_table(result, dataset=None, effects=EFFECT_ORDER):
    """Return estimates and 95% confidence intervals in tidy form."""
    rows = []
    for effect in effects:
        uncertainty = result["uncertainty"][effect]
        low, high = uncertainty["confidence_interval"]
        rows.append({
            "dataset": dataset,
            "effect": effect,
            "estimate": result["effects"][effect],
            "standard_error": uncertainty["standard_error"],
            "ci_low": low,
            "ci_high": high,
        })
    return pd.DataFrame(rows)


def diagnostic_table(result, dataset=None):
    """Return effective sample size and clipping diagnostics in tidy form."""
    frame = pd.DataFrame(result["diagnostics"]).T.rename_axis("nested_mean").reset_index()
    frame.insert(0, "dataset", dataset)
    return frame


def simulation_result_table(result, truth, keys=SIMULATION_ORDER):
    """Combine DR estimates, known truth, and effect confidence intervals."""
    estimates = {**result["nested_means"], **result["effects"]}
    true_values = {**truth["nested_means"], **truth["effects"]}
    rows = []
    for key in keys:
        row = {"quantity": key, "estimate": estimates[key], "truth": true_values[key]}
        if key in result["uncertainty"]:
            low, high = result["uncertainty"][key]["confidence_interval"]
            row.update({"ci_low": low, "ci_high": high})
        else:
            row.update({"ci_low": np.nan, "ci_high": np.nan})
        rows.append(row)
    return pd.DataFrame(rows).set_index("quantity")


def plot_estimated_vs_true_effects(panels, *, outcome_label="Effect on outcome scale"):
    """Plot estimated effects with confidence intervals and known truth."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5*len(panels), 4.5), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        table, keys, labels, title = panel
        values = table.loc[keys, "estimate"].to_numpy(dtype=float)
        low = table.loc[keys, "ci_low"].to_numpy(dtype=float)
        high = table.loc[keys, "ci_high"].to_numpy(dtype=float)
        ax.bar(labels, values, yerr=np.vstack([values-low, high-values]), capsize=5, alpha=.8)
        ax.scatter(labels, table.loc[keys, "truth"], color="black", marker="x",
                   s=65, label="Truth")
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_title(title)
        ax.legend()
    axes[0, 0].set_ylabel(outcome_label)
    fig.tight_layout()
    return fig


def _mediator_plot_values(simulation):
    """Return first mediator, exposure, observed group means, and parameters."""
    data, roles = simulation["data"], simulation["roles"]
    mediator = data[roles["mediators"][0]].to_numpy(dtype=float)
    exposure = data[roles["protected_attribute"]].to_numpy(dtype=int)
    means = {group: float(mediator[exposure == group].mean()) for group in (0, 1)}
    return mediator, exposure, means, simulation["provenance"]["parameters"]


def plot_binary_reference_curves(simulation, table, *, n_rug=220, random_state=31):
    """Plot binary response curves and estimated reference-specific NIEs."""
    import matplotlib.pyplot as plt

    mediator, exposure, means, parameters = _mediator_plot_values(simulation)
    grid = np.linspace(mediator.min()-.5, mediator.max()+.5, 500)
    b_a, b_m = parameters["exposure_logit_effect"], parameters["mediator_logit_effect"]
    curves = {a: 1/(1+np.exp(-(b_a*a+b_m*grid))) for a in (0, 1)}
    rng = np.random.default_rng(random_state)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for a in (0, 1):
        axes[0].plot(grid, curves[a], label=f"A={a} response curve")
        axes[0].axvline(means[a], linestyle="--" if a == 0 else ":", color="grey",
                        label=f"Mean observed M | A={a}")
        index = rng.choice(np.flatnonzero(exposure == a), n_rug, replace=False)
        axes[0].scatter(mediator[index], np.full(n_rug, -.055-.05*a), marker="|", alpha=.45,
                        label=f"Observed M, A={a}")
        y0 = 1/(1+np.exp(-(b_a*a+b_m*means[0])))
        y1 = 1/(1+np.exp(-(b_a*a+b_m*means[1])))
        axes[0].annotate("", xy=(means[1], y1), xytext=(means[0], y0),
                         arrowprops={"arrowstyle": "->", "lw": 3, "color": f"C{a}"})
    axes[0].set(xlabel="Mediator M", ylabel="Outcome probability",
                title="Same mediator shift, different sigmoid regions")
    axes[0].set_ylim(-.14, 1.03)
    axes[0].legend(fontsize=8)
    keys, labels = ["nie_a0", "nie_a1"], ["NIE while A=0", "NIE while A=1"]
    values = table.loc[keys, "estimate"].to_numpy()
    low, high = table.loc[keys, "ci_low"].to_numpy(), table.loc[keys, "ci_high"].to_numpy()
    axes[1].bar(labels, values, yerr=np.vstack([values-low, high-values]), capsize=5, alpha=.8)
    axes[1].scatter(labels, table.loc[keys, "truth"], color="black", marker="x", s=65,
                    label="Truth")
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set(ylabel="Effect on probability scale", title="Estimated reference-specific NIEs")
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_binary_scale_comparison(simulation, *, n_rug=220, random_state=31):
    """Compare additive log-odds and expit curves with observed mediator rugs."""
    import matplotlib.pyplot as plt

    mediator, exposure, means, parameters = _mediator_plot_values(simulation)
    grid = np.linspace(mediator.min()-.5, mediator.max()+.5, 500)
    b_a, b_m = parameters["exposure_logit_effect"], parameters["mediator_logit_effect"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    rng = np.random.default_rng(random_state)
    log_odds_curves = {a: b_a*a+b_m*grid for a in (0, 1)}
    probability_curves = {
        a: 1/(1+np.exp(-log_odds_curves[a])) for a in (0, 1)
    }
    for a in (0, 1):
        axes[0].plot(grid, log_odds_curves[a], label=f"A={a}")
        axes[1].plot(grid, probability_curves[a], label=f"A={a}")
        for ax in axes:
            ax.axvline(means[a], linestyle="--" if a == 0 else ":", color="grey",
                       label=f"Mean observed M | A={a}")
        index = rng.choice(np.flatnonzero(exposure == a), n_rug, replace=False)
        axes[0].scatter(
            mediator[index], np.full(n_rug, -10.4-.8*a),
            marker="|", alpha=.45, label=f"Observed M, A={a}",
        )
        axes[1].scatter(
            mediator[index], np.full(n_rug, -.055-.05*a),
            marker="|", alpha=.45, label=f"Observed M, A={a}",
        )
        log_y0, log_y1 = b_a*a+b_m*means[0], b_a*a+b_m*means[1]
        probability_y0 = 1/(1+np.exp(-log_y0))
        probability_y1 = 1/(1+np.exp(-log_y1))
        axes[0].annotate(
            "", xy=(means[1], log_y1), xytext=(means[0], log_y0),
            arrowprops={"arrowstyle": "->", "lw": 3, "color": f"C{a}"},
        )
        axes[1].annotate(
            "", xy=(means[1], probability_y1), xytext=(means[0], probability_y0),
            arrowprops={"arrowstyle": "->", "lw": 3, "color": f"C{a}"},
        )
    additive = rf"$\eta(A,M)={b_a:g}A+{b_m:g}M$"
    probability = rf"$\operatorname{{expit}}({b_a:g}A+{b_m:g}M)$"
    axes[0].set(
        xlabel="Mediator M", ylabel="Log-odds of outcome",
        title=f"Additive function: {additive}",
    )
    axes[1].set(
        xlabel="Mediator M", ylabel="Outcome probability",
        title=f"Expit function: {probability}",
    )
    axes[1].set_ylim(-.14, 1.03)
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

def plot_real_data_effects(
    effects,
    *,
    datasets=("Law", "COMPAS", "Reddit", "SCUT"),
):
    """Plot estimated causal effects and 95% confidence intervals by dataset."""
    effect_order = (
        "nde_m0",
        "nde_m1",
        "nie_a0",
        "nie_a1",
        "interaction",
        "total_effect",
    )
    labels = (
        r"NDE$_{M(0)}$",
        r"NDE$_{M(1)}$",
        r"NIE$_{A=0}$",
        r"NIE$_{A=1}$",
        "Interaction",
        "Total",
    )
    colors = (
        "#0871B2",
        "#D55E00",
        "#0871B2",
        "#D55E00",
        "#7B3294",
        "#555555",
    )

    # Extra space between effect families.
    x = np.array([0.00, 0.62, 1.65, 2.27, 3.45, 4.55])
    bar_width = 0.62

    n_columns = 2
    n_rows = int(np.ceil(len(datasets) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(12, 3.75 * n_rows),
        squeeze=False,
    )

    for ax, dataset in zip(axes.flat, datasets):
        data = (
            effects.loc[effects["dataset"] == dataset]
            .set_index("effect")
            .reindex(effect_order)
        )

        if data["estimate"].isna().any():
            missing = data.index[data["estimate"].isna()].tolist()
            raise ValueError(
                f"Missing effects for dataset {dataset!r}: {missing}"
            )

        estimates = data["estimate"].to_numpy()
        lower = estimates - data["ci_low"].to_numpy()
        upper = data["ci_high"].to_numpy() - estimates

        ax.bar(
            x,
            estimates,
            width=bar_width,
            color=colors,
            yerr=np.vstack([lower, upper]),
            error_kw={
                "ecolor": "black",
                "capsize": 3,
                "elinewidth": 1,
            },
        )

        ax.axhline(0, color="0.4", linestyle="--", linewidth=1)
        ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=13)
        ax.tick_params(axis="y", labelsize=13)
        ax.set_title(dataset, fontweight="bold")
        ax.set_ylabel("Estimated effect", fontsize=14)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    # Remove unused panels if fewer than an even number are requested.
    for ax in axes.flat[len(datasets):]:
        fig.delaxes(ax)

    fig.tight_layout()
    return fig, axes


def robustness_metadata(dataset, check, assessment, config):
    """Describe the models and reduction used by a robustness check."""
    outcome_name = config['outcome_model']['name']
    if check == 'outcome_alternative':
        outcome_name = (
            'hgb_with_interactions'
            if outcome_name == 'linear_with_interactions'
            else 'linear_with_interactions'
        )
    outcome_label = _ROBUSTNESS_MODEL_LABELS[outcome_name]
    if outcome_name == 'linear_with_interactions' and config['outcome_type'] == 'binary':
        outcome_label = 'Logistic + A interactions'

    propensity_c = _ROBUSTNESS_MODEL_LABELS[
        config['propensity_models']['a_given_c']['name']
    ]
    propensity_mc_name = config['propensity_models']['a_given_mc']['name']
    if check == 'propensity_mc_hgb':
        propensity_mc_name = 'hist_gradient_boosting'
    propensity_mc = _ROBUSTNESS_MODEL_LABELS[propensity_mc_name]
    propensity_label = (
        propensity_c if propensity_c == propensity_mc
        else f'{propensity_c} C; {propensity_mc} MC'
    )

    reduction_name = (
        check
        if check in _ROBUSTNESS_REDUCTION_CHECKS
        else config['reduction']['name']
    )
    return {
        'id': (
            f"{_ROBUSTNESS_DATASET_ID[dataset]}-"
            f"{_ROBUSTNESS_CHECK_NUMBER[check]}"
        ),
        'outcome_model': outcome_label,
        'propensity_model': propensity_label,
        'reduction': _ROBUSTNESS_REDUCTION_LABELS[reduction_name],
        'input_dim': len(assessment['mediators']),
    }


def robustness_row(dataset, label, check, assessment, config, result):
    """Flatten one robustness-check result into a report-ready row."""
    diagnostics = list(result['diagnostics'].values())
    first = diagnostics[0]
    dimensions = [
        dimension
        for item in diagnostics
        for dimension in item['mediator_dimensions_by_fold']
    ]
    row = {
        'dataset': dataset, 'specification': label, 'check': check,
        **robustness_metadata(dataset, check, assessment, config),
        'n': len(assessment['data']),
        'reduced_dim_min': min(dimensions),
        'reduced_dim_max': max(dimensions),
        'min_ess': min(item['effective_sample_size'] for item in diagnostics),
        'max_clipping': max(
            max(item['fraction_propensity_c_clipped'],
                item['fraction_propensity_mc_clipped'])
            for item in diagnostics
        ),
        'max_absolute_weight': max(item['max_absolute_weight'] for item in diagnostics),
        'max_weight_p95': max(item['absolute_weight_p95'] for item in diagnostics),
        'max_weight_p99': max(item['absolute_weight_p99'] for item in diagnostics),
        'outcome_loss': first.get('outcome_log_loss', first.get('outcome_rmse')),
        'outcome_loss_name': 'log_loss' if 'outcome_log_loss' in first else 'rmse',
        'propensity_c_brier_score': first['propensity_c_brier_score'],
        'propensity_mc_brier_score': first['propensity_mc_brier_score'],
    }
    for effect in EFFECT_ORDER:
        low, high = result['uncertainty'][effect]['confidence_interval']
        row.update({f'{effect}_estimate': result['effects'][effect],
                    f'{effect}_ci_low': low, f'{effect}_ci_high': high})
    return row
