"""Summary figures and tables for mitigation analyses."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


LOSS_METRICS = {"Law": "log_loss", "COMPAS": "log_loss", "Reddit": "rmse", "SCUT": "rmse"}
METRICS = {
    "Law": ("log_loss", "auroc", "Log loss", "AUROC"),
    "COMPAS": ("log_loss", "auroc", "Log loss", "AUROC"),
    "Reddit": ("rmse", "r2", "RMSE", r"R²"),
    "SCUT": ("rmse", "r2", "RMSE", r"R²"),
}
PATHWAYS = {
    "Direct": ("direct", "nde_m0", "nde_m1", "symmetric_nde",
               ("No mitigation", "NDE M(0)", "NDE M(1)", "Symmetric NDE", "Both NDEs")),
    "Indirect": ("indirect", "nie_a0", "nie_a1", "symmetric_nie",
                 ("No mitigation", "NIE A=0", "NIE A=1", "Symmetric NIE", "Both NIEs")),
}


def plot_mitigation_summary_figure(mitigation_results):
    """Plot reference-specific effects and relative predictive loss by dataset."""
    panels = {
        "Direct effects": ("direct", ("nde_m0", "nde_m1"),
                           (r"NDE$_{M(0)}$", r"NDE$_{M(1)}$")),
        "Indirect effects": ("indirect", ("nie_a0", "nie_a1"),
                             (r"NIE$_{A=0}$", r"NIE$_{A=1}$")),
    }
    labels = ["No mitigation", "Reference 0", "Reference 1", "Symmetric", "Both references"]
    reference_colors = ("#0871B2", "#D55E00")
    style = {"font.size": 17, "axes.titlesize": 21, "axes.labelsize": 18,
             "xtick.labelsize": 17, "ytick.labelsize": 16, "legend.fontsize": 17,
             "figure.titlesize": 23}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(16, 9.5))
        outer = fig.add_gridspec(2, len(mitigation_results), wspace=.30, hspace=.30)
        for row, (title, (result_key, effect_names, effect_labels)) in enumerate(panels.items()):
            for column, (dataset, result) in enumerate(mitigation_results.items()):
                performance = (result["performance"].loc[lambda x: x.scope.eq("overall")]
                               .drop_duplicates("strategy").set_index("strategy"))
                inner = outer[row, column].subgridspec(2, 1, height_ratios=(3.1, 1), hspace=.05)
                ax_effect = fig.add_subplot(inner[0])
                ax_loss = fig.add_subplot(inner[1], sharex=ax_effect)
                effects = result[result_key]
                strategies = effects.index.tolist()
                x = np.arange(len(strategies))
                for offset, effect, label, color in zip((-.5, .5), effect_names, effect_labels,
                                                        reference_colors):
                    estimates = effects[effect].to_numpy()
                    low = effects[f"{effect}_lower"].to_numpy()
                    high = effects[f"{effect}_upper"].to_numpy()
                    ax_effect.bar(x + offset*.36, estimates, width=.36, color=color, alpha=.9,
                                  yerr=np.vstack([estimates-low, high-estimates]),
                                  error_kw={"ecolor": "black", "capsize": 3, "elinewidth": 1.1},
                                  label=label, zorder=2)
                ax_effect.axhline(0, color=".4", linestyle="--", linewidth=1)
                if row == 0:
                    ax_effect.set_title(dataset, fontweight="bold", pad=10)
                if column == 0:
                    ax_effect.set_ylabel(f"{title}\nEffect", fontweight="bold", labelpad=10)
                ax_effect.tick_params(axis="x", labelbottom=False)
                metric = LOSS_METRICS[dataset]
                loss = performance[metric].reindex(strategies)
                if loss.isna().any():
                    raise ValueError(f"Missing {metric} values for {dataset}: {loss.index[loss.isna()].tolist()}")
                relative = 100*(loss/loss.loc["No mitigation"]-1)
                ax_loss.axhline(0, color=".4", linewidth=1, zorder=1)
                ax_loss.plot(x, relative, color="#555555", linewidth=1.8, zorder=2)
                ax_loss.scatter(x, relative, color=np.where(relative.to_numpy() > 0, "#D55E00", "#0871B2"),
                                edgecolor="white", linewidth=.7, s=55, zorder=3)
                if column == 0:
                    ax_loss.set_ylabel(r"$\Delta$ loss (%)", fontsize=15, labelpad=7)
                ax_loss.set_xticks(x)
                ax_loss.set_xticklabels(labels if row == 1 else [], rotation=35, ha="right",
                                        rotation_mode="anchor", fontsize=15)
                for axis in (ax_effect, ax_loss):
                    axis.grid(axis="y", color="#E6E6E6", linewidth=.8)
                    axis.set_axisbelow(True)
                    axis.spines[["top", "right"]].set_visible(False)
        fig.legend(handles=[Patch(facecolor=reference_colors[0], alpha=.9, label="Reference value 0"),
                            Patch(facecolor=reference_colors[1], alpha=.9, label="Reference value 1"),
                            Line2D([0], [0], color="#555555", marker="o", markerfacecolor="#D55E00",
                                   markeredgecolor="white", markersize=8, linewidth=1.8,
                                   label="Predictive-loss increase")],
                   loc="lower center", ncol=3, frameon=False, handlelength=2, columnspacing=1.8)
        fig.subplots_adjust(left=.065, right=.99, bottom=.22, top=.91)
    return fig


def _overview_rows(mitigation_results, decimals, positions):
    rows = []
    for dataset, result in mitigation_results.items():
        loss, secondary, loss_label, secondary_label = METRICS[dataset]
        performance = (result["performance"].loc[lambda x: x.scope.eq("overall")]
                       .drop_duplicates("strategy").set_index("strategy"))
        missing = [metric for metric in (loss, secondary) if metric not in performance]
        if missing:
            raise ValueError(f"Missing performance metrics for {dataset}: {missing}. Available columns are {performance.columns.tolist()}.")
        baseline_loss = performance.loc["No mitigation", loss]
        baseline_secondary = performance.loc["No mitigation", secondary]
        for pathway, (key, ref0, ref1, symmetric, names) in PATHWAYS.items():
            selected = result[key].iloc[list(positions)]
            for display_name, (actual_name, values) in zip([names[i] for i in positions], selected.iterrows()):
                ci = lambda effect: f"[{values[f'{effect}_lower']:.{decimals}f}, {values[f'{effect}_upper']:.{decimals}f}]"
                predictive_loss = performance.loc[actual_name, loss]
                secondary_value = performance.loc[actual_name, secondary]
                rows.append({"Dataset": dataset, "Pathway": pathway, "Strategy": display_name,
                             "Reference 0": values[ref0], "Reference 0 95% CI": ci(ref0),
                             "Reference 1": values[ref1], "Reference 1 95% CI": ci(ref1),
                             "Symmetric effect": values[symmetric], "Symmetric effect 95% CI": ci(symmetric),
                             "Loss metric": loss_label, "Predictive loss": predictive_loss,
                             "Loss increase (%)": 100*(predictive_loss/baseline_loss-1),
                             "Secondary metric": secondary_label, "Secondary performance": secondary_value,
                             "Secondary change": secondary_value-baseline_secondary})
    frame = pd.DataFrame(rows)
    numeric = ["Reference 0", "Reference 1", "Symmetric effect", "Predictive loss",
               "Loss increase (%)", "Secondary performance", "Secondary change"]
    frame[numeric] = frame[numeric].round(decimals)
    return frame


def mitigation_compact_overview_table(mitigation_results, decimals=3):
    """Return baseline, symmetric, and both-reference mitigation rows."""
    frame = _overview_rows(mitigation_results, decimals, (0, 3, 4))
    frame["Loss metric"] = frame["Dataset"].map(LOSS_METRICS)
    return frame[["Dataset", "Pathway", "Strategy", "Reference 0", "Reference 1",
                  "Symmetric effect", "Symmetric effect 95% CI", "Loss metric",
                  "Predictive loss", "Loss increase (%)"]].rename(
        columns={"Symmetric effect 95% CI": "95% CI"})


def mitigation_overview_table(mitigation_results, decimals=3):
    """Return all mitigation strategies with effect and performance summaries."""
    return _overview_rows(mitigation_results, decimals, (0, 1, 2, 3, 4))
