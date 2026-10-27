"""Small, reproducible nuisance-model selection for the effect estimator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_preparation import make_mediator_reducer


HGB_GRID = {
    "learning_rate": [0.05, 0.1],
    "max_leaf_nodes": [15, 31],
    "l2_regularization": [0.0, 1.0],
}


class ExposureInteractions(BaseEstimator, TransformerMixin):
    """Append interactions between the first (protected) column and all others."""

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Unexpected number of outcome-model features")
        return np.column_stack([X, X[:, [0]] * X[:, 1:]])


def _binary(values):
    return set(np.unique(np.asarray(values))).issubset({0, 1})


def reproducible_sample(data, target, max_samples, *, random_state=4):
    """Return a bounded sample, stratified when the requested target is binary."""
    if len(data) <= max_samples:
        return data.copy().reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    if not _binary(data[target]):
        chosen = rng.choice(len(data), size=max_samples, replace=False)
    else:
        chosen_parts = []
        proportions = data[target].value_counts(normalize=True).sort_index()
        allocations = np.floor(proportions * max_samples).astype(int)
        remainder = max_samples - int(allocations.sum())
        for group in proportions.sort_values(ascending=False).index[:remainder]:
            allocations.loc[group] += 1
        for group, count in allocations.items():
            positions = np.flatnonzero(data[target].to_numpy() == group)
            chosen_parts.append(rng.choice(positions, size=int(count), replace=False))
        chosen = np.concatenate(chosen_parts)
        rng.shuffle(chosen)
    return data.iloc[chosen].copy().reset_index(drop=True)


def _tabular_preprocessor(frame):
    numeric = list(frame.select_dtypes(include=[np.number, "bool"]).columns)
    categorical = [column for column in frame if column not in numeric]
    steps = []
    if numeric:
        steps.append(("numeric", StandardScaler(), numeric))
    if categorical:
        steps.append((
            "categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical,
        ))
    if not steps:
        raise ValueError("At least one feature is required")
    return ColumnTransformer(steps, sparse_threshold=0)


def _transform_block(train, test, columns, reduction):
    if not columns:
        return np.zeros((len(train), 0)), np.zeros((len(test), 0))
    train_block, test_block = train[list(columns)], test[list(columns)]
    if reduction["name"] == "pca70":
        transformer = PCA(n_components=0.70, random_state=reduction["params"]["random_state"])
    else:
        transformer = _tabular_preprocessor(train_block)
    return (
        np.asarray(transformer.fit_transform(train_block), dtype=float),
        np.asarray(transformer.transform(test_block), dtype=float),
    )


def _folds(data, target, cv, random_state):
    y = data[target].to_numpy()
    splitter = (
        StratifiedKFold(cv, shuffle=True, random_state=random_state)
        if _binary(y) else KFold(cv, shuffle=True, random_state=random_state)
    )
    return list(splitter.split(np.zeros((len(data), 1)), y if _binary(y) else None))


def _prepared_folds(data, outcome, protected_attribute, mediators, confounders,
                    reduction, cv, random_state):
    prepared = []
    for train_index, test_index in _folds(data, outcome, cv, random_state):
        train, test = data.iloc[train_index], data.iloc[test_index]
        Mt, Mv = _transform_block(train, test, mediators, reduction)
        Ct, Cv = _transform_block(
            train, test, confounders,
            {"name": "none", "params": {"random_state": random_state}},
        )
        if not confounders:
            # Match estimate_nested_mean's explicit representation of an empty C set.
            Ct, Cv = np.zeros((len(train), 1)), np.zeros((len(test), 1))
        At = train[protected_attribute].to_numpy(dtype=float)
        Av = test[protected_attribute].to_numpy(dtype=float)
        prepared.append({
            "train_index": train_index, "test_index": test_index,
            "y_train": train[outcome].to_numpy(), "y_test": test[outcome].to_numpy(),
            "a_train": At, "a_test": Av,
            "outcome_train": np.column_stack([At, Mt, Ct]),
            "outcome_test": np.column_stack([Av, Mv, Cv]),
            "c_train": Ct, "c_test": Cv,
            "mc_train": np.column_stack([Mt, Ct]),
            "mc_test": np.column_stack([Mv, Cv]),
        })
    return prepared


def _outcome_estimator(name, params, binary, random_state):
    if name == "linear_with_interactions":
        base = LogisticRegression(max_iter=3000) if binary else LinearRegression()
        return make_pipeline(ExposureInteractions(), base)
    model_class = HistGradientBoostingClassifier if binary else HistGradientBoostingRegressor
    options = {"max_iter": 150, "early_stopping": True, "random_state": random_state, **params}
    model = model_class(**options)
    return make_pipeline(ExposureInteractions(), model) if name == "hgb_with_interactions" else model


def _propensity_estimator(name, params, random_state):
    if name == "logistic":
        return LogisticRegression(max_iter=3000, C=1.0)
    options = {"max_iter": 150, "early_stopping": True, "random_state": random_state, **params}
    return HistGradientBoostingClassifier(**options)


def _candidate_rows(prepared, candidates, *, kind, binary=True, random_state=4,
                    propensity_clip=0.01):
    rows = []
    n = sum(len(fold["test_index"]) for fold in prepared)
    for name, params in candidates:
        fold_metrics, oof, observed = [], np.empty(n), np.empty(n)
        for fold in prepared:
            if kind == "outcome":
                estimator = _outcome_estimator(name, params, binary, random_state)
                X_train, X_test = fold["outcome_train"], fold["outcome_test"]
                y_train, y_test = fold["y_train"], fold["y_test"]
            else:
                estimator = _propensity_estimator(name, params, random_state)
                X_train, X_test = fold[f"{kind}_train"], fold[f"{kind}_test"]
                y_train, y_test = fold["a_train"], fold["a_test"]
            fitted = clone(estimator).fit(X_train, y_train)
            prediction = (
                fitted.predict_proba(X_test)[:, 1]
                if binary or kind != "outcome" else fitted.predict(X_test)
            )
            oof[fold["test_index"]], observed[fold["test_index"]] = prediction, y_test
            if binary or kind != "outcome":
                fold_metrics.append({
                    "primary": log_loss(y_test, prediction, labels=[0, 1]),
                    "secondary": brier_score_loss(y_test, prediction),
                    "auroc": roc_auc_score(y_test, prediction),
                })
            else:
                fold_metrics.append({
                    "primary": mean_squared_error(y_test, prediction) ** .5,
                    "secondary": mean_absolute_error(y_test, prediction),
                })
        row = {
            "name": name, "params": dict(params),
            "primary_mean": float(np.mean([metric["primary"] for metric in fold_metrics])),
            "primary_std": float(np.std([metric["primary"] for metric in fold_metrics], ddof=1)),
            "secondary_mean": float(np.mean([metric["secondary"] for metric in fold_metrics])),
            "secondary_std": float(np.std([metric["secondary"] for metric in fold_metrics], ddof=1)),
            "oof_predictions": oof,
        }
        if binary or kind != "outcome":
            row.update({
                "auroc_mean": float(np.mean([metric["auroc"] for metric in fold_metrics])),
                "auroc_std": float(np.std([metric["auroc"] for metric in fold_metrics], ddof=1)),
                "clipped_fraction": float(np.mean((oof <= propensity_clip) | (oof >= 1-propensity_clip)))
                if kind != "outcome" else np.nan,
            })
        rows.append(row)
    return rows


def _select(rows, *, simple_name, tolerance=0.01):
    best = min(rows, key=lambda row: row["primary_mean"])
    simple = min(
        (row for row in rows if row["name"] == simple_name),
        key=lambda row: row["primary_mean"],
    )
    relative_gain = (simple["primary_mean"] - best["primary_mean"]) / simple["primary_mean"]
    if relative_gain < tolerance:
        return simple
    if (
        "clipped_fraction" in best and np.isfinite(best["clipped_fraction"])
        and best["clipped_fraction"] > simple["clipped_fraction"] + 0.05
        and relative_gain < 0.05
    ):
        return simple
    return best


def _public_results(rows, primary_metric, secondary_metric):
    return pd.DataFrame([{
        "model": row["name"], **row["params"],
        f"{primary_metric}_mean": row["primary_mean"],
        f"{primary_metric}_std": row["primary_std"],
        f"{secondary_metric}_mean": row["secondary_mean"],
        f"{secondary_metric}_std": row["secondary_std"],
        **({"auroc_mean": row["auroc_mean"], "auroc_std": row["auroc_std"],
            "clipped_fraction": row["clipped_fraction"]}
           if "auroc_mean" in row else {}),
    } for row in rows]).sort_values(f"{primary_metric}_mean").reset_index(drop=True)


def _selection_record(row, primary_metric, secondary_metric):
    secondary = {secondary_metric: row["secondary_mean"]}
    if "auroc_mean" in row:
        secondary["auroc"] = row["auroc_mean"]
    params = dict(row["params"])
    if "hgb" in row["name"] or row["name"] == "hist_gradient_boosting":
        params.update({"max_iter": 150, "early_stopping": True})
    return {
        "name": row["name"], "params": params,
        "primary_metric": primary_metric, "primary_score": row["primary_mean"],
        "secondary_scores": secondary,
    }


def _overlap_diagnostics(a, p_c, p_mc, clip):
    p_c, p_mc = np.clip(p_c, clip, 1-clip), np.clip(p_mc, clip, 1-clip)
    ess, weights = [], []
    for outcome_level in (0, 1):
        for mediator_level in (0, 1):
            pa_c = p_c if outcome_level else 1-p_c
            pap_c = p_c if mediator_level else 1-p_c
            pa_mc = p_mc if outcome_level else 1-p_mc
            pap_mc = p_mc if mediator_level else 1-p_mc
            weight = (a == outcome_level) / pa_c * (pap_mc/pa_mc) * (pa_c/pap_c)
            weights.append(np.abs(weight))
            ess.append(float(weight.sum() ** 2 / np.sum(weight**2)) if np.any(weight) else 0.0)
    all_weights = np.concatenate(weights)
    return {
        "min_effective_sample_size": min(ess),
        "max_clipped_fraction": float(max(
            np.mean((p_c <= clip) | (p_c >= 1-clip)),
            np.mean((p_mc <= clip) | (p_mc >= 1-clip)),
        )),
        "propensity_c_percentiles": dict(zip(
            ("p01", "p05", "p50", "p95", "p99"),
            map(float, np.quantile(p_c, [.01, .05, .5, .95, .99])),
        )),
        "propensity_mc_percentiles": dict(zip(
            ("p01", "p05", "p50", "p95", "p99"),
            map(float, np.quantile(p_mc, [.01, .05, .5, .95, .99])),
        )),
        "absolute_weight_percentiles": dict(zip(
            ("p50", "p90", "p95", "p99"),
            map(float, np.quantile(all_weights, [.5, .9, .95, .99])),
        )),
    }


def tune_nuisance_models(dataset, assessment, *, max_samples=1000, cv=2,
                         random_state=4, propensity_clip=0.01,
                         simplicity_tolerance=0.01):
    """Tune one complete nuisance configuration using predictive losses only."""
    data = reproducible_sample(
        assessment["data"], assessment["outcome"], max_samples,
        random_state=random_state,
    )
    outcome, protected = assessment["outcome"], assessment["protected_attribute"]
    mediators, confounders = assessment["mediators"], assessment["confounders"]
    binary = _binary(data[outcome])
    high_dimensional = dataset.lower() in {"scut", "reddit"}
    reduction = {
        "name": "pca70" if high_dimensional else "none",
        "params": {"n_components": 0.70, "random_state": random_state}
        if high_dimensional else {},
    }
    prepared = _prepared_folds(
        data, outcome, protected, mediators, confounders, reduction, cv, random_state,
    )
    hgb_params = list(ParameterGrid(HGB_GRID))
    outcome_candidates = [("linear_with_interactions", {})]
    outcome_candidates += [("hgb", params) for params in hgb_params]
    outcome_candidates += [("hgb_with_interactions", params) for params in hgb_params]
    # Propensity complexity is prespecified rather than selected: probability
    # calibration and stable overlap take priority over small predictive gains.
    propensity_candidates = [("logistic", {"C": 1.0})]
    outcome_rows = _candidate_rows(
        prepared, outcome_candidates, kind="outcome", binary=binary,
        random_state=random_state, propensity_clip=propensity_clip,
    )
    c_rows = _candidate_rows(
        prepared, propensity_candidates, kind="c", random_state=random_state,
        propensity_clip=propensity_clip,
    )
    mc_rows = _candidate_rows(
        prepared, propensity_candidates, kind="mc", random_state=random_state,
        propensity_clip=propensity_clip,
    )
    outcome_choice = _select(
        outcome_rows, simple_name="linear_with_interactions",
        tolerance=simplicity_tolerance,
    )
    c_choice = _select(c_rows, simple_name="logistic", tolerance=simplicity_tolerance)
    mc_choice = _select(mc_rows, simple_name="logistic", tolerance=simplicity_tolerance)
    primary, secondary = ("log_loss", "brier") if binary else ("rmse", "mae")
    config = {
        "dataset": dataset, "outcome_type": "binary" if binary else "continuous",
        "selection": {
            "criterion": "lowest mean out-of-sample primary loss for outcome model only",
            "simplicity_tolerance_relative": simplicity_tolerance,
            "causal_effects_used": False,
            "propensity_models_prespecified": "LogisticRegression(C=1.0)",
            "nested_model_prespecified": "LinearRegression",
        },
        "sample_size": len(data), "cv": cv, "random_state": random_state,
        "outcome_model": _selection_record(outcome_choice, primary, secondary),
        "propensity_models": {
            "a_given_c": _selection_record(c_choice, "log_loss", "brier"),
            "a_given_mc": _selection_record(mc_choice, "log_loss", "brier"),
        },
        "reduction": reduction,
        "diagnostics": _overlap_diagnostics(
            data[protected].to_numpy(), c_choice["oof_predictions"],
            mc_choice["oof_predictions"], propensity_clip,
        ),
    }
    tables = {
        "outcome": _public_results(outcome_rows, primary, secondary),
        "a_given_c": _public_results(c_rows, "log_loss", "brier"),
        "a_given_mc": _public_results(mc_rows, "log_loss", "brier"),
    }
    return config, tables


def save_tuned_configs(configs, path):
    """Write named dataset configurations as portable JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {config["dataset"]: config for config in configs}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_tuned_config(dataset, path="results/tuning/best_nuisance_models.json"):
    """Load one dataset configuration without constructing fitted state."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if dataset not in payload:
        raise KeyError(f"No tuned configuration for {dataset!r}")
    return payload[dataset]


def make_estimator_options(config):
    """Construct new, unfitted estimator objects from a saved configuration."""
    binary = config["outcome_type"] == "binary"
    outcome = config["outcome_model"]
    propensity_c = config["propensity_models"]["a_given_c"]
    propensity_mc = config["propensity_models"]["a_given_mc"]
    random_state = config.get("random_state", 4)
    options = {
        "outcome_model": _outcome_estimator(
            outcome["name"], outcome["params"], binary, random_state,
        ),
        "propensity_c_model": _propensity_estimator(
            propensity_c["name"], propensity_c["params"], random_state,
        ),
        "propensity_mc_model": _propensity_estimator(
            propensity_mc["name"], propensity_mc["params"], random_state,
        ),
        "nested_model": LinearRegression(),
    }
    reduction = config["reduction"]
    if reduction["name"] == "pca70":
        options["transformer"] = PCA(
            n_components=reduction["params"]["n_components"],
            random_state=reduction["params"].get("random_state", random_state),
        )
    return options


def make_robustness_options(config, check):
    """Return fresh options for one prespecified, one-at-a-time sensitivity check."""
    options = make_estimator_options(config)
    binary = config["outcome_type"] == "binary"
    random_state = config.get("random_state", 4)
    if check == "primary":
        return options
    if check == "outcome_alternative":
        selected = config["outcome_model"]["name"]
        if selected == "linear_with_interactions":
            options["outcome_model"] = _outcome_estimator(
                "hgb_with_interactions",
                {"learning_rate": .1, "max_leaf_nodes": 31, "l2_regularization": 1.0},
                binary, random_state,
            )
        else:
            options["outcome_model"] = _outcome_estimator(
                "linear_with_interactions", {}, binary, random_state,
            )
        return options
    if check == "propensity_mc_hgb":
        options["propensity_mc_model"] = _propensity_estimator(
            "hist_gradient_boosting",
            {"learning_rate": .1, "max_leaf_nodes": 31, "l2_regularization": 1.0},
            random_state,
        )
        return options
    reduction_components = {
        "pca60": ("pca", .60),
        "pca90": ("pca", .90),
        "pfi20": ("pfi", .20),
    }
    if check in reduction_components:
        method, components = reduction_components[check]
        kwargs = {"n_repeats": 2, "max_iter": 75} if method == "pfi" else {}
        options["transformer"] = make_mediator_reducer(
            method, components, random_state=random_state, **kwargs,
        )
        return options
    raise ValueError(f"Unknown robustness check: {check!r}")
