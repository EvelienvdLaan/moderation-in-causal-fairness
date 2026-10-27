"""Cross-fitted EIF estimation of psi(a, a') = E[Y(a, M(a'))]."""
from dataclasses import dataclass
from statistics import NormalDist
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_preparation import PFIReducer

from .diagnostics import outcome_model_diagnostics, overlap_diagnostics
from .effects import EFFECT_COEFFICIENTS, effects_from_nested_means


@dataclass
class NestedMeanEstimate:
    estimate: float
    standard_error: float
    confidence_interval: tuple
    influence_values: np.ndarray
    diagnostics: dict


def _prediction(model, x, binary):
    return model.predict_proba(x)[:, 1] if binary else model.predict(x)


def _probability(model, x, clip):
    return np.clip(model.predict_proba(x)[:, 1], clip, 1 - clip)


def _default_outcome(binary, seed):
    return HistGradientBoostingClassifier(max_iter=100, random_state=seed) if binary else HistGradientBoostingRegressor(max_iter=100, random_state=seed)


def _default_preprocessor(frame):
    """Dense, fold-fitted encoding for a mixed numeric/categorical block."""
    numeric = list(frame.select_dtypes(include=[np.number, "bool"]).columns)
    categorical = [column for column in frame.columns if column not in numeric]
    steps = []
    if numeric:
        steps.append(("numeric", StandardScaler(), numeric))
    if categorical:
        steps.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
    return ColumnTransformer(steps, sparse_threshold=0)


def estimate_nested_mean(data, outcome, protected_attribute, mediators, confounders,
                         a_outcome, a_mediator, *, outcome_model=None,
                         propensity_model=None, propensity_c_model=None,
                         propensity_mc_model=None, nested_model=None, transformer=None,
                         confounder_transformer=None,
                         n_splits=5, random_state=0, propensity_clip=0.01,
                         confidence_level=0.95):
    """Estimate a binary-exposure nested mean with a cross-fitted mediation EIF.

    ``transformer``, if supplied, is cloned and fit on each training fold only.
    All nuisance estimators must follow the sklearn API.
    """
    if a_outcome not in (0, 1) or a_mediator not in (0, 1):
        raise ValueError("a_outcome and a_mediator must be 0 or 1")
    required_complete = [outcome, protected_attribute]
    if transformer is None:
        required_complete.extend(mediators)
    if confounders and confounder_transformer is None:
        required_complete.extend(confounders)
    if data[required_complete].isnull().any().any():
        raise ValueError(
            "Missing values require an explicit fold-fitted transformer with imputation"
        )
    A = data[protected_attribute].to_numpy(dtype=int)
    if set(np.unique(A)) != {0, 1}:
        raise ValueError("protected_attribute must contain both binary values 0 and 1")
    Y = data[outcome].to_numpy(dtype=float)
    M = data[list(mediators)].copy()
    # An empty adjustment set is represented internally by a constant column.
    # This estimates an unadjusted mediation functional; it does not remove the
    # causal requirement that exchangeability hold without conditioning.
    C = data[list(confounders)].copy() if confounders else None
    binary_y = set(np.unique(Y)).issubset({0.0, 1.0})
    n = len(data)
    score, p_c_all, p_mc_all, weights = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    outcome_predictions = np.empty(n)
    mediator_dimensions, mediator_variance = [], []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train, test) in enumerate(splitter.split(np.zeros((n, 1)), A)):
        Mt_raw, Mv_raw = M.iloc[train], M.iloc[test]
        Ct_raw = C.iloc[train] if C is not None else None
        Cv_raw = C.iloc[test] if C is not None else None
        mediator_preprocessor = clone(transformer) if transformer is not None else _default_preprocessor(Mt_raw)
        if isinstance(mediator_preprocessor, PFIReducer):
            mediator_preprocessor.fit(Mt_raw, Y[train], A=A[train])
            Mt = np.asarray(mediator_preprocessor.transform(Mt_raw), dtype=float)
        else:
            Mt = np.asarray(mediator_preprocessor.fit_transform(Mt_raw, Y[train]), dtype=float)
        Mv = np.asarray(mediator_preprocessor.transform(Mv_raw), dtype=float)
        mediator_dimensions.append(int(Mt.shape[1]))
        explained = getattr(mediator_preprocessor, "explained_variance_ratio_", None)
        mediator_variance.append(
            float(np.sum(explained)) if explained is not None else np.nan
        )
        if C is None:
            Ct, Cv = np.zeros((len(train), 1)), np.zeros((len(test), 1))
        else:
            confounder_preprocessor = clone(confounder_transformer) if confounder_transformer is not None else _default_preprocessor(Ct_raw)
            Ct = np.asarray(confounder_preprocessor.fit_transform(Ct_raw), dtype=float)
            Cv = np.asarray(confounder_preprocessor.transform(Cv_raw), dtype=float)
        At, Av, Yt, Yv = A[train], A[test], Y[train], Y[test]
        oc = clone(outcome_model) if outcome_model is not None else _default_outcome(binary_y, random_state + fold)
        pc_template = propensity_c_model if propensity_c_model is not None else propensity_model
        pmc_template = propensity_mc_model if propensity_mc_model is not None else propensity_model
        pc = clone(pc_template) if pc_template is not None else LogisticRegression(max_iter=1000)
        pmc = clone(pmc_template) if pmc_template is not None else LogisticRegression(max_iter=1000)
        nm = clone(nested_model) if nested_model is not None else HistGradientBoostingRegressor(max_iter=100, random_state=random_state + fold)
        oc.fit(np.column_stack([At, Mt, Ct]), Yt)
        pc.fit(Ct, At)
        pmc.fit(np.column_stack([Mt, Ct]), At)
        p_c = _probability(pc, Cv, propensity_clip)
        p_mc = _probability(pmc, np.column_stack([Mv, Cv]), propensity_clip)
        outcome_predictions[test] = _prediction(
            oc, np.column_stack([Av, Mv, Cv]), binary_y
        )
        mu_train = _prediction(oc, np.column_stack([np.full(len(train), a_outcome), Mt, Ct]), binary_y)
        reference = At == a_mediator
        nm.fit(Ct[reference], mu_train[reference])
        eta = nm.predict(Cv)
        mu = _prediction(oc, np.column_stack([np.full(len(test), a_outcome), Mv, Cv]), binary_y)
        pa_c = p_c if a_outcome == 1 else 1 - p_c
        pap_c = p_c if a_mediator == 1 else 1 - p_c
        pa_mc = p_mc if a_outcome == 1 else 1 - p_mc
        pap_mc = p_mc if a_mediator == 1 else 1 - p_mc
        density_ratio = (pap_mc / pa_mc) * (pa_c / pap_c)
        first_weight = (Av == a_outcome) / pa_c * density_ratio
        second_weight = (Av == a_mediator) / pap_c
        score[test] = first_weight * (Yv - mu) + second_weight * (mu - eta) + eta
        p_c_all[test], p_mc_all[test], weights[test] = p_c, p_mc, first_weight
    estimate = float(np.mean(score))
    influence = score - estimate
    se = float(np.std(influence, ddof=1) / np.sqrt(n))
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2)
    diag = overlap_diagnostics(p_c_all, p_mc_all, weights, propensity_clip, A)
    diag.update(outcome_model_diagnostics(Y, outcome_predictions, binary=binary_y))
    diag["mediator_dimensions_by_fold"] = tuple(mediator_dimensions)
    diag["mediator_explained_variance_by_fold"] = tuple(mediator_variance)
    if diag["fraction_propensity_c_clipped"] > 0.05 or diag["fraction_propensity_mc_clipped"] > 0.05:
        warnings.warn("More than 5% of propensity estimates were clipped; inspect overlap.", RuntimeWarning)
    return NestedMeanEstimate(estimate, se, (estimate-z*se, estimate+z*se), influence, diag)


def estimate_reference_specific_effects(data, outcome, protected_attribute, mediators,
                                        confounders, **kwargs):
    """Estimate all four nested means, effects, covariance-aware SEs, and diagnostics."""
    fitted = {}
    for a in (0, 1):
        for ap in (0, 1):
            fitted[f"psi_{a}{ap}"] = estimate_nested_mean(
                data, outcome, protected_attribute, mediators, confounders, a, ap, **kwargs)
    means = {key: value.estimate for key, value in fitted.items()}
    effects = effects_from_nested_means(means)
    uncertainty = {}
    for name, coef in EFFECT_COEFFICIENTS.items():
        influence = sum(value * fitted[key].influence_values for key, value in coef.items())
        se = float(np.std(influence, ddof=1) / np.sqrt(len(data)))
        uncertainty[name] = {"standard_error": se, "confidence_interval": (effects[name]-1.96*se, effects[name]+1.96*se)}
    return {
        "nested_means": means,
        "nested_mean_details": fitted,
        "effects": effects,
        "uncertainty": uncertainty,
        "diagnostics": {key: value.diagnostics for key, value in fitted.items()},
    }
