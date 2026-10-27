"""Reference-specific in-processing with cross-fitted augmented penalties.

This module targets path-specific effects of a *prediction function*.  For a
candidate model ``f_theta``, the model-output nested mean is

    E_C E[f_theta(a, M, C) | A=a', C].

Because f_theta is known, its outcome-regression residual is zero.  We estimate
the remaining transported mean with a cross-fitted AIPW estimator.  It is
doubly robust if either P(A|C) or the linear nested regression is correct.
This is narrower than double robustness for effects of an observed outcome.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from estimate_effects.effects import EFFECT_COEFFICIENTS, effects_from_nested_means


@dataclass(frozen=True)
class InprocessingResult:
    coefficients: np.ndarray
    nested_means: dict
    effects: dict
    prediction_loss: float
    objective: float
    converged: bool
    message: str


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


class DoublyRobustInprocessor:
    """Linear/logistic predictor with reference-specific causal penalties.

    Parameters
    ----------
    outcome_type:
        ``"binary"`` for logistic prediction or ``"continuous"`` for linear.
    penalty_weights:
        Mapping from any effect in ``EFFECT_COEFFICIENTS``, including the
        reference-specific and symmetric NDE/NIE estimands, to a nonnegative
        squared-penalty weight. An empty mapping fits the unmitigated model.
    nested_ridge:
        Small ridge penalty used by the cross-fitted linear nested regression.
    """
    def __init__(self, outcome_type="binary", penalty_weights=None, *,
                 n_splits=5, random_state=0, propensity_clip=0.01,
                 nested_ridge=1e-6, l2=1e-6, max_iter=500,
                 propensity_model=None):
        self.outcome_type = outcome_type
        self.penalty_weights = {} if penalty_weights is None else dict(penalty_weights)
        self.n_splits = n_splits
        self.random_state = random_state
        self.propensity_clip = propensity_clip
        self.nested_ridge = nested_ridge
        self.l2 = l2
        self.max_iter = max_iter
        self.propensity_model = propensity_model

    def _validate(self, data, outcome, protected_attribute, mediators, confounders):
        if self.outcome_type not in {"binary", "continuous"}:
            raise ValueError("outcome_type must be 'binary' or 'continuous'")
        allowed = set(EFFECT_COEFFICIENTS)
        if not set(self.penalty_weights).issubset(allowed) or any(v < 0 for v in self.penalty_weights.values()):
            raise ValueError("Unknown penalty target or negative penalty weight")
        columns = [outcome, protected_attribute, *mediators, *confounders]
        if data[columns].isnull().any().any():
            raise ValueError("Missing values are not supported")
        A = data[protected_attribute].to_numpy(dtype=int)
        if set(np.unique(A)) != {0, 1}:
            raise ValueError("protected_attribute must contain 0 and 1")
        Y = data[outcome].to_numpy(dtype=float)
        if self.outcome_type == "binary" and not set(np.unique(Y)).issubset({0.0, 1.0}):
            raise ValueError("binary outcomes must contain only 0 and 1")
        return A, Y

    @staticmethod
    def _encode_frame(frame, fitted=None):
        if frame is None or frame.shape[1] == 0:
            return np.zeros((len(frame) if frame is not None else 0, 0)), None
        numeric = list(frame.select_dtypes(include=[np.number, "bool"]).columns)
        categorical = [column for column in frame.columns if column not in numeric]
        if fitted is None:
            scaler = StandardScaler().fit(frame[numeric]) if numeric else None
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(frame[categorical]) if categorical else None
            fitted = (numeric, categorical, scaler, encoder)
        numeric, categorical, scaler, encoder = fitted
        blocks = []
        if numeric:
            blocks.append(scaler.transform(frame[numeric]))
        if categorical:
            blocks.append(encoder.transform(frame[categorical]))
        return np.column_stack(blocks), fitted

    def _prepare_prediction_features(self, data, mediators, confounders, fit):
        M_frame, C_frame = data[list(mediators)], data[list(confounders)]
        if fit:
            M, self.mediator_encoder_ = self._encode_frame(M_frame)
            C, self.confounder_encoder_ = self._encode_frame(C_frame)
        else:
            M, _ = self._encode_frame(M_frame, self.mediator_encoder_)
            C, _ = self._encode_frame(C_frame, self.confounder_encoder_)
        return M, C

    @staticmethod
    def _design(A, M, C):
        return np.column_stack([np.ones(len(A)), A, M, C, A[:, None] * M])

    def _cross_fitted_integration_weights(self, data, A, confounders):
        """Precompute linear AIPW operators for mediator references 0 and 1."""
        n = len(A)
        weights = {0: np.zeros(n), 1: np.zeros(n)}
        splitter = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        C_frame = data[list(confounders)]
        for train, test in splitter.split(np.zeros((n, 1)), A):
            if confounders:
                C_train, encoder = self._encode_frame(C_frame.iloc[train])
                C_test, _ = self._encode_frame(C_frame.iloc[test], encoder)
            else:
                C_train, C_test = np.zeros((len(train), 0)), np.zeros((len(test), 0))
            X_train = np.column_stack([np.ones(len(train)), C_train])
            X_test = np.column_stack([np.ones(len(test)), C_test])
            propensity = clone(self.propensity_model) if self.propensity_model is not None else LogisticRegression(max_iter=1000)
            propensity.fit(C_train if C_train.shape[1] else np.zeros((len(train), 1)), A[train])
            p1 = np.clip(propensity.predict_proba(C_test if C_test.shape[1] else np.zeros((len(test), 1)))[:, 1], self.propensity_clip, 1-self.propensity_clip)
            for reference in (0, 1):
                selected_train = A[train] == reference
                X_reference = X_train[selected_train]
                penalty = np.eye(X_reference.shape[1]) * self.nested_ridge
                penalty[0, 0] = 0
                inverse = np.linalg.pinv(X_reference.T @ X_reference + penalty)
                p_reference = p1 if reference == 1 else 1-p1
                inverse_weight = (A[test] == reference) / p_reference
                weights[reference][test] += inverse_weight / n
                correction = X_reference @ (
                    inverse @ (X_test.T @ (1-inverse_weight))
                )
                weights[reference][train[selected_train]] += correction / n
        return weights

    def _predict_from_design(self, design, coefficients):
        linear = design @ coefficients
        return _sigmoid(linear) if self.outcome_type == "binary" else linear

    def _nested_means_and_gradients(self, coefficients, M, C, integration_weights):
        means, gradients = {}, {}
        for a in (0, 1):
            design = self._design(np.full(len(M), a), M, C)
            prediction = self._predict_from_design(design, coefficients)
            derivative = prediction*(1-prediction) if self.outcome_type == "binary" else np.ones(len(M))
            for reference in (0, 1):
                key = f"psi_{a}{reference}"
                weight = integration_weights[reference]
                means[key] = float(weight @ prediction)
                gradients[key] = design.T @ (weight * derivative)
        return means, gradients

    @staticmethod
    def _effect_coefficients():
        return EFFECT_COEFFICIENTS

    def fit(self, data, outcome, protected_attribute, mediators, confounders=()):
        A, Y = self._validate(data, outcome, protected_attribute, mediators, confounders)
        M, C = self._prepare_prediction_features(data, mediators, confounders, fit=True)
        observed_design = self._design(A, M, C)
        integration_weights = self._cross_fitted_integration_weights(data, A, confounders)
        effect_coefficients = self._effect_coefficients()

        def objective_and_gradient(coefficients):
            prediction = self._predict_from_design(observed_design, coefficients)
            if self.outcome_type == "binary":
                clipped = np.clip(prediction, 1e-9, 1-1e-9)
                prediction_loss = -np.mean(Y*np.log(clipped) + (1-Y)*np.log(1-clipped))
                gradient = observed_design.T @ (prediction-Y) / len(Y)
            else:
                residual = prediction-Y
                prediction_loss = np.mean(residual**2)
                gradient = 2*observed_design.T @ residual / len(Y)
            means, mean_gradients = self._nested_means_and_gradients(coefficients, M, C, integration_weights)
            effects = effects_from_nested_means(means)
            value = prediction_loss + self.l2*np.sum(coefficients[1:]**2)
            gradient = gradient + np.r_[0, 2*self.l2*coefficients[1:]]
            for effect, weight in self.penalty_weights.items():
                effect_gradient = sum(multiplier*mean_gradients[key] for key, multiplier in effect_coefficients[effect].items())
                value += weight*effects[effect]**2
                gradient += 2*weight*effects[effect]*effect_gradient
            return value, gradient

        initial = np.zeros(observed_design.shape[1])
        optimized = minimize(objective_and_gradient, initial, jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.coefficients_ = optimized.x
        self.feature_count_ = observed_design.shape[1]
        self.outcome_ = outcome
        self.protected_attribute_ = protected_attribute
        self.mediators_ = list(mediators)
        self.confounders_ = list(confounders)
        means, _ = self._nested_means_and_gradients(self.coefficients_, M, C, integration_weights)
        effects = effects_from_nested_means(means)
        prediction = self._predict_from_design(observed_design, self.coefficients_)
        if self.outcome_type == "binary":
            clipped = np.clip(prediction, 1e-9, 1-1e-9)
            loss = float(-np.mean(Y*np.log(clipped)+(1-Y)*np.log(1-clipped)))
        else:
            loss = float(np.mean((prediction-Y)**2))
        self.result_ = InprocessingResult(self.coefficients_.copy(), means, effects, loss, float(optimized.fun), bool(optimized.success), optimized.message)
        return self

    def predict(self, data):
        M, C = self._prepare_prediction_features(data, self.mediators_, self.confounders_, fit=False)
        A = data[self.protected_attribute_].to_numpy(dtype=int)
        return self._predict_from_design(self._design(A, M, C), self.coefficients_)

    def predict_proba(self, data):
        if self.outcome_type != "binary":
            raise AttributeError("predict_proba is only available for binary outcomes")
        probability = self.predict(data)
        return np.column_stack([1-probability, probability])

    def estimate_model_effects(self, data):
        """Estimate all model-output effects on new data with cross-fitted AIPW."""
        A = data[self.protected_attribute_].to_numpy(dtype=int)
        M, C = self._prepare_prediction_features(data, self.mediators_, self.confounders_, fit=False)
        weights = self._cross_fitted_integration_weights(data, A, self.confounders_)
        means, _ = self._nested_means_and_gradients(self.coefficients_, M, C, weights)
        return {"nested_means": means, "effects": effects_from_nested_means(means)}
