"""Leakage-explicit PCA and PFI reduction for mediator representations."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


class PFIReducer(BaseEstimator, TransformerMixin):
    """Select original coordinates using held-out permutation importance."""

    def __init__(self, n_components=0.20, validation_fraction=0.25,
                 n_repeats=3, max_iter=100, random_state=0):
        self.n_components = n_components
        self.validation_fraction = validation_fraction
        self.n_repeats = n_repeats
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None, A=None):
        if y is None:
            raise ValueError("PFIReducer requires y during fit")
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if X.ndim != 2 or len(X) != len(y) or not np.isfinite(X).all():
            raise ValueError("X must be a finite 2D matrix aligned with y")
        if isinstance(self.n_components, (float, np.floating)):
            if not 0 < self.n_components <= 1:
                raise ValueError("Fractional n_components must be in (0, 1]")
            n_selected = max(1, int(np.ceil(self.n_components * X.shape[1])))
        else:
            n_selected = int(self.n_components)
            if not 1 <= n_selected <= X.shape[1]:
                raise ValueError("n_components must be between 1 and the number of columns")
        if A is not None:
            A = np.asarray(A, dtype=float).reshape(-1, 1)
            if len(A) != len(X) or not np.isfinite(A).all():
                raise ValueError("A must be finite and aligned with X")
            model_X = np.column_stack([X, A])
        else:
            model_X = X
        binary = set(np.unique(y)).issubset({0, 1})
        X_fit, X_valid, y_fit, y_valid = train_test_split(
            model_X, y, test_size=self.validation_fraction,
            random_state=self.random_state, stratify=y if binary else None,
        )
        if binary:
            model = HistGradientBoostingClassifier(
                max_iter=self.max_iter, random_state=self.random_state
            )
            scoring = "neg_log_loss"
        else:
            model = HistGradientBoostingRegressor(
                max_iter=self.max_iter, random_state=self.random_state
            )
            scoring = "neg_mean_squared_error"
        model.fit(X_fit, y_fit)
        all_importance = permutation_importance(
            model, X_valid, y_valid, scoring=scoring,
            n_repeats=self.n_repeats, random_state=self.random_state, n_jobs=1,
        ).importances_mean
        importance = all_importance[:X.shape[1]]
        ranked = np.argsort(-importance, kind="stable")
        self.selected_indices_ = np.sort(ranked[:n_selected])
        self.feature_importances_ = np.asarray(importance)
        self.protected_attribute_importance_ = (
            float(all_importance[-1]) if A is not None else None
        )
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if not hasattr(self, "selected_indices_"):
            raise ValueError("PFIReducer is not fitted")
        return np.asarray(X, dtype=float)[:, self.selected_indices_]

    def get_support(self, indices=False):
        """Return selected-coordinate indices or a Boolean mask."""
        if not hasattr(self, "selected_indices_"):
            raise ValueError("PFIReducer is not fitted")
        if indices:
            return self.selected_indices_.copy()
        support = np.zeros(self.n_features_in_, dtype=bool)
        support[self.selected_indices_] = True
        return support


def make_mediator_reducer(method="pfi", n_components=0.20, *, random_state=0,
                          **kwargs):
    """Construct a fold-compatible PCA or PFI mediator reducer."""
    method = method.lower()
    if method == "pca":
        return PCA(n_components=n_components, random_state=random_state, **kwargs)
    if method == "pfi":
        return PFIReducer(
            n_components=n_components, random_state=random_state, **kwargs
        )
    raise ValueError("method must be 'pca' or 'pfi'")


def fit_external_reducer(train, mediator_columns, *, method="pca",
                         n_components=0.70, outcome=None, random_state=0,
                         output_prefix="mediator_pc", protected_attribute=None,
                         **kwargs):
    """Fit a mediator reducer using only external-training rows."""
    columns = tuple(mediator_columns)
    X = np.asarray(train[list(columns)], dtype=float)
    if X.ndim != 2 or not columns or not np.isfinite(X).all():
        raise ValueError("Mediators must be a nonempty finite numeric matrix")
    reducer = make_mediator_reducer(
        method, n_components=n_components, random_state=random_state, **kwargs
    )
    y = None if outcome is None else train[outcome].to_numpy()
    A = None if protected_attribute is None else train[protected_attribute].to_numpy()
    reducer.fit(X, y, A=A) if method.lower() == "pfi" else reducer.fit(X, y)
    if method.lower() == "pca":
        output_columns = tuple(
            f"{output_prefix}_{index:03d}" for index in range(reducer.n_components_)
        )
        diagnostics = {
            "n_input_features": len(columns),
            "n_output_features": int(reducer.n_components_),
            "explained_variance": float(reducer.explained_variance_ratio_.sum()),
        }
    else:
        output_columns = tuple(columns[index] for index in reducer.get_support(indices=True))
        diagnostics = {
            "n_input_features": len(columns),
            "n_output_features": len(output_columns),
        }
    return {
        "method": method.lower(),
        "input_columns": columns,
        "output_columns": output_columns,
        "transformer": reducer,
        "fit_n": len(train),
        "diagnostics": diagnostics,
    }


def apply_external_reducer(data, fitted):
    """Apply a train-fitted reducer and return aligned mediator columns."""
    values = fitted["transformer"].transform(
        np.asarray(data[list(fitted["input_columns"])], dtype=float)
    )
    return pd.DataFrame(
        np.asarray(values, dtype=float),
        columns=list(fitted["output_columns"]),
        index=data.index,
    )


def reduce_external_workflow_data(workflow, *, method="pca", n_components=0.70,
                                  random_state=0, output_prefix="mediator_pc",
                                  **kwargs):
    """Reduce a workflow's mediators using only its external training rows."""
    mediators = tuple(workflow["mediators"])
    fitted = fit_external_reducer(
        workflow["train"], mediators, method=method, n_components=n_components,
        outcome=workflow["outcome"], random_state=random_state,
        output_prefix=output_prefix,
        protected_attribute=workflow.get("protected_attribute"), **kwargs,
    )
    reduced = {}
    for partition in ("train", "test"):
        data = workflow[partition]
        transformed = apply_external_reducer(data, fitted)
        reduced[partition] = pd.concat(
            [data.drop(columns=list(mediators)), transformed], axis=1
        )
    return {
        **workflow,
        "train": reduced["train"],
        "test": reduced["test"],
        "mediators": fitted["output_columns"],
    }, fitted
