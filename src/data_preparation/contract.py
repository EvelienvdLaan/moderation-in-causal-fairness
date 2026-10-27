"""Small dictionary-based contracts for prepared datasets."""

import numpy as np
import pandas as pd


def assign_variable_roles(*, outcome, protected_attribute, mediators,
                          confounders=(), identifiers=()):
    """Return an explicit, immutable-by-convention role mapping."""
    return {
        "outcome": str(outcome),
        "protected_attribute": str(protected_attribute),
        "mediators": tuple(mediators),
        "confounders": tuple(confounders),
        "identifiers": tuple(identifiers),
    }


def make_dataset(name, data, roles, *, provenance=None, copy_data=True):
    """Return a row-aligned dataset contract without copying hidden state."""
    return {
        "name": str(name),
        "data": data.copy() if copy_data else data,
        "roles": dict(roles),
        "provenance": {} if provenance is None else dict(provenance),
    }


def validate_dataset(dataset, *, require_complete=()):
    """Validate roles, binary exposure coding, and requested completeness."""
    data, roles = dataset["data"], dataset["roles"]
    if not isinstance(data, pd.DataFrame):
        raise TypeError("dataset['data'] must be a pandas DataFrame")
    role_columns = [
        roles["outcome"], roles["protected_attribute"],
        *roles["mediators"], *roles["confounders"], *roles["identifiers"],
    ]
    missing_columns = sorted(set(role_columns) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing role columns: {missing_columns}")
    if len(role_columns) != len(set(role_columns)):
        raise ValueError("A variable cannot have more than one role")
    A = data[roles["protected_attribute"]].dropna().to_numpy()
    if set(np.unique(A)) != {0, 1}:
        raise ValueError("Protected attribute must contain both binary values 0 and 1")
    incomplete = [column for column in require_complete if data[column].isna().any()]
    if incomplete:
        raise ValueError(f"Required complete columns contain missing values: {incomplete}")
    return {
        "n_rows": len(data),
        "columns": tuple(data.columns),
        "missing_fraction": data[role_columns].isna().mean().to_dict(),
        "protected_counts": data[roles["protected_attribute"]].value_counts().sort_index().to_dict(),
    }


def make_assessment_data(dataset):
    """Return the full cohort and estimator inputs without copying or splitting."""
    roles = dataset["roles"]
    return {
        "data": dataset["data"],
        "outcome": roles["outcome"],
        "protected_attribute": roles["protected_attribute"],
        "mediators": roles["mediators"],
        "confounders": roles["confounders"],
    }
