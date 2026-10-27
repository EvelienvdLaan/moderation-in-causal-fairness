"""Functional preparation of the complete-case Law School cohort."""

from pathlib import Path

import pandas as pd

from ..contract import assign_variable_roles, make_dataset, validate_dataset
from ..splitting import apply_split, make_external_split


LAW_SCHOOL_MEDIATORS = (
    "lsat", "ugpa", "fam_inc", "fulltime", "tier", "zfygpa",
)
LAW_SCHOOL_CONFOUNDERS = ("sex", "DOB_yr")
LAW_SCHOOL_REQUIRED_COLUMNS = (
    "ID", "race1", "pass_bar", *LAW_SCHOOL_MEDIATORS, *LAW_SCHOOL_CONFOUNDERS,
)
LAW_SCHOOL_OUTPUT_COLUMNS = (
    "ID", "race1", "A", "Y", *LAW_SCHOOL_MEDIATORS,
    *LAW_SCHOOL_CONFOUNDERS, "race_binary",
)


def load_law_school_source(path):
    """Load the unfiltered Law School CSV and normalize column names."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    missing = sorted(set(LAW_SCHOOL_REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing Law School columns: {missing}")
    return data


def construct_law_school_cohort(source):
    """Apply valid exposure/outcome and complete-case eligibility rules."""
    data = source.copy()
    data.columns = data.columns.str.strip()
    missing = sorted(set(LAW_SCHOOL_REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing Law School columns: {missing}")
    counts = {"source_rows": len(data)}

    data["race1"] = data["race1"].astype("string").str.strip().str.lower()
    data["Y"] = pd.to_numeric(data["pass_bar"], errors="coerce")
    data = data[data["race1"].notna()].copy()
    counts["after_observed_race"] = len(data)
    data = data[data["Y"].isin([0, 1])].copy()
    counts["after_binary_bar_outcome"] = len(data)
    data["A"] = data["race1"].ne("white").astype(int)
    data["Y"] = data["Y"].astype(int)
    selected = [
        "ID", "race1", "A", "Y", *LAW_SCHOOL_MEDIATORS, *LAW_SCHOOL_CONFOUNDERS,
    ]
    data = data[selected].dropna().copy()
    counts["complete_case_rows"] = len(data)
    data["race_binary"] = data["A"].map({0: "White", 1: "Non-White"})
    cohort = data[list(LAW_SCHOOL_OUTPUT_COLUMNS)].reset_index(drop=True)
    audit = {
        "counts": counts,
        "excluded_total": counts["source_rows"] - counts["complete_case_rows"],
        "cohort_rules": (
            "observed race1",
            "pass_bar in {0, 1}",
            "complete cases for A, Y, M, and C",
        ),
    }
    return cohort, audit


def prepare_law_school(path):
    """Load Law School data and return its validated shared contract."""
    source = load_law_school_source(path)
    cohort, audit = construct_law_school_cohort(source)
    roles = assign_variable_roles(
        outcome="Y", protected_attribute="A", mediators=LAW_SCHOOL_MEDIATORS,
        confounders=LAW_SCHOOL_CONFOUNDERS, identifiers=("ID",),
    )
    dataset = make_dataset(
        "Law School", cohort, roles,
        provenance={
            "source": str(Path(path).expanduser()),
            "protected_coding": {0: "White", 1: "Non-White"},
            "outcome_definition": "binary eventual bar passage",
            "missing_policy": "complete case after cohort construction",
            "cohort_audit": audit,
        },
    )
    validate_dataset(dataset, require_complete=tuple(cohort.columns))
    return dataset


def make_law_school_workflow_data(dataset, *, test_size=0.30, random_state=2026):
    """Return full-cohort assessment and one shared external workflow split."""
    roles, full = dataset["roles"], dataset["data"].copy()
    split = make_external_split(
        full, test_size=test_size,
        stratify=(roles["protected_attribute"], roles["outcome"]),
        random_state=random_state,
    )
    train, test = apply_split(full, split)
    common = {
        "outcome": roles["outcome"],
        "protected_attribute": roles["protected_attribute"],
        "mediators": roles["mediators"],
        "confounders": roles["confounders"],
    }
    return {
        "assessment": {"data": full, **common},
        "prediction": {"train": train.copy(), "test": test.copy(), **common},
        "mitigation": {"train": train.copy(), "test": test.copy(), **common},
        "split": split,
    }
