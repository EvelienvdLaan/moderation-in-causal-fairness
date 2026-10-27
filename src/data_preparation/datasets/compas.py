"""Functional preparation of the approved COMPAS analysis cohort."""

from pathlib import Path

import pandas as pd

from ..contract import assign_variable_roles, make_dataset, validate_dataset
from ..splitting import apply_split, make_external_split


COMPAS_MEDIATORS = (
    "priors_count", "juv_fel_count", "juv_misd_count",
    "juv_other_count", "c_charge_degree",
)
COMPAS_CONFOUNDERS = ("age", "race")
COMPAS_OUTPUT_COLUMNS = (
    "id", "sex", "A", "Y", "age", "juv_fel_count", "juv_misd_count",
    "juv_other_count", "priors_count", "race", "c_charge_degree",
)
COMPAS_REQUIRED_COLUMNS = (
    "id", "sex", "is_recid", "days_b_screening_arrest", "score_text",
    *COMPAS_MEDIATORS, *COMPAS_CONFOUNDERS,
)


def load_compas_source(path):
    """Load the unfiltered COMPAS CSV and check its source schema."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    missing = sorted(set(COMPAS_REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing COMPAS columns: {missing}")
    return data


def construct_compas_cohort(source):
    """Apply the approved ProPublica-style eligibility and complete-case rules."""
    missing = sorted(set(COMPAS_REQUIRED_COLUMNS) - set(source.columns))
    if missing:
        raise ValueError(f"Missing COMPAS columns: {missing}")
    data = source.copy()
    counts = {"source_rows": len(data)}

    data = data[data["days_b_screening_arrest"].between(-30, 30)].copy()
    counts["after_screening_window"] = len(data)
    data = data[data["is_recid"] != -1].copy()
    counts["after_valid_recidivism"] = len(data)
    data = data[data["c_charge_degree"] != "O"].copy()
    counts["after_valid_charge_degree"] = len(data)
    data = data[data["score_text"] != "N/A"].copy()
    counts["after_available_score"] = len(data)
    data = data.drop_duplicates("id").copy()
    counts["after_unique_person"] = len(data)
    data = data[data["sex"].isin(["Male", "Female"])].copy()
    counts["after_binary_recorded_sex"] = len(data)

    data["A"] = (data["sex"] == "Female").astype(int)
    data["Y"] = pd.to_numeric(data["is_recid"], errors="raise").astype(int)
    cohort = data[list(COMPAS_OUTPUT_COLUMNS)].dropna().reset_index(drop=True)
    counts["complete_case_rows"] = len(cohort)
    audit = {
        "counts": counts,
        "excluded_total": counts["source_rows"] - counts["complete_case_rows"],
        "cohort_rules": (
            "days_b_screening_arrest in [-30, 30]",
            "is_recid != -1",
            "c_charge_degree != 'O'",
            "score_text != 'N/A'",
            "first eligible row per id",
            "sex in {'Male', 'Female'}",
            "complete cases for A, Y, M, and C",
        ),
    }
    return cohort, audit


def prepare_compas(path):
    """Load COMPAS and return its validated shared dataset contract."""
    source = load_compas_source(path)
    cohort, audit = construct_compas_cohort(source)
    roles = assign_variable_roles(
        outcome="Y", protected_attribute="A", mediators=COMPAS_MEDIATORS,
        confounders=COMPAS_CONFOUNDERS, identifiers=("id",),
    )
    dataset = make_dataset(
        "COMPAS", cohort, roles,
        provenance={
            "source": str(Path(path).expanduser()),
            "protected_coding": {0: "Male", 1: "Female"},
            "outcome_definition": "binary is_recid",
            "missing_policy": "complete case after cohort construction",
            "cohort_audit": audit,
        },
    )
    validate_dataset(dataset, require_complete=tuple(cohort.columns))
    return dataset


def make_compas_workflow_data(dataset, *, test_size=0.30, random_state=2026):
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
