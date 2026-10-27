"""Functional preparation of Reddit metadata and MiniLM embeddings."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..contract import assign_variable_roles, make_dataset, validate_dataset
from ..embeddings import attach_embeddings, load_embeddings
from ..splitting import apply_split, make_external_split


REDDIT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_reddit_metadata(path):
    """Load the row identifiers, gender, body validity, and score from JSON lines."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            record = json.loads(line)
            try:
                score = float(record.get("score"))
            except (TypeError, ValueError):
                score = np.nan
            records.append((row_index, record.get("gender"), record.get("body") is not None, score))
    return pd.DataFrame(records, columns=["row_index", "gender", "body_valid", "score_raw"])


def construct_reddit_cohort(metadata, *, sample_size=None, random_state=2026):
    """Apply eligibility and optional protected-group-stratified subsampling."""
    required = {"row_index", "gender", "body_valid", "score_raw"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Missing Reddit metadata columns: {missing}")
    data = metadata.copy()
    counts = {"source_rows": len(data)}
    data = data[
        data["gender"].isin(["male", "female"])
        & data["body_valid"]
        & np.isfinite(data["score_raw"])
    ].copy()
    counts["eligible_rows"] = len(data)
    data["A"] = data["gender"].map({"male": 0, "female": 1}).astype(int)
    data["Y"] = np.sign(data["score_raw"]) * np.log1p(np.abs(data["score_raw"]))
    if sample_size is not None:
        if not 0 < sample_size <= len(data):
            raise ValueError("sample_size must be positive and no larger than the eligible cohort")
        if sample_size < len(data):
            data, _ = train_test_split(
                data, train_size=sample_size, random_state=random_state, stratify=data["A"],
            )
    data = data.sort_values("row_index").reset_index(drop=True)
    counts["analysis_rows"] = len(data)
    return data, {
        "counts": counts,
        "excluded_ineligible": counts["source_rows"] - counts["eligible_rows"],
        "sample_size": sample_size,
        "sampling_random_state": random_state if sample_size is not None else None,
    }


def prepare_reddit(metadata_path, embedding_path, *, sample_size=None, random_state=2026):
    """Return the aligned Reddit contract, using all eligible posts by default."""
    metadata = load_reddit_metadata(metadata_path)
    embeddings = load_embeddings(embedding_path, mmap=True)
    if len(embeddings["matrix"]) != len(metadata):
        raise ValueError("Reddit metadata and embedding cache row counts differ")
    cohort, audit = construct_reddit_cohort(
        metadata, sample_size=sample_size, random_state=random_state,
    )
    data, mediator_columns = attach_embeddings(
        cohort[["row_index", "gender", "A", "Y", "score_raw"]], embeddings,
        row_ids=cohort["row_index"].to_numpy(), prefix="embedding",
    )
    roles = assign_variable_roles(
        outcome="Y", protected_attribute="A", mediators=mediator_columns,
        confounders=(), identifiers=("row_index",),
    )
    dataset = make_dataset(
        "Reddit", data, roles, copy_data=False,
        provenance={
            "metadata_source": str(Path(metadata_path).expanduser()),
            "embedding_source": str(Path(embedding_path).expanduser()),
            "embedding_model": REDDIT_EMBEDDING_MODEL,
            "protected_coding": {0: "Male", 1: "Female"},
            "outcome_definition": "sign(score) * log1p(abs(score))",
            "missing_policy": "exclude invalid gender, body, or score before sampling",
            "cohort_audit": audit,
        },
    )
    validate_dataset(dataset, require_complete=("A", "Y", *mediator_columns))
    return dataset


def make_reddit_workflow_data(dataset, *, test_size=0.30, random_state=2026):
    """Return full-cohort assessment and one shared A-stratified external split."""
    roles, full = dataset["roles"], dataset["data"]
    split = make_external_split(
        full, test_size=test_size, stratify=(roles["protected_attribute"],),
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
