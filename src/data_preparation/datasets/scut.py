"""Functional preparation of SCUT labels, demographics, and ArcFace embeddings."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..contract import assign_variable_roles, make_dataset, validate_dataset
from ..embeddings import attach_embeddings, generate_scut_embeddings, load_embeddings
from ..splitting import apply_split, make_external_split


SCUT_EMBEDDING_MODEL = "insightface/buffalo_l"
SCUT_LABEL_FILE = Path("train_test_files") / "All_labels.txt"
SCUT_SPLIT_DIR = Path("train_test_files") / "split_of_60%training and 40%testing"
SCUT_CACHE_FILE = "arcface_buffalo_l_padded_v1_embeddings.npz"


def _read_partition(path):
    """Read ordered filenames from an official SCUT partition file."""
    return tuple(
        line.split()[0] for line in Path(path).read_text().splitlines() if line.strip()
    )


def load_scut_metadata(data_dir):
    """Load labels, filename-derived demographics, and the official split."""
    data_dir = Path(data_dir).expanduser()
    label_path = data_dir / SCUT_LABEL_FILE
    split_dir = data_dir / SCUT_SPLIT_DIR
    required = (label_path, split_dir / "train.txt", split_dir / "test.txt")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing SCUT files: {missing}")
    labels = pd.read_csv(label_path, sep=r"\s+", names=["filename", "Y"])
    if labels["filename"].duplicated().any():
        raise ValueError("SCUT labels contain duplicate filenames")
    parts = labels["filename"].str.extract(r"^(?P<ethnicity>[AC])(?P<gender>[FM])")
    labels["ethnicity"] = parts["ethnicity"].map({"A": "Asian", "C": "Caucasian"})
    labels["gender"] = parts["gender"].map({"F": "Female", "M": "Male"})
    labels["A"] = labels["gender"].map({"Male": 0, "Female": 1})
    if labels[["ethnicity", "gender", "A", "Y"]].isna().any().any():
        raise ValueError("SCUT filenames or labels have invalid demographic/outcome coding")
    labels["A"] = labels["A"].astype(int)
    train_names = _read_partition(split_dir / "train.txt")
    test_names = _read_partition(split_dir / "test.txt")
    if set(train_names).intersection(test_names):
        raise ValueError("Official SCUT train and test partitions overlap")
    if len(train_names) != len(set(train_names)) or len(test_names) != len(set(test_names)):
        raise ValueError("Official SCUT partitions contain duplicate filenames")
    return labels, {"train_filenames": train_names, "test_filenames": test_names}


def construct_scut_cohort(metadata, partitions, embeddings):
    """Align official partitions, labels, and successfully generated embeddings."""
    names = (*partitions["train_filenames"], *partitions["test_filenames"])
    label_lookup = metadata.set_index("filename")
    missing_labels = [name for name in names if name not in label_lookup.index]
    if missing_labels:
        raise ValueError(f"Partition filenames lack labels; first: {missing_labels[:5]}")
    identifiers = embeddings.get("identifiers")
    if identifiers is None:
        raise ValueError("SCUT embeddings require filename identifiers")
    embedding_lookup = {name: index for index, name in enumerate(identifiers.astype(str))}
    embedded_names = [name for name in names if name in embedding_lookup]
    excluded = [name for name in names if name not in embedding_lookup]
    selected = label_lookup.loc[embedded_names].reset_index()
    selected["partition"] = np.where(
        selected["filename"].isin(partitions["train_filenames"]), "train", "test"
    )
    row_ids = np.asarray([embedding_lookup[name] for name in embedded_names], dtype=int)
    data, mediator_columns = attach_embeddings(
        selected[["filename", "partition", "A", "Y", "gender", "ethnicity"]],
        embeddings, row_ids=row_ids, prefix="embedding",
    )
    audit = {
        "label_rows": len(metadata),
        "official_train_rows": len(partitions["train_filenames"]),
        "official_test_rows": len(partitions["test_filenames"]),
        "analysis_rows": len(data),
        "embedding_failures": tuple(excluded),
    }
    return data, mediator_columns, audit


def prepare_scut(data_dir, *, embedding_path=None, image_dir=None,
                 face_app=None, image_reader=None):
    """Return the aligned SCUT contract, generating ArcFace embeddings if absent."""
    data_dir = Path(data_dir).expanduser()
    metadata, partitions = load_scut_metadata(data_dir)
    embedding_path = data_dir / SCUT_CACHE_FILE if embedding_path is None else Path(embedding_path)
    if embedding_path.is_file():
        embeddings = load_embeddings(embedding_path)
    else:
        image_dir = data_dir / "Images" if image_dir is None else Path(image_dir)
        image_paths = [image_dir / filename for filename in metadata["filename"]]
        embeddings = generate_scut_embeddings(
            image_paths, metadata["filename"], embedding_path,
            face_app=face_app, image_reader=image_reader,
        )
    data, mediator_columns, audit = construct_scut_cohort(metadata, partitions, embeddings)
    roles = assign_variable_roles(
        outcome="Y", protected_attribute="A", mediators=mediator_columns,
        confounders=("ethnicity",), identifiers=("filename",),
    )
    dataset = make_dataset(
        "SCUT", data, roles, copy_data=False,
        provenance={
            "data_dir": str(data_dir),
            "embedding_source": str(embedding_path),
            "embedding_model": SCUT_EMBEDDING_MODEL,
            "protected_coding": {0: "Male", 1: "Female"},
            "outcome_definition": "dataset-provided mean attractiveness rating",
            "confounder_definition": {"ethnicity": ("Asian", "Caucasian")},
            "missing_policy": "exclude images without valid labels, coding, or face embedding",
            "cohort_audit": audit,
        },
    )
    validate_dataset(dataset, require_complete=("A", "Y", "ethnicity", *mediator_columns))
    return dataset


def make_scut_workflow_data(dataset):
    """Return full assessment data and the untouched official external split."""
    roles, full = dataset["roles"], dataset["data"]
    train_index = np.flatnonzero(full["partition"].to_numpy() == "train")
    test_index = np.flatnonzero(full["partition"].to_numpy() == "test")
    split = make_external_split(full, predefined=(train_index, test_index))
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
