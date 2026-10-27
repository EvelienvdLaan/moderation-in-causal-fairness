"""External train/test splitting with explicit returned indices."""

import numpy as np
from sklearn.model_selection import train_test_split


def make_external_split(data, *, test_size=0.30, stratify=(), random_state=0,
                        predefined=None):
    """Return fixed positional train/test indices and the split policy."""
    if predefined is not None:
        train_index, test_index = (np.asarray(values, dtype=int) for values in predefined)
        source = "predefined"
    else:
        positions = np.arange(len(data))
        strata = None
        if stratify:
            strata = data[list(stratify)].astype(str).agg("|".join, axis=1)
        train_index, test_index = train_test_split(
            positions, test_size=test_size, random_state=random_state, stratify=strata,
        )
        source = "random"
    if set(train_index).intersection(test_index):
        raise ValueError("Train and test indices overlap")
    if len(train_index) + len(test_index) != len(data):
        raise ValueError("Split must partition every supplied row exactly once")
    return {
        "train_index": np.asarray(train_index, dtype=int),
        "test_index": np.asarray(test_index, dtype=int),
        "policy": {
            "source": source,
            "test_size": None if predefined is not None else float(test_size),
            "stratify": tuple(stratify),
            "random_state": None if predefined is not None else int(random_state),
        },
    }


def apply_split(data, split):
    """Return independent train and test frames from positional indices."""
    train = data.iloc[split["train_index"]].reset_index(drop=True).copy()
    test = data.iloc[split["test_index"]].reset_index(drop=True).copy()
    return train, test
