"""Dataset-specific functional preparation modules."""

from .compas import (
    COMPAS_CONFOUNDERS,
    COMPAS_MEDIATORS,
    construct_compas_cohort,
    load_compas_source,
    make_compas_workflow_data,
    prepare_compas,
)
from .law_school import (
    LAW_SCHOOL_CONFOUNDERS,
    LAW_SCHOOL_MEDIATORS,
    construct_law_school_cohort,
    load_law_school_source,
    make_law_school_workflow_data,
    prepare_law_school,
)
from .reddit import (
    REDDIT_EMBEDDING_MODEL,
    construct_reddit_cohort,
    load_reddit_metadata,
    make_reddit_workflow_data,
    prepare_reddit,
)
from .scut import (
    SCUT_EMBEDDING_MODEL,
    construct_scut_cohort,
    load_scut_metadata,
    make_scut_workflow_data,
    prepare_scut,
)

__all__ = [
    "COMPAS_CONFOUNDERS",
    "COMPAS_MEDIATORS",
    "LAW_SCHOOL_CONFOUNDERS",
    "LAW_SCHOOL_MEDIATORS",
    "REDDIT_EMBEDDING_MODEL",
    "SCUT_EMBEDDING_MODEL",
    "construct_compas_cohort",
    "construct_law_school_cohort",
    "construct_reddit_cohort",
    "construct_scut_cohort",
    "load_compas_source",
    "load_law_school_source",
    "load_reddit_metadata",
    "load_scut_metadata",
    "make_compas_workflow_data",
    "make_law_school_workflow_data",
    "make_reddit_workflow_data",
    "make_scut_workflow_data",
    "prepare_compas",
    "prepare_law_school",
    "prepare_reddit",
    "prepare_scut",
]
