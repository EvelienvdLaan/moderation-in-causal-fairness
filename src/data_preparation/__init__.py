"""Functional data preparation shared by all analysis workflows."""

from .contract import assign_variable_roles, make_assessment_data, make_dataset, validate_dataset
from .datasets.compas import (
    COMPAS_CONFOUNDERS,
    COMPAS_MEDIATORS,
    construct_compas_cohort,
    load_compas_source,
    make_compas_workflow_data,
    prepare_compas,
)
from .embeddings import (
    attach_embeddings,
    generate_reddit_embeddings,
    generate_scut_embeddings,
    load_embeddings,
)
from .datasets.law_school import (
    LAW_SCHOOL_CONFOUNDERS,
    LAW_SCHOOL_MEDIATORS,
    construct_law_school_cohort,
    load_law_school_source,
    make_law_school_workflow_data,
    prepare_law_school,
)
from .datasets.reddit import (
    REDDIT_EMBEDDING_MODEL,
    construct_reddit_cohort,
    load_reddit_metadata,
    make_reddit_workflow_data,
    prepare_reddit,
)
from .reduction import (
    PFIReducer,
    apply_external_reducer,
    fit_external_reducer,
    reduce_external_workflow_data,
    make_mediator_reducer,
)
from .splitting import apply_split, make_external_split
from .simulation import (
    generate_binary_mitigation_data,
    generate_binary_reference_data,
    generate_embedding_like_reference_data,
)
from .datasets.scut import (
    SCUT_EMBEDDING_MODEL,
    construct_scut_cohort,
    load_scut_metadata,
    make_scut_workflow_data,
    prepare_scut,
)

__all__ = [
    "apply_external_reducer",
    "apply_split",
    "assign_variable_roles",
    "attach_embeddings",
    "COMPAS_CONFOUNDERS",
    "COMPAS_MEDIATORS",
    "construct_compas_cohort",
    "construct_reddit_cohort",
    "construct_scut_cohort",
    "fit_external_reducer",
    "generate_reddit_embeddings",
    "generate_scut_embeddings",
    "generate_binary_reference_data",
    "generate_binary_mitigation_data",
    "generate_embedding_like_reference_data",
    "LAW_SCHOOL_CONFOUNDERS",
    "LAW_SCHOOL_MEDIATORS",
    "load_embeddings",
    "load_compas_source",
    "load_law_school_source",
    "load_reddit_metadata",
    "load_scut_metadata",
    "make_dataset",
    "make_assessment_data",
    "make_external_split",
    "make_mediator_reducer",
    "make_compas_workflow_data",
    "make_law_school_workflow_data",
    "make_reddit_workflow_data",
    "make_scut_workflow_data",
    "prepare_compas",
    "prepare_law_school",
    "prepare_reddit",
    "prepare_scut",
    "PFIReducer",
    "REDDIT_EMBEDDING_MODEL",
    "reduce_external_workflow_data",
    "SCUT_EMBEDDING_MODEL",
    "construct_law_school_cohort",
    "validate_dataset",
]
