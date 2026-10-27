"""Reproducible data generators for causal-effect assessment."""

import numpy as np
import pandas as pd

from .contract import assign_variable_roles, make_dataset, validate_dataset


def _effects_from_truth_means(means):
    """Derive compatible effects from four population nested means."""
    nde_m0 = means["psi_10"] - means["psi_00"]
    nde_m1 = means["psi_11"] - means["psi_01"]
    nie_a0 = means["psi_01"] - means["psi_00"]
    nie_a1 = means["psi_11"] - means["psi_10"]
    total = means["psi_11"] - means["psi_00"]
    interaction_nie = nie_a1-nie_a0
    interaction_nde = nde_m1-nde_m0
    return {
        "nde_m0": nde_m0, "nde_m1": nde_m1,
        "nie_a0": nie_a0, "nie_a1": nie_a1,
        "interaction": interaction_nie,
        "interaction_via_nie": interaction_nie,
        "interaction_via_nde": interaction_nde,
        "total_effect": total,
        "decomposition_m0": nde_m0+nie_a1,
        "decomposition_m1": nie_a0+nde_m1,
        "decomposition_error_m0": total-(nde_m0+nie_a1),
        "decomposition_error_m1": total-(nie_a0+nde_m1),
    }


def _simulation_contract(name, data, mediators, truth, parameters):
    """Return a validated shared dataset contract with known truth."""
    roles = assign_variable_roles(
        outcome="Y", protected_attribute="A", mediators=mediators,
        confounders=("C",) if "C" in data else (),
    )
    dataset = make_dataset(
        name, data, roles,
        provenance={"parameters": parameters, "outcome_definition": name},
    )
    validate_dataset(dataset, require_complete=("A", "Y", *mediators, *roles["confounders"]))
    return {**dataset, "truth": truth}


def generate_binary_reference_data(
        n=10_000, *, mediator_dim=1, exposure_logit_effect=4.0,
        mediator_logit_effect=2.0, mediator_intercept=-3.0,
        mediator_shift=1.0, mediator_noise_sd=0.7, random_state=31):
    """Generate the additive-logit reference-sensitivity example and MC truth."""
    if n < 2 or mediator_dim < 1 or mediator_noise_sd < 0:
        raise ValueError("Invalid sample size, mediator dimension, or noise")

    rng = np.random.default_rng(random_state)
    A = rng.binomial(1, 0.2, size=n)
    U = rng.normal(scale=mediator_noise_sd, size=(n, mediator_dim))

    def resize(values):
        return np.pad(
            values[:mediator_dim],
            (0, max(0, mediator_dim - len(values)))
        )

    mediator_shifts = mediator_shift * resize(np.array([
        1.0, 0.6, 0.0, 0.0, 0.2, 3.0
    ]))

    effect_pattern = resize(np.array([
        0.30, 0.20, 0.25, -0.15, 0.0, 0.40
    ]))

    # Scale coefficients so they sum to mediator_logit_effect.
    mediator_effects = (
        mediator_logit_effect
        * effect_pattern
        / effect_pattern.sum()
    )

    M = (
        mediator_intercept
        + A[:, None] * mediator_shifts
        + U
    )

    linear = exposure_logit_effect * A + M @ mediator_effects
    probability = 1 / (1 + np.exp(-linear))
    Y = rng.binomial(1, probability)

    mediators = tuple(f"M_{index:03d}" for index in range(mediator_dim))
    data = pd.DataFrame(M, columns=mediators)
    data.insert(0, "A", A)
    data["Y"] = Y

    means = {}
    for a in (0, 1):
        for reference in (0, 1):
            mediator = (
                mediator_intercept
                + reference * mediator_shifts
                + U
            )
            value = 1 / (
                1 + np.exp(-(
                    exposure_logit_effect * a
                    + mediator @ mediator_effects
                ))
            )
            means[f"psi_{a}{reference}"] = float(value.mean())

    truth = {
        "nested_means": means,
        "effects": _effects_from_truth_means(means),
    }

    parameters = {
        "mediator_dim": int(mediator_dim),
        "exposure_logit_effect": float(exposure_logit_effect),
        "mediator_logit_effect": float(mediator_logit_effect),
        "mediator_effects": mediator_effects.tolist(),
        "mediator_intercept": float(mediator_intercept),
        "mediator_shift": float(mediator_shift),
        "mediator_shifts": mediator_shifts.tolist(),
        "mediator_noise_sd": float(mediator_noise_sd),
        "random_state": int(random_state),
        "extra_mediator_policy": "mixed exposure and outcome associations",
    }

    return _simulation_contract(
        "binary additive-logit simulation",
        data, mediators, truth, parameters
    )


def generate_binary_mitigation_data(
        n=6000, *, mediator_dim=1, mediator_intercept=-1.2, mediator_shift=1.4,
        mediator_noise_sd=0.7, outcome_intercept=-0.5,
        mediator_logit_effect=1.5, exposure_logit_effect=1.4,
        interaction_logit_effect=-0.5, truth_n=1_000_000,
        truth_random_state=2718, random_state=7):
    """Generate the binary interaction model used by mitigation simulations."""
    if n < 2 or truth_n < 2 or mediator_dim < 1 or mediator_noise_sd < 0:
        raise ValueError("Sample sizes and dimension must be positive and noise nonnegative")
    rng = np.random.default_rng(random_state)
    A = rng.binomial(1, 0.5, n)
    M = mediator_intercept + mediator_shift*A[:, None] + rng.normal(
        scale=mediator_noise_sd, size=(n, mediator_dim)
    )
    linear = (
        outcome_intercept + mediator_logit_effect*M[:, 0]
        + exposure_logit_effect*A + interaction_logit_effect*A*M[:, 0]
    )
    probability = 1 / (1 + np.exp(-linear))
    Y = rng.binomial(1, probability)
    mediators = (
        ("M",) if mediator_dim == 1
        else tuple(f"M_{index:03d}" for index in range(mediator_dim))
    )
    data = pd.DataFrame(M, columns=mediators)
    data.insert(0, "A", A)
    data["Y"] = Y

    truth_rng = np.random.default_rng(truth_random_state)
    mediator_noise = truth_rng.normal(scale=mediator_noise_sd, size=truth_n)
    means = {}
    for a in (0, 1):
        for reference in (0, 1):
            mediator = mediator_intercept + mediator_shift*reference + mediator_noise
            logit = (
                outcome_intercept + mediator_logit_effect*mediator
                + exposure_logit_effect*a + interaction_logit_effect*a*mediator
            )
            means[f"psi_{a}{reference}"] = float(
                np.mean(1 / (1 + np.exp(-logit)))
            )
    truth = {"nested_means": means, "effects": _effects_from_truth_means(means)}
    parameters = {
        "mediator_dim": int(mediator_dim),
        "mediator_intercept": float(mediator_intercept),
        "mediator_shift": float(mediator_shift),
        "mediator_noise_sd": float(mediator_noise_sd),
        "outcome_intercept": float(outcome_intercept),
        "mediator_logit_effect": float(mediator_logit_effect),
        "exposure_logit_effect": float(exposure_logit_effect),
        "interaction_logit_effect": float(interaction_logit_effect),
        "truth_n": int(truth_n),
        "truth_random_state": int(truth_random_state),
        "random_state": int(random_state),
        "truth_method": "counterfactual Monte Carlo on the probability scale",
        "extra_mediator_policy": "exposure-associated and outcome-null",
    }
    return _simulation_contract(
        "binary mitigation interaction simulation", data, mediators, truth, parameters
    )


def _build_mlp(rng, input_dim, hidden_dims, output_dim):
    """Initialize a small fully connected neural network."""
    dimensions = (input_dim, *hidden_dims, output_dim)
    return tuple(
        (
            rng.normal(scale=np.sqrt(2/max(1, fan_in)), size=(fan_in, fan_out)),
            np.zeros(fan_out),
        )
        for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:])
    )


def _mlp_forward(values, parameters):
    """Apply tanh hidden layers and a linear output layer."""
    output = np.asarray(values)
    for index, (weights, bias) in enumerate(parameters):
        output = output@weights + bias
        if index < len(parameters)-1:
            output = np.tanh(output)
    return output


def _sigmoid(values):
    """Apply a numerically stable logistic transform."""
    values = np.clip(values, -35, 35)
    return 1/(1 + np.exp(-values))


def generate_embedding_like_reference_data(
        n=10_000, *, mediator_dim=500, mediator_hidden_dims=(16,),
        outcome_hidden_dims=(16,), treatment_noise_sd=0.5,
        mediator_noise_sd=0.5, outcome_noise_sd=0.35, random_state=42):
    """Generate nonlinear high-dimensional mediation data with MLPs."""
    if n < 2 or mediator_dim < 1:
        raise ValueError("Require n >= 2 and mediator_dim >= 1")
    if min(treatment_noise_sd, mediator_noise_sd, outcome_noise_sd) < 0:
        raise ValueError("Noise scales must be nonnegative")
    if any(width < 1 for width in (*mediator_hidden_dims, *outcome_hidden_dims)):
        raise ValueError("Hidden-layer widths must be positive")

    rng = np.random.default_rng(random_state)
    treatment_network = _build_mlp(rng, 1, (8,), 1)
    mediator_network = _build_mlp(
        rng, 2, tuple(mediator_hidden_dims), mediator_dim
    )
    outcome_network = _build_mlp(
        rng, mediator_dim, tuple(outcome_hidden_dims), 1
    )
    confounder_network = _build_mlp(rng, 1, (8,), 1)

    C = rng.normal(size=(n, 1))
    treatment_noise = rng.normal(scale=treatment_noise_sd, size=(n, 1))
    mediator_noise = rng.normal(scale=mediator_noise_sd, size=(n, mediator_dim))
    outcome_noise = rng.normal(scale=outcome_noise_sd, size=(n, 1))
    propensity = _sigmoid(_mlp_forward(C, treatment_network) + treatment_noise)
    A = rng.binomial(1, propensity) # .astype(float)

    def mediator_under(treatment):
        inputs = np.column_stack((treatment.reshape(-1), C.reshape(-1)))
        deterministic_signal = _mlp_forward(inputs, mediator_network)
        return (
            deterministic_signal / np.sqrt(mediator_dim)
            + mediator_noise
            )

    M0 = mediator_under(np.zeros_like(A))
    M1 = mediator_under(np.ones_like(A))
    M = np.where(A == 1, M1, M0)

    # Direction in M that is affected by A
    effect_direction = np.mean(M1 - M0, axis=0)
    effect_direction /= np.linalg.norm(effect_direction) + 1e-8

    # Scale using the M0 distribution
    score_m0 = M0 @ effect_direction
    score_center = score_m0.mean()
    score_scale = score_m0.std() + 1e-8

    def outcome_probability(treatment, mediators):
        mediator_score = (
            mediators @ effect_direction - score_center
        ) / score_scale
        mediator_score = mediator_score.reshape(-1, 1)

        confounder_signal = _mlp_forward(C, confounder_network)

        logit = (
            -0.3
            + 0.2 * treatment
            + 1 * mediator_score
            + 0.3 * confounder_signal
            + 3.0 * treatment * mediator_score
            + outcome_noise
        )

        return _sigmoid(logit).reshape(-1)

    counterfactuals = {
        f"psi_{a}{reference}": outcome_probability(
            np.full_like(A, a), M0 if reference == 0 else M1
        )
        for a in (0, 1) for reference in (0, 1)
    }
    observed_probability = outcome_probability(A, M)
    Y = rng.binomial(1, observed_probability)
    mediators = tuple(f"M_{index:03d}" for index in range(mediator_dim))
    data = pd.DataFrame(M, columns=mediators)
    data.insert(0, "C", C.reshape(-1))
    data.insert(0, "A", A.astype(int).reshape(-1))
    data["Y"] = Y
    means = {name: float(values.mean()) for name, values in counterfactuals.items()}
    truth = {"nested_means": means, "effects": _effects_from_truth_means(means)}
    parameters = {
        "mediator_dim": int(mediator_dim),
        "mediator_hidden_dims": tuple(mediator_hidden_dims),
        "outcome_hidden_dims": tuple(outcome_hidden_dims),
        "treatment_noise_sd": float(treatment_noise_sd),
        "mediator_noise_sd": float(mediator_noise_sd),
        "outcome_noise_sd": float(outcome_noise_sd),
        "random_state": int(random_state),
        "treatment_network": treatment_network,
        "mediator_network": mediator_network,
        "outcome_network": outcome_network,
        "confounder_network": confounder_network,
        "outcome_interaction": "3.0 * A * outcome_network(M)",
        "construction": "nonlinear structural equations with fixed random MLPs",
    }
    return _simulation_contract(
        "neural-network high-dimensional simulation", data, mediators, truth, parameters
    )
