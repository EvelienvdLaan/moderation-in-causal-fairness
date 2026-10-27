"""Deterministic effect definitions from the four nested means."""


EFFECT_COEFFICIENTS = {
    "nde_m0": {"psi_10": 1, "psi_00": -1},
    "nde_m1": {"psi_11": 1, "psi_01": -1},
    "nie_a0": {"psi_01": 1, "psi_00": -1},
    "nie_a1": {"psi_11": 1, "psi_10": -1},
    "symmetric_nde": {"psi_10": .5, "psi_00": -.5, "psi_11": .5, "psi_01": -.5},
    "symmetric_nie": {"psi_01": .5, "psi_00": -.5, "psi_11": .5, "psi_10": -.5},
    "interaction": {"psi_11": 1, "psi_10": -1, "psi_01": -1, "psi_00": 1},
    "total_effect": {"psi_11": 1, "psi_00": -1},
}


def effects_from_nested_means(nested_means):
    """Calculate reference-specific effects from ``psi_00`` ... ``psi_11``."""
    p00, p01 = nested_means["psi_00"], nested_means["psi_01"]
    p10, p11 = nested_means["psi_10"], nested_means["psi_11"]
    nde_m0, nde_m1 = p10 - p00, p11 - p01
    nie_a0, nie_a1 = p01 - p00, p11 - p10
    symmetric_nde = .5 * (nde_m0 + nde_m1)
    symmetric_nie = .5 * (nie_a0 + nie_a1)
    te = p11 - p00
    interaction_nie = nie_a1 - nie_a0
    interaction_nde = nde_m1 - nde_m0
    return {
        "nde_m0": nde_m0, "nde_m1": nde_m1,
        "nie_a0": nie_a0, "nie_a1": nie_a1,
        "symmetric_nde": symmetric_nde,
        "symmetric_nie": symmetric_nie,
        "interaction": interaction_nie,
        "interaction_via_nie": interaction_nie,
        "interaction_via_nde": interaction_nde,
        "total_effect": te,
        "decomposition_m0": nde_m0 + nie_a1,
        "decomposition_m1": nie_a0 + nde_m1,
        "decomposition_error_m0": te - (nde_m0 + nie_a1),
        "decomposition_error_m1": te - (nie_a0 + nde_m1),
    }
