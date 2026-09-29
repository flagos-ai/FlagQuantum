from __future__ import annotations

import math

import pytest
import torch

from flagquantum.simulation import (
    EvolutionValidationError,
    evolve_density_matrix,
    plan_density_matrix_evolution,
)

pytestmark = pytest.mark.unit

X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
Z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)
EXCITED = torch.tensor([0.0, 1.0], dtype=torch.complex128)


def test_driven_amplitude_damping_returns_bounded_trajectory() -> None:
    times = torch.linspace(0.0, 8.0, 161, dtype=torch.float64)
    result = evolve_density_matrix(
        (X + 0.2 * Z) / 2,
        EXCITED,
        1,
        times,
        [{"operator": "amplitude_damping", "rate": 0.1, "wire": 0}],
        {"z": Z},
        return_density_matrices=True,
    )

    assert result.times.shape == (161,)
    assert result.populations.shape == (161, 2)
    assert result.density_matrices is not None
    assert result.density_matrices.shape == (161, 2, 2)
    assert result.observables["z"].shape == (161,)
    assert result.population_bounded
    assert result.maximum_trace_drift <= result.trace_tolerance
    assert result.method == "runge-kutta"
    assert result.order == 4
    assert result.step_size == pytest.approx(0.025)
    assert result.precision == "complex128"


def test_zero_hamiltonian_matches_exponential_decay() -> None:
    gamma = 0.1
    times = torch.linspace(0.0, 8.0, 161, dtype=torch.float64)
    result = evolve_density_matrix(
        torch.zeros((2, 2), dtype=torch.complex128),
        EXCITED,
        1,
        times,
        [(torch.tensor([[0.0, 1.0], [0.0, 0.0]]), gamma)],
    )

    expected = torch.exp(-gamma * times)
    torch.testing.assert_close(result.populations[:, 1], expected, atol=1e-8, rtol=1e-8)


def test_zero_rate_matches_undamped_rabi_oscillation() -> None:
    times = torch.linspace(0.0, 8.0, 161, dtype=torch.float64)
    frequency = math.sqrt(1.04)
    result = evolve_density_matrix(
        (X + 0.2 * Z) / 2,
        EXCITED,
        1,
        times,
        [{"operator": "amplitude_damping", "rate": 0.0}],
    )

    expected = 1.0 - torch.sin(frequency * times / 2) ** 2 / 1.04
    torch.testing.assert_close(result.populations[:, 1], expected, atol=1e-7, rtol=1e-7)


def test_physical_scaling_leaves_populations_invariant() -> None:
    times = torch.linspace(0.0, 8.0, 161, dtype=torch.float64)
    base = evolve_density_matrix(
        (X + 0.2 * Z) / 2,
        EXCITED,
        1,
        times,
        [{"operator": "amplitude_damping", "rate": 0.1}],
    )
    scale = 3.0
    scaled = evolve_density_matrix(
        scale * (X + 0.2 * Z) / 2,
        EXCITED,
        1,
        times / scale,
        [{"operator": "amplitude_damping", "rate": scale * 0.1}],
    )

    torch.testing.assert_close(scaled.populations, base.populations)


def test_json_native_request_and_plan_use_the_public_contract() -> None:
    request = {
        "hamiltonian": [
            {"pauli": "X", "coefficient": 0.5, "wires": [0]},
            {"pauli": "Z", "coefficient": 0.1, "wires": [0]},
        ],
        "initial_state": "1",
        "n_wires": 1,
        "times": [0.0, 0.05, 0.1],
        "collapse_operators": [
            {"operator": "amplitude_damping", "rate": 0.1, "wire": 0}
        ],
        "observables": [{"name": "z", "pauli": "Z", "wires": [0]}],
        "return_density_matrices": True,
    }

    plan = plan_density_matrix_evolution(**request)
    result = evolve_density_matrix(**request)

    assert plan.n_times == 3
    assert plan.trajectory_bytes == 3 * 2 * 2 * 16
    assert result.observables["z"].shape == (3,)
    assert result.to_dict()["density_matrices"] is not None


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        (
            {"collapse_operators": [{"operator": "amplitude_damping", "rate": -0.1}]},
            "negative_rate",
        ),
        ({"times": [0.0, 0.5, 0.5]}, "non_monotone_time_grid"),
        ({"times": [0.0, 1.0, 0.5]}, "non_monotone_time_grid"),
        (
            {"collapse_operators": [{"operator": "not-a-channel", "rate": 0.1}]},
            "unknown_collapse_operator",
        ),
        (
            {"hamiltonian": torch.tensor([[0.0, 1.0], [0.0, 0.0]])},
            "non_hermitian_hamiltonian",
        ),
    ],
)
def test_invalid_inputs_have_structured_errors(
    overrides: dict[str, object], code: str
) -> None:
    inputs: dict[str, object] = {
        "hamiltonian": X,
        "initial_state": EXCITED,
        "n_wires": 1,
        "times": [0.0, 0.5, 1.0],
    }
    inputs.update(overrides)

    with pytest.raises(EvolutionValidationError) as caught:
        evolve_density_matrix(**inputs)

    assert caught.value.to_dict()["code"] == code
