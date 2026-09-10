"""Focused tests for staged classical and quantum-aware optimization."""

import math
from dataclasses import replace

import pytest
import torch

import flagquantum as fq
import flagquantum.algorithms as fqa
from flagquantum.algorithms import Hamiltonian, pauli_term, vqe_loss
from flagquantum.algorithms.optimization import OptimizationStage, optimize_hybrid
from flagquantum.core.runtime_config import runtime_config


@pytest.mark.unit
@pytest.mark.parametrize("method", ["rotosolve", "lbfgs"])
@pytest.mark.parametrize("transposed", [False, True])
def test_optimizer_handles_parameter_layout_without_mutating_input(
    method: str, transposed: bool
) -> None:
    initial = torch.linspace(0.1, 0.6, 6, dtype=torch.float64).reshape(2, 3)
    if transposed:
        initial = initial.T
        assert not initial.is_contiguous()
    original = initial.clone()
    result = optimize_hybrid(
        lambda groups: (
            groups["quantum"].cos().sum()
            if method == "rotosolve"
            else (groups["quantum"] - 2).square().sum()
        ),
        {"quantum": initial},
        stages=(OptimizationStage("quantum", method, steps=1, lr=1.0),),
    )
    assert result.parameters["quantum"].shape == initial.shape
    assert result.parameters["quantum"].dtype == initial.dtype
    torch.testing.assert_close(initial, original)
    if method == "rotosolve":
        torch.testing.assert_close(
            result.parameters["quantum"].cos(), -torch.ones_like(initial)
        )
        assert result.history[-1] == pytest.approx(-6.0)
    else:
        torch.testing.assert_close(
            result.parameters["quantum"], torch.full_like(initial, 2.0)
        )
        assert result.history[-1] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("field", ["lr", "damping"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_optimization_stage_rejects_nonfinite_configuration(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError, match="learning rate and damping must be finite"):
        if field == "lr":
            OptimizationStage("quantum", "qng", steps=1, lr=value)
        else:
            OptimizationStage("quantum", "qng", steps=1, damping=value)


@pytest.mark.parametrize("field", ["steps", "max_iter", "history_size", "block_size"])
@pytest.mark.parametrize("value", [True, 1.5, float("nan"), float("inf")])
def test_optimization_stage_rejects_noninteger_counts(
    field: str, value: object
) -> None:
    stage = OptimizationStage("quantum", "qng", steps=1)
    with pytest.raises(ValueError, match="must be integers"):
        replace(stage, **{field: value})


def _one_qubit_problem():
    hamiltonian = Hamiltonian([pauli_term(1.0, "Z", (0,))])

    def builder(parameters):
        return fq.Circuit(1).rx(0, theta=parameters[0])

    return builder, hamiltonian


def test_heisenberg_dimer_singlet_has_exact_ground_energy():
    count = fqa.heisenberg_hva_parameter_count(2, 1)
    circuit = fqa.heisenberg_hva(2, 1, torch.zeros(count))
    energy = fqa.heisenberg_chain_hamiltonian(2).expectation(circuit).sum()

    torch.testing.assert_close(energy, torch.tensor(-3.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        fqa.heisenberg_chain_hamiltonian(2).ground_energy(),
        torch.tensor(-3.0, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )


def test_staged_adam_lbfgs_vqe_reaches_ground_state():
    builder, hamiltonian = _one_qubit_problem()
    result = fqa.run_hybrid_vqe(
        builder,
        torch.tensor([0.2]),
        hamiltonian,
        stages=(
            OptimizationStage("quantum", "adam", steps=12, lr=0.25),
            OptimizationStage("quantum", "lbfgs", steps=1, lr=0.8, max_iter=20),
        ),
    )

    assert result.history[-1] < -0.999
    assert [record.method for record in result.records][-1] == "lbfgs"
    assert result.evaluations >= len(result.records)


def test_rotosolve_uses_pauli_rotation_structure():
    builder, hamiltonian = _one_qubit_problem()
    result = fqa.run_hybrid_vqe(
        builder,
        torch.tensor([0.17]),
        hamiltonian,
        stages=(OptimizationStage("quantum", "rotosolve", steps=1),),
    )

    assert result.history[-1] < -0.999999
    assert torch.allclose(
        torch.remainder(result.parameters["quantum"], 2 * math.pi),
        torch.tensor([math.pi]),
        atol=1e-5,
    )


@pytest.mark.parametrize("scale", [0.0, float("nan"), float("inf")])
def test_qng_rejects_zero_or_nonfinite_state(scale: float) -> None:
    initial = torch.tensor([1.0])
    with pytest.raises(ValueError, match="finite nonzero state"):
        optimize_hybrid(
            lambda groups: groups["quantum"].square().sum(),
            {"quantum": initial},
            stages=(OptimizationStage("quantum", "qng", steps=1),),
            state_function=lambda groups: groups["quantum"].to(torch.complex64) * scale,
        )
    torch.testing.assert_close(initial, torch.tensor([1.0]))


def test_qng_uses_state_geometry_and_reduces_energy():
    builder, hamiltonian = _one_qubit_problem()
    initial = torch.tensor([1.0])
    result = fqa.run_hybrid_vqe(
        builder,
        initial,
        hamiltonian,
        stages=(OptimizationStage("quantum", "qng", steps=4, lr=0.2, damping=1e-3),),
    )

    assert result.history[-1] < float(vqe_loss(builder, initial, hamiltonian))
    assert all(record.gradient_norm is not None for record in result.records)
    assert all(record.wall_time_seconds > 0 for record in result.records)
    assert all(record.objective_gradient_seconds > 0 for record in result.records)
    assert all(record.quantum_metric_seconds is not None for record in result.records)
    assert all(record.linear_solve_seconds is not None for record in result.records)


def test_named_groups_can_mix_classical_and_quantum_updates():
    def objective(groups):
        return (groups["classical"] - 2.0).square().sum() + torch.cos(
            groups["quantum"]
        ).sum()

    result = optimize_hybrid(
        objective,
        {"classical": torch.tensor([0.0]), "quantum": torch.tensor([0.1])},
        stages=(
            OptimizationStage("classical", "adam", steps=60, lr=0.12),
            OptimizationStage("quantum", "rotosolve", steps=1),
        ),
    )

    assert abs(float(result.parameters["classical"][0]) - 2.0) < 0.2
    assert torch.cos(result.parameters["quantum"]).item() < -0.999999


def test_qng_requires_explicit_state_function_for_generic_hybrid_objective():
    try:
        optimize_hybrid(
            lambda groups: groups["quantum"].square().sum(),
            {"quantum": torch.tensor([1.0])},
            stages=(OptimizationStage("quantum", "qng", steps=1),),
        )
    except ValueError as error:
        assert "state_function" in str(error)
    else:
        raise AssertionError("QNG must fail closed without quantum state geometry")


def test_layerwise_vqe_preserves_shallow_solution_in_identity_layer():
    builder, hamiltonian = _one_qubit_problem()

    result = fqa.run_layerwise_vqe(
        lambda depth, parameters: builder(parameters[:depth]),
        lambda depth: depth,
        torch.tensor([0.2]),
        hamiltonian,
        depths=(1, 2),
        stages=(OptimizationStage("quantum", "lbfgs", 1, lr=0.8, max_iter=20),),
    )

    assert result.depths == (1, 2)
    assert result.parameters.shape == (2,)
    assert result.stages[0].history[-1] < -0.999
    assert result.stages[1].history[-1] < -0.999


def test_hybrid_vqe_preserves_requested_parameter_precision():
    builder, hamiltonian = _one_qubit_problem()
    result = fqa.run_hybrid_vqe(
        builder,
        torch.tensor([0.2], dtype=torch.float64),
        hamiltonian,
        stages=(OptimizationStage("quantum", "adam", 1, lr=0.1),),
    )

    assert result.parameters["quantum"].dtype == torch.float64


@pytest.mark.integration
def test_eight_qubit_heisenberg_hybrid_recipe_meets_convergence_gate():
    with runtime_config(complex_dtype="complex128"):
        n_wires, depth = 8, 5
        count = fqa.heisenberg_hva_parameter_count(n_wires, depth)
        generator = torch.Generator().manual_seed(260720)
        initial = 0.02 * torch.randn(count, generator=generator, dtype=torch.float64)
        hamiltonian = fqa.heisenberg_chain_hamiltonian(n_wires)

        result = fqa.run_hybrid_vqe(
            lambda parameters: fqa.heisenberg_hva(
                n_wires,
                depth,
                parameters,
                parameterization="bond_resolved_phase",
                initial_state="dimer_singlet",
            ),
            initial,
            hamiltonian,
            stages=(
                OptimizationStage("quantum", "adam", 100, lr=0.02),
                OptimizationStage(
                    "quantum",
                    "lbfgs",
                    1,
                    lr=0.8,
                    max_iter=150,
                    history_size=10,
                ),
            ),
        )
        exact = float(hamiltonian.ground_energy())
        relative_error = (result.history[-1] - exact) / abs(exact)

        assert relative_error <= 1e-5
