"""Static finite-product dynamics through the existing public Lindblad path."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.errors import CapabilityError, SerializationError
from flagquantum.observables.boson import annihilate
from flagquantum.simulation.lindblad import (
    EvolutionResult,
    EvolutionValidationError,
    evolve_density_matrix,
    plan_density_matrix_evolution,
)

pytestmark = pytest.mark.unit
TIMES = torch.linspace(0.0, 0.5, 51, dtype=torch.float64)


def _transition(coupling=0.7):
    operator = torch.zeros((3, 3), dtype=torch.complex128)
    operator[1, 2] = operator[2, 1] = 1
    return coupling * operator


def _spin_mode(levels, coupling=0.7):
    lower = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex128)
    ladder = annihilate(0).to_matrix({0: levels})
    return coupling * (
        torch.kron(lower.T.contiguous(), ladder)
        + torch.kron(lower, ladder.T.contiguous())
    )


def _resign(payload):
    unsigned = {key: value for key, value in payload.items() if key != "identity"}
    text = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["identity"] = hashlib.sha256(text.encode()).hexdigest()
    return payload


def test_qubit_plan_content_and_identity_match_pre_m1_fixture():
    # Captured with unmodified M0 code at bbabf066, before adding dimensions.
    path = (
        Path(__file__).resolve().parents[1] / "fixtures/lindblad_plan_v1_3_qubit.json"
    )
    fixture = json.loads(path.read_text())
    hamiltonian = torch.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=torch.complex128)
    for keywords in ({}, {"dimensions": (2,)}, {"n_qubits": 1, "dimensions": (2,)}):
        plan = fql.plan(hamiltonian, "1", [0.0, 0.1], **keywords)
        assert plan.to_dict() == fixture
        assert plan.dimensions == (2,)
        assert plan.n_qubits == 1
    restored = fql.LindbladPlan.from_dict(fixture)
    assert restored.identity == fixture["identity"]
    assert torch.equal(fql.run(plan).populations, fql.run(restored).populations)


@pytest.mark.parametrize("method", fql.INTEGRATOR_NAMES)
@pytest.mark.integration
def test_qutrit_public_workflow_and_serialized_plan_match_analytic_rabi(method):
    plan = fql.plan(
        _transition(),
        [0.0, 1.0, 0.0],
        TIMES,
        dimensions=(3,),
        method=method,
        return_density_matrices=True,
    )
    assert plan.dimensions == (3,)
    assert plan.dimension == 3
    assert plan.trajectory_bytes == len(TIMES) * 3 * 3 * 16
    assert plan.to_dict()["version"] == "1.4"
    assert plan.to_dict()["request"]["n_qubits"] is None
    with pytest.raises(CapabilityError, match="not a qubit register"):
        _ = plan.n_qubits
    result = fql.run(plan)
    assert isinstance(result, EvolutionResult)
    assert result.plan is plan
    tolerance = 1e-6 if method == "crank-nicolson" else 1e-10
    assert torch.allclose(
        result.populations[:, 2], torch.sin(0.7 * TIMES) ** 2, atol=tolerance, rtol=0
    )
    assert result.populations.shape == (51, 3)
    assert result.density_matrices.shape == (51, 3, 3)
    assert result.maximum_trace_drift < 1e-12
    assert result.population_bounded
    restored = fql.LindbladPlan.from_json(plan.to_json())
    assert restored.dimensions == (3,)
    assert restored.identity == plan.identity
    assert torch.equal(fql.run(restored).populations, result.populations)
    with pytest.raises(TypeError, match="semantic overrides"):
        fql.run(plan, dimensions=(3,))


@pytest.mark.parametrize("threads", [1, 10])
def test_crank_nicolson_repeated_json_replay_is_bitwise_identical(
    threads: int,
) -> None:
    plan = fql.plan(
        _transition(),
        [0.0, 1.0, 0.0],
        TIMES,
        dimensions=(3,),
        method="crank-nicolson",
        return_density_matrices=True,
    )
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        reference = fql.run(plan)
        for _ in range(8):
            replay = fql.run(fql.LindbladPlan.from_json(plan.to_json()))
            assert torch.equal(replay.populations, reference.populations)
            assert torch.equal(replay.density_matrices, reference.density_matrices)
            assert replay.maximum_trace_drift < 1e-12
    finally:
        torch.set_num_threads(previous_threads)
    assert torch.allclose(
        reference.populations[:, 2],
        torch.sin(0.7 * TIMES) ** 2,
        atol=1e-6,
        rtol=0,
    )


@pytest.mark.parametrize("method", fql.INTEGRATOR_NAMES)
@pytest.mark.integration
def test_spin_mode_reuses_solver_with_dense_observables_and_basis_order(method):
    initial = torch.zeros(6, dtype=torch.complex128)
    initial[3] = 1  # |spin=1, occupation=0>, spin index most significant.
    number = torch.diag(torch.tensor([0.0, 1.0, 2.0], dtype=torch.complex128))
    result = evolve_density_matrix(
        _spin_mode(3),
        initial,
        None,
        TIMES,
        dimensions=(2, 3),
        method=method,
        observables={"motional": torch.kron(torch.eye(2), number)},
    )
    tolerance = 1e-6 if method == "crank-nicolson" else 1e-10
    expected = torch.sin(0.7 * TIMES) ** 2
    assert torch.allclose(result.populations[:, 1], expected, atol=tolerance, rtol=0)
    assert torch.allclose(
        result.expectation("motional"), expected, atol=tolerance, rtol=0
    )
    assert result.maximum_trace_drift < 1e-12


@pytest.mark.parametrize("method", fql.INTEGRATOR_NAMES)
def test_qutrit_decay_and_initial_density_adjoint_have_analytic_reference(method):
    collapse = torch.zeros((3, 3), dtype=torch.complex128)
    collapse[1, 2] = 1
    hamiltonian = torch.zeros_like(collapse)
    operators = [(collapse, 0.4)]
    result = fql.run(
        hamiltonian,
        [0.0, 0.0, 1.0],
        TIMES,
        dimensions=(3,),
        collapse_operators=operators,
        method=method,
    )
    tolerance = 1e-6 if method == "crank-nicolson" else 1e-10
    assert torch.allclose(
        result.populations[:, 2], torch.exp(-0.4 * TIMES), atol=tolerance, rtol=0
    )
    assert torch.allclose(
        result.populations[:, 1], 1 - torch.exp(-0.4 * TIMES), atol=tolerance, rtol=0
    )
    cotangent = fql.adjoint_gradient(
        hamiltonian,
        torch.eye(3, dtype=torch.complex128) / 3,
        TIMES,
        lambda trajectory: trajectory[-1, 2, 2].real,
        dimensions=(3,),
        collapse_operators=operators,
        method=method,
    )
    expected = torch.diag(
        torch.tensor([0.0, 0.0, math.exp(-0.2)], dtype=torch.complex128)
    )
    assert torch.allclose(cotangent, expected, atol=tolerance, rtol=0)
    assert not cotangent.requires_grad


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
def test_batched_live_plan_preserves_coupling_gradient_but_json_does_not(precision):
    coupling = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    plan = fql.plan(
        _transition(coupling),
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        TIMES,
        dimensions=(3,),
        return_density_matrices=True,
        options=fq.ExecutionOptions(
            precision=precision, batch_size=2, require_gradients=True
        ),
    )
    result = fql.run(plan)
    assert result.populations.shape == (51, 2, 3)
    assert result.density_matrices.shape == (51, 2, 3, 3)
    assert plan.trajectory_bytes == 51 * 2 * 9 * (8 if precision == "complex64" else 16)
    (gradient,) = torch.autograd.grad(result.populations[-1, :, 2].sum(), coupling)
    tolerance = 1e-6 if precision == "complex64" else 1e-9
    assert gradient.item() == pytest.approx(0.5 * math.sin(0.7), abs=tolerance)
    restored = fql.LindbladPlan.from_json(plan.to_json())
    with pytest.raises(EvolutionValidationError) as error:
        fql.run(restored)
    assert error.value.code == "gradients_not_available"


def test_coupling_gradient_matches_finite_difference_through_public_plan():
    coupling = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    result = fql.run(_transition(coupling), [0.0, 1.0, 0.0], TIMES, dimensions=(3,))
    (gradient,) = torch.autograd.grad(result.populations[-1, 2], coupling)
    step = 1e-4
    values = [
        fql.run(
            _transition(0.7 + offset * step), [0.0, 1.0, 0.0], TIMES, dimensions=(3,)
        ).populations[-1, 2]
        for offset in (-2, -1, 1, 2)
    ]
    finite_difference = (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (
        12 * step
    )
    assert gradient.item() == pytest.approx(finite_difference.item(), abs=1e-9)


@pytest.mark.parametrize(
    ("method", "minimum_ratio"), [("runge-kutta", 10), ("crank-nicolson", 3)]
)
def test_time_grid_refinement_is_separate_from_truncation(method, minimum_ratio):
    errors = []
    for count in (9, 17):
        result = fql.run(
            _transition(1.3),
            [0.0, 1.0, 0.0],
            torch.linspace(0.0, 1.0, count, dtype=torch.float64),
            dimensions=(3,),
            method=method,
        )
        errors.append(abs(result.populations[-1, 2].item() - math.sin(1.3) ** 2))
    assert errors[0] / errors[1] > minimum_ratio


def test_motional_truncation_converges_in_a_fixed_excitation_sector():
    ground_populations = []
    for levels in (2, 3, 5):
        initial = torch.zeros(2 * levels, dtype=torch.complex128)
        initial[levels + 1] = 1  # |spin=1, occupation=1> couples to |0, 2>.
        result = fql.run(_spin_mode(levels), initial, TIMES, dimensions=(2, levels))
        ground_populations.append(result.populations[-1, :levels].sum().item())
    expected = math.sin(math.sqrt(2) * 0.7 * 0.5) ** 2
    assert abs(ground_populations[0] - expected) > 0.1
    assert ground_populations[1] == pytest.approx(expected, abs=1e-9)
    assert ground_populations[2] == pytest.approx(expected, abs=1e-9)


def test_subsystem_order_is_part_of_plan_identity_even_at_equal_total_dimension():
    plans = [
        fql.plan(
            torch.zeros((6, 6)),
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0],
            dimensions=dimensions,
        )
        for dimensions in ((2, 3), (3, 2))
    ]
    assert plans[0].identity != plans[1].identity
    assert plans[0].dimension == plans[1].dimension == 6


@pytest.mark.parametrize("dimensions", [(), (0,), (-1, 3), (True, 3), (2.5,), {0: 3}])
def test_invalid_dimensions_fail_during_planning(dimensions):
    with pytest.raises(EvolutionValidationError) as error:
        fql.plan(_transition(), [0.0, 1.0, 0.0], TIMES, dimensions=dimensions)
    assert error.value.code == "invalid_dimensions"


def test_qutrit_dimensions_must_be_explicit_and_cannot_conflict_with_qubits():
    with pytest.raises(EvolutionValidationError) as error:
        fql.plan(_transition(), [0.0, 1.0, 0.0], TIMES)
    assert error.value.code == "cannot_infer_n_qubits"
    with pytest.raises(EvolutionValidationError) as error:
        fql.plan(torch.eye(4), [1.0, 0.0, 0.0, 0.0], TIMES, n_qubits=2, dimensions=(4,))
    assert error.value.code == "conflicting_state_space"


@pytest.mark.parametrize(
    "form", ["hamiltonian", "initial_state", "collapse", "outputs"]
)
def test_four_level_system_refuses_qubit_forms_before_any_integration(
    form, monkeypatch
):
    import flagquantum.simulation.lindblad as engine

    def forbidden(*args, **kwargs):
        pytest.fail("invalid request reached an integrator")

    monkeypatch.setattr(engine, "advance_interval", forbidden)
    hamiltonian = fq.X(0) if form == "hamiltonian" else torch.eye(4)
    initial = "00" if form == "initial_state" else [1.0, 0.0, 0.0, 0.0]
    kwargs = (
        {"collapse_operators": [fql.amplitude_damping(0.4, 0)]}
        if form == "collapse"
        else {}
    )
    if form == "outputs":
        kwargs["outputs"] = fq.expectation(fq.Z(0))
    with pytest.raises(EvolutionValidationError) as error:
        fql.run(hamiltonian, initial, TIMES, dimensions=(4,), **kwargs)
    assert error.value.code == "qubit_only_input"


@pytest.mark.parametrize("tamper", ["order", "dimension", "boolean", "version"])
def test_forged_mixed_plan_refuses_inconsistent_dimensions(tamper):
    plan = fql.plan(_transition(), [0.0, 1.0, 0.0], TIMES, dimensions=(3,))
    payload = plan.to_dict()
    if tamper == "order":
        payload["request"]["dimensions"] = [1, 3]
    elif tamper == "dimension":
        payload["decision"]["dimension"] = 4
    elif tamper == "boolean":
        payload["request"]["dimensions"] = [True, 3]
    else:
        payload["version"] = "1.3"
    with pytest.raises(SerializationError):
        fql.LindbladPlan.from_dict(_resign(payload))


@pytest.mark.parametrize("method", fql.INTEGRATOR_NAMES)
def test_vacuum_only_space_does_not_require_qubit_padding(method):
    result = fql.run([[0.2]], [1.0], [0.0, 0.1], dimensions=(1,), method=method)
    assert result.populations.shape == (2, 1)
    assert torch.equal(result.populations, torch.ones((2, 1), dtype=torch.float64))


def test_planner_reports_exact_mixed_dimension_trajectory_size():
    plan = plan_density_matrix_evolution(
        _spin_mode(3),
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        None,
        TIMES,
        dimensions=(2, 3),
    )
    assert plan.to_dict()["dimensions"] == [2, 3]
    assert plan.dimension == 6
    assert plan.trajectory_bytes == len(TIMES) * 6 * 6 * 16
