"""Scheduled controls share static dynamics, serialization, and differentiation."""

from __future__ import annotations

import copy
import hashlib
import json
import math

import pytest
import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.errors import SerializationError
from flagquantum.observables import boson
from flagquantum.simulation import lindblad as dynamics
from flagquantum.simulation.lindblad import EvolutionValidationError

pytestmark = pytest.mark.unit


def _template():
    transition = torch.zeros((3, 3), dtype=torch.complex128)
    transition[1, 2] = transition[2, 1] = 1
    return [{"operator": transition, "coefficient": 0.5 * fq.Parameter("drive")}]


def _plan(**overrides):
    arguments = dict(
        dimensions=(3,),
        parameter_values=[{"drive": 1.0}, {"drive": 3.0}],
        method="krylov-exponential",
        return_density_matrices=True,
    )
    arguments.update(overrides)
    return fql.plan(_template(), [0.0, 1.0, 0.0], [0.0, 0.1, 0.3], **arguments)


def _resign(payload):
    unsigned = {key: value for key, value in payload.items() if key != "identity"}
    payload["identity"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return payload


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
@pytest.mark.parametrize("method", fql.INTEGRATOR_NAMES)
def test_live_and_json_plans_share_interval_matrices_and_results(precision, method):
    plan = _plan(method=method, options=fq.ExecutionOptions(precision=precision))
    payload = plan.to_dict()
    assert payload["version"] == "1.5"
    assert "hamiltonian" not in payload["request"]
    assert len(payload["request"]["hamiltonians"]) == 2
    assert payload["decision"]["numerics"]["time_dependence"] == "piecewise_constant"
    assert payload["decision"]["numerics"]["n_intervals"] == 2
    assert payload["request"]["dimensions"] == [3]
    assert payload["request"]["n_qubits"] is None
    assert plan.trajectory_bytes == 3 * 3 * 3 * (8 if precision == "complex64" else 16)
    restored = fql.LindbladPlan.from_json(plan.to_json())
    assert restored.identity == plan.identity
    assert torch.equal(
        fql.run(plan).density_matrices, fql.run(restored).density_matrices
    )
    tolerance = 0.002 if method == "crank-nicolson" else 2e-5
    assert fql.run(plan).populations[-1, 2].item() == pytest.approx(
        math.sin(0.35) ** 2,
        abs=tolerance,
    )
    with pytest.raises(TypeError, match="semantic overrides"):
        fql.run(plan, parameter_values=[{"drive": 2.0}, {"drive": 3.0}])


def test_json_cannot_restore_a_coefficient_graph():
    value = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    plan = _plan(
        parameter_values=[{"drive": value}, {"drive": value}],
        options=fq.ExecutionOptions(require_gradients=True),
    )
    assert fql.run(plan).populations.requires_grad
    with pytest.raises(EvolutionValidationError) as caught:
        fql.run(fql.LindbladPlan.from_json(plan.to_json()))
    assert caught.value.code == "gradients_not_available"


def test_identity_depends_on_ordered_generators_instead_of_symbolic_spelling():
    plan = _plan()
    renamed = [
        {
            "operator": _template()[0]["operator"],
            "coefficient": 0.5 * fq.Parameter("renamed")
            + 0 * fq.Parameter("unused_effect"),
        }
    ]
    equivalent = fql.plan(
        renamed,
        [0.0, 1.0, 0.0],
        [0.0, 0.1, 0.3],
        dimensions=(3,),
        parameter_values=[
            {"renamed": 1.0, "unused_effect": 8.0},
            {"renamed": 3.0, "unused_effect": -4.0},
        ],
        method="krylov-exponential",
        return_density_matrices=True,
    )
    assert equivalent.identity == plan.identity
    assert (
        _plan(parameter_values=[{"drive": 3.0}, {"drive": 1.0}]).identity
        != plan.identity
    )
    constant = _plan(parameter_values=[{"drive": 1.0}, {"drive": 1.0}])
    static = fql.plan(
        0.5 * _template()[0]["operator"],
        [0.0, 1.0, 0.0],
        [0.0, 0.1, 0.3],
        dimensions=(3,),
        method="krylov-exponential",
        return_density_matrices=True,
    )
    assert constant.to_dict()["version"] == "1.5"
    assert static.to_dict()["version"] == "1.4"
    assert constant.identity != static.identity
    assert torch.equal(fql.run(constant).populations, fql.run(static).populations)


@pytest.mark.parametrize(
    "mutation",
    [
        "count",
        "metadata_count",
        "model",
        "unknown_request",
        "unknown_tensor",
        "unknown_root",
        "unknown_decision",
        "dimensions",
        "disguised_static",
        "scalar_hamiltonian",
        "non_hermitian",
        "shape",
    ],
)
def test_rehashed_inconsistent_or_unknown_scheduled_payloads_are_refused(mutation):
    payload = _plan().to_dict()
    request, decision = payload["request"], payload["decision"]
    if mutation == "count":
        request["hamiltonians"].pop()
    elif mutation == "metadata_count":
        decision["numerics"]["n_intervals"] = 3
    elif mutation == "model":
        decision["numerics"]["time_dependence"] = "linear"
    elif mutation == "unknown_request":
        request["interpolation"] = "linear"
    elif mutation == "unknown_tensor":
        request["hamiltonians"][0]["pulse"] = 1
    elif mutation == "unknown_root":
        payload["controls"] = []
    elif mutation == "unknown_decision":
        decision["numerics"]["extra"] = 1
    elif mutation == "dimensions":
        request["dimensions"] = [1, 3]
    elif mutation == "disguised_static":
        payload["version"] = "1.4"
    elif mutation == "scalar_hamiltonian":
        request["hamiltonian"] = request["hamiltonians"][0]
    elif mutation == "non_hermitian":
        request["hamiltonians"][1]["values"][1][2] = [0.0, 1.0]
    elif mutation == "shape":
        request["hamiltonians"][0]["shape"] = [2, 2]
    with pytest.raises((SerializationError, EvolutionValidationError)):
        fql.LindbladPlan.from_dict(_resign(payload))


def test_reordering_changes_identity_and_cannot_keep_the_original_seal():
    payload = _plan().to_dict()
    original = copy.deepcopy(payload)
    payload["request"]["hamiltonians"].reverse()
    with pytest.raises(SerializationError, match="identity"):
        fql.LindbladPlan.from_dict(payload)
    # A fully resealed, consistent different request is valid and has a new ID.
    reordered = fql.LindbladPlan.from_dict(_resign(payload))
    assert reordered.identity != original["identity"]
    assert not torch.equal(fql.run(reordered).populations, fql.run(_plan()).populations)


@pytest.mark.parametrize(
    ("values", "code"),
    [
        ([], "invalid_parameter_values"),
        (lambda: [], "invalid_parameter_values"),
        (iter([{}, {}]), "invalid_parameter_values"),
        ({"drive": 1.0}, "invalid_parameter_values"),
        ("drive", "invalid_parameter_values"),
        ([{"drive": 1.0}], "invalid_parameter_values"),
        ([None, {}], "invalid_parameter_binding"),
        ([{}, {}], "invalid_parameter_binding"),
        ([{"drive": 1.0, "extra": 2.0}] * 2, "invalid_parameter_binding"),
        ([{"drive": 1.0, fq.Parameter("drive"): 1.0}] * 2, "invalid_parameter_binding"),
        ([{1: 1.0}] * 2, "invalid_parameter_binding"),
        ([{"drive": True}] * 2, "invalid_coefficient"),
        ([{"drive": "1"}] * 2, "invalid_coefficient"),
        ([{"drive": 10**1000}] * 2, "invalid_coefficient"),
        ([{"drive": torch.tensor(True)}] * 2, "invalid_coefficient"),
        ([{"drive": float("nan")}] * 2, "invalid_coefficient"),
        ([{"drive": float("inf")}] * 2, "invalid_coefficient"),
        ([{"drive": torch.ones(2)}] * 2, "invalid_coefficient"),
        ([{"drive": fq.Parameter("unbound")}] * 2, "invalid_coefficient"),
    ],
)
def test_invalid_controls_fail_before_any_integration(values, code, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("integration started before all controls were validated")

    monkeypatch.setattr(dynamics, "advance_interval", forbidden)
    with pytest.raises(EvolutionValidationError) as caught:
        dynamics.evolve_density_matrix(
            _template(),
            [0.0, 1.0, 0.0],
            None,
            [0.0, 0.1, 0.3],
            dimensions=(3,),
            parameter_values=values,
        )
    assert caught.value.code == code


@pytest.mark.parametrize(
    "times",
    [
        torch.tensor([0.0, 0.1, 0.3], requires_grad=True),
        [0.0, torch.tensor(0.1, requires_grad=True), 0.3],
    ],
)
def test_time_gradients_are_refused_before_conversion(times):
    with pytest.raises(EvolutionValidationError) as caught:
        fql.plan(
            _template(),
            [0.0, 1.0, 0.0],
            times,
            dimensions=(3,),
            parameter_values=[{"drive": 1.0}] * 2,
        )
    assert caught.value.code == "unsupported_time_gradient"


def test_a_later_nonhermitian_interval_is_validated_before_the_first_step(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("integration started")

    monkeypatch.setattr(dynamics, "advance_interval", forbidden)
    operator = torch.zeros((3, 3), dtype=torch.complex128)
    operator[1, 2] = 1
    with pytest.raises(EvolutionValidationError) as caught:
        dynamics.evolve_density_matrix(
            [{"operator": operator, "coefficient": fq.Parameter("drive")}],
            [0.0, 1.0, 0.0],
            None,
            [0.0, 0.1, 0.3],
            dimensions=(3,),
            parameter_values=[{"drive": 0.0}, {"drive": 1.0}],
        )
    assert caught.value.code == "non_hermitian_hamiltonian"
    assert "parameter_values[1]" in caught.value.field


@pytest.mark.parametrize(
    "template",
    [
        lambda time: time,
        {"X": fq.Parameter("drive")},
        [{"pauli": "X", "coefficient": fq.Parameter("drive")}],
        [{"operator": torch.eye(2), "coefficient": fq.Parameter("drive"), "qubit": 0}],
    ],
)
def test_unsupported_templates_do_not_become_dense_pauli_controls(template):
    with pytest.raises(EvolutionValidationError) as caught:
        fql.plan(template, "0", [0.0, 0.1], parameter_values=[{"drive": 1.0}])
    assert caught.value.code == "unsupported_time_dependence"


def test_no_symbolic_names_or_no_interval_is_refused():
    with pytest.raises(EvolutionValidationError) as caught:
        fql.plan(
            [{"operator": torch.eye(2), "coefficient": 1.0}],
            "0",
            [0.0, 0.1],
            parameter_values=[{}],
        )
    assert caught.value.code == "invalid_parameter_values"
    with pytest.raises(EvolutionValidationError) as caught:
        fql.plan(
            _template(), [0.0, 1.0, 0.0], [0.0], dimensions=(3,), parameter_values=[]
        )
    assert caught.value.code == "invalid_parameter_values"


@pytest.mark.parametrize("method", fql.INTEGRATOR_NAMES)
@pytest.mark.parametrize("batched", [False, True])
def test_noncommuting_dissipative_initial_density_adjoint_matches_autograd(
    method, batched
):
    first, second = fq.Parameter("first"), fq.Parameter("second")
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    z = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.complex128))
    h = [{"operator": x, "coefficient": first}, {"operator": z, "coefficient": second}]
    initial = torch.tensor([[0.6, 0.1j], [-0.1j, 0.4]], dtype=torch.complex128)
    if batched:
        initial = torch.stack((initial, initial.conj()))
    initial.requires_grad_()
    control = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    kwargs = dict(
        parameter_values=[
            {"first": control, "second": 0.1},
            {"first": -0.4, "second": 1.2},
        ],
        collapse_operators=[fql.amplitude_damping(0.2, 0)],
        method=method,
    )

    def cost(trajectory):
        return (
            trajectory[0, ..., 1, 1].real.sum() * 0.2
            + trajectory[1:, ..., 0, 0].real.sum()
        )

    eye = torch.eye(2, dtype=torch.complex128)
    identity = torch.eye(4, dtype=torch.complex128)
    lower = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex128)
    a = lower.mH @ lower
    dissipator = 0.2 * (
        torch.kron(lower, lower.conj())
        - 0.5 * (torch.kron(a, eye) + torch.kron(eye, a.T.contiguous()))
    )
    rows = [initial]
    for matrix, dt in [
        (control.detach() * x + 0.1 * z, 0.1),
        (-0.4 * x + 1.2 * z, 0.2),
    ]:
        generator = (
            -1j * (torch.kron(matrix, eye) - torch.kron(eye, matrix.T.contiguous()))
            + dissipator
        )
        if method == "runge-kutta":
            scaled = dt / 2 * generator
            step = (
                identity
                + scaled
                + scaled @ scaled / 2
                + torch.linalg.matrix_power(scaled, 3) / 6
                + torch.linalg.matrix_power(scaled, 4) / 24
            )
            propagator = step @ step
        elif method == "crank-nicolson":
            step = torch.linalg.solve(
                identity - dt / 4 * generator, identity + dt / 4 * generator
            )
            propagator = step @ step
        else:
            propagator = torch.matrix_exp(dt * generator)
        rows.append(
            (rows[-1].reshape(*initial.shape[:-2], 4) @ propagator.T).reshape_as(
                initial
            )
        )
    (expected,) = torch.autograd.grad(cost(torch.stack(rows)), initial)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda value: saved.append(value) or value, lambda value: value
    ):
        adjoint = fql.adjoint_gradient(h, initial, [0.0, 0.1, 0.3], cost, **kwargs)
    torch.testing.assert_close(adjoint, expected, atol=2e-10, rtol=0)
    assert saved == []
    assert not adjoint.requires_grad
    with pytest.raises(EvolutionValidationError) as caught:
        fql.adjoint_gradient(
            h,
            initial,
            [0.0, 0.1, 0.3],
            cost,
            **kwargs,
            options=fq.ExecutionOptions(require_gradients=True),
        )
    assert caught.value.code == "adjoint_requires_no_graph"


@pytest.mark.integration
def test_boson_exchange_optimizer_updates_interval_controls():
    exchange = boson.create(0) * boson.annihilate(1) + boson.create(
        1
    ) * boson.annihilate(0)
    h = [{"operator": exchange, "coefficient": fq.Parameter("g")}]
    controls = torch.tensor([0.2, 0.4], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([controls], lr=0.03)
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        plan = fql.plan(
            h,
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.2, 0.5],
            dimensions=(2, 3),
            parameter_values=[{"g": value} for value in controls.unbind()],
            method="krylov-exponential",
            options=fq.ExecutionOptions(require_gradients=True),
        )
        population = fql.run(plan).populations[-1, 1]
        expected = torch.sin(0.2 * controls[0] + 0.3 * controls[1]) ** 2
        torch.testing.assert_close(population, expected, atol=1e-10, rtol=0)
        loss = (population - 0.15) ** 2
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0] / 30


@pytest.mark.parametrize("method", fql.INTEGRATOR_NAMES)
def test_refinement_preserves_controls_and_checks_global_error_separately(method):
    errors = []
    for pieces in (1, 2, 4):
        times, values = [0.0], []
        for start, stop, value in [(0.0, 0.1, 1.0), (0.1, 0.3, 3.0), (0.3, 0.35, 7.0)]:
            times.extend(
                start + (stop - start) * k / pieces for k in range(1, pieces + 1)
            )
            values.extend([{"drive": value}] * pieces)
        result = fql.run(
            _template(),
            [0.0, 1.0, 0.0],
            times,
            dimensions=(3,),
            parameter_values=values,
            method=method,
            solve_tolerance=None if method == "runge-kutta" else 1e-14,
        )
        errors.append(abs(result.populations[-1, 2].item() - math.sin(0.525) ** 2))
    if method == "runge-kutta":
        assert errors[1] < errors[0] / 10
        assert errors[2] < errors[1] / 10
    elif method == "crank-nicolson":
        assert errors[1] < errors[0] / 3
        assert errors[2] < errors[1] / 3
    else:
        # The pre-implementation experiment observed a 1.37e-11 numerical floor
        # despite a 1e-14 local defect target. This independent global budget is
        # deliberately distinct from that target; existing solver tests stay.
        assert max(errors) < 1e-9


def test_finite_difference_and_complex64_have_independent_error_budgets():
    values = torch.tensor([1.0, 3.0, 7.0], dtype=torch.float64, requires_grad=True)

    def probability(controls, precision="complex128"):
        return fql.run(
            _template(),
            [0.0, 1.0, 0.0],
            [0.0, 0.1, 0.3, 0.35],
            dimensions=(3,),
            parameter_values=[{"drive": value} for value in controls.unbind()],
            method="krylov-exponential",
            options=fq.ExecutionOptions(precision=precision),
        ).populations[-1, 2]

    (gradient,) = torch.autograd.grad(probability(values), values)
    for index in range(3):
        perturbation = torch.zeros_like(values)
        perturbation[index] = 1e-5
        finite_difference = (
            probability(values.detach() + perturbation)
            - probability(values.detach() - perturbation)
        ) / 2e-5
        torch.testing.assert_close(
            gradient[index], finite_difference, atol=2e-9, rtol=0
        )
    assert probability(values.detach(), "complex64").item() == pytest.approx(
        math.sin(0.525) ** 2, abs=2e-6
    )
    single_precision = values.detach().requires_grad_()
    (single_gradient,) = torch.autograd.grad(
        probability(single_precision, "complex64"), single_precision
    )
    analytic = 0.5 * values.new_tensor([0.1, 0.2, 0.05]) * math.sin(1.05)
    torch.testing.assert_close(single_gradient, analytic, atol=2e-6, rtol=0)


def test_unreachable_scheduled_residual_target_fails_closed(monkeypatch):
    from flagquantum.simulation import lindblad_integrators as integrators

    monkeypatch.setattr(integrators, "SOLVE_RESTART", 1)
    monkeypatch.setattr(dynamics, "_solve_max_iterations", lambda dimension: 1)
    with pytest.raises(EvolutionValidationError) as caught:
        fql.run(
            [{"operator": torch.zeros((2, 2)), "coefficient": fq.Parameter("g")}],
            "1",
            [0.0, 0.2, 0.5],
            parameter_values=[{"g": 0.0}, {"g": 1.0}],
            collapse_operators=[fql.amplitude_damping(0.4, 0)],
            method="crank-nicolson",
            solve_tolerance=1e-15,
        )
    assert caught.value.code == "integration_not_converged"


def test_scheduled_qubit_canonicalization_and_low_level_entry_points():
    operator = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    template = [{"operator": operator, "coefficient": fq.Parameter("g")}]
    values = [{"g": 0.2}, {"g": 0.4}]
    plan = fql.plan(template, "0", [0.0, 0.1, 0.3], parameter_values=values)
    explicit = fql.plan(
        template, "0", [0.0, 0.1, 0.3], dimensions=(2,), parameter_values=values
    )
    assert plan.identity == explicit.identity
    assert plan.to_dict()["request"]["n_qubits"] == 1
    assert "dimensions" not in plan.to_dict()["request"]
    summary = dynamics.plan_density_matrix_evolution(
        template, "0", 1, [0.0, 0.1, 0.3], parameter_values=values
    )
    assert summary.time_dependence == "piecewise_constant" and summary.n_intervals == 2
    expected = dynamics.evolve_density_matrix(
        template, "0", 1, [0.0, 0.1, 0.3], parameter_values=values
    )
    torch.testing.assert_close(
        fql.run(fql.LindbladPlan.from_json(plan.to_json())).populations,
        expected.populations,
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize(
    "key",
    [
        "hamiltonians",
        "parameter_values",
        "time_dependence",
        "n_intervals",
        "bound_hamiltonians",
    ],
)
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("location", ["request", "root"])
def test_scheduled_data_cannot_hide_in_a_valid_static_request(key, dimension, location):
    plan = fql.plan(
        torch.eye(dimension),
        [1.0] + [0.0] * (dimension - 1),
        [0.0, 0.1],
        dimensions=(dimension,),
    )
    payload = plan.to_dict()
    target = payload["request"] if location == "request" else payload
    target[key] = []
    with pytest.raises(SerializationError, match="version 1.5"):
        fql.LindbladPlan.from_dict(_resign(payload))


@pytest.mark.parametrize(
    "key", ["n_wires", "n_times", "dimension", "trajectory_bytes", "batch_size"]
)
def test_rehashed_decision_counts_are_typed(key):
    payload = _plan().to_dict()
    payload["decision"][key] = float(payload["decision"][key])
    with pytest.raises(SerializationError, match="integer"):
        fql.LindbladPlan.from_dict(_resign(payload))


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
def test_complex_conjugate_boson_controls_preserve_the_same_torch_gradient(precision):
    dtype = getattr(torch, precision)
    tolerance = 2e-6 if precision == "complex64" else 1e-10
    alpha = torch.tensor(0.3 + 0.2j, dtype=dtype, requires_grad=True)
    raising = boson.create(0)
    terms = [
        {"operator": raising, "coefficient": fq.Parameter("a")},
        {"operator": raising.dagger(), "coefficient": fq.Parameter("conjugate")},
    ]
    result = fql.run(
        terms,
        [1.0, 0.0, 0.0],
        [0.0, 0.1, 0.3],
        dimensions=(3,),
        parameter_values=[
            {"a": alpha, "conjugate": alpha.conj()},
            {"a": -2 * alpha, "conjugate": -2 * alpha.conj()},
        ],
        method="krylov-exponential",
        options=fq.ExecutionOptions(precision=precision),
    )
    (actual_gradient,) = torch.autograd.grad(result.populations[-1, 1], alpha)
    matrix = raising.to_matrix({0: 3}).to(dtype)
    hamiltonian = alpha * matrix + alpha.conj() * matrix.mH
    state = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    for scale, dt in [(1.0, 0.1), (-2.0, 0.2)]:
        state = torch.matrix_exp(-1j * dt * scale * hamiltonian) @ state
    expected = state[1].abs() ** 2
    (expected_gradient,) = torch.autograd.grad(expected, alpha)
    torch.testing.assert_close(
        result.populations[-1, 1], expected, atol=tolerance, rtol=0
    )
    torch.testing.assert_close(
        actual_gradient, expected_gradient, atol=10 * tolerance, rtol=0
    )


@pytest.mark.parametrize(
    ("precision", "method", "expected_identity"),
    [
        (
            "complex64",
            "runge-kutta",
            "ff561f72a4828666047f04af875f06b32440ba9cd2ca769c733360d1f59bc172",
        ),
        (
            "complex64",
            "crank-nicolson",
            "466bfcc8496ff34d01d66e1fbbdff197b76ae3307490efc736107b2a9cb13101",
        ),
        (
            "complex64",
            "krylov-exponential",
            "ccb00939b8bbc28e827e6fdc37ce4e256025d012e826dbdacbcbe88930ad6bc8",
        ),
        (
            "complex128",
            "runge-kutta",
            "373cdfb0d8d5bcdd7994145e6aad17ef4a80507941fa54247cd7c4b06feca47d",
        ),
        (
            "complex128",
            "crank-nicolson",
            "cf5c998f281de2851949b2d9b5a5a7a1d8004b05c115e10e57adfe4927a29f8b",
        ),
        (
            "complex128",
            "krylov-exponential",
            "4084dd27b83ddf4456170d1f7aca48f0ea867b3d699c413fb2b5b30be60a0761",
        ),
    ],
)
def test_static_v1_4_identities_match_pre_implementation_capture(
    precision, method, expected_identity
):
    # Captured at ca4b99ac109d9bbec6010324c7fd302235d262ed, before runtime edits.
    plan = fql.plan(
        0.7 * _template()[0]["operator"],
        [0.0, 1.0, 0.0],
        [0.0, 0.1, 0.3],
        dimensions=(3,),
        method=method,
        return_density_matrices=True,
        options=fq.ExecutionOptions(precision=precision),
    )
    assert plan.identity == expected_identity
    assert plan.to_dict()["version"] == "1.4"
    assert "time_dependence" not in plan.to_dict()["decision"]["numerics"]
    assert "n_intervals" not in plan.to_dict()["decision"]["numerics"]
