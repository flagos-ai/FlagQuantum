"""Ordered non-product coordinates through the existing Lindblad execution path."""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.errors import CapabilityError, SerializationError
from flagquantum.lindblad._plan import _complex_tensor_payload, _identity
from flagquantum.observables.boson import annihilate, create
from flagquantum.simulation import lindblad as engine
from flagquantum.simulation.lindblad import EvolutionValidationError
from flagquantum.simulation.lindblad_adjoint import adjoint_gradient as initial_adjoint

pytestmark = pytest.mark.unit
LABELS = ((0, 1), (1, 0))
X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)


def _plan(*, scheduled: bool = False, gradients: bool = False) -> fql.LindbladPlan:
    if scheduled:
        h: object = [{"operator": X, "coefficient": fq.Parameter("g")}]
        values = [
            {"g": torch.tensor(value, requires_grad=gradients)} for value in (0.2, 0.7)
        ]
    else:
        h = torch.tensor(0.3, requires_grad=gradients) * X
        values = None
    return fql.plan(
        h,
        [1.0, 0.0],
        [0.0, 0.1, 0.3],
        basis_labels=LABELS,
        method="krylov-exponential",
        parameter_values=values,
        return_density_matrices=True,
        options=fq.ExecutionOptions(require_gradients=gradients),
    )


def _resign(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload, identity=_identity(payload))


def _dense_two_state_trajectory(
    hamiltonians: Sequence[torch.Tensor],
    initial: torch.Tensor,
    times: Sequence[float],
    operator: torch.Tensor,
    rate: float,
    method: str,
) -> torch.Tensor:
    dtype = initial.dtype
    states = [initial]
    state = initial.flatten()
    eye = torch.eye(2, dtype=dtype)
    identity = torch.eye(4, dtype=dtype)
    product = operator.mH @ operator
    dissipator = rate * (
        torch.kron(operator, operator.conj())
        - 0.5 * (torch.kron(product, eye) + torch.kron(eye, product.T.contiguous()))
    )
    for interval_h, start, stop in zip(
        hamiltonians, times[:-1], times[1:], strict=True
    ):
        generator = (
            -1j
            * (torch.kron(interval_h, eye) - torch.kron(eye, interval_h.T.contiguous()))
            + dissipator
        )
        step = (stop - start) / 2
        if method == "krylov-exponential":
            state = torch.matrix_exp(2 * step * generator) @ state
        else:
            for _ in range(2):
                if method == "crank-nicolson":
                    state = torch.linalg.solve(
                        identity - step * generator / 2,
                        (identity + step * generator / 2) @ state,
                    )
                else:
                    k1 = generator @ state
                    k2 = generator @ (state + step * k1 / 2)
                    k3 = generator @ (state + step * k2 / 2)
                    k4 = generator @ (state + step * k3)
                    state = state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        states.append(state.reshape(2, 2))
    return torch.stack(states)


@pytest.mark.parametrize(
    "entry",
    [
        fql.plan,
        fql.run,
        fql.adjoint_gradient,
        engine.plan_density_matrix_evolution,
        engine.evolve_density_matrix,
        initial_adjoint,
    ],
)
def test_only_the_approved_optional_keyword_is_appended(
    entry: Callable[..., object],
) -> None:
    parameters = list(inspect.signature(entry).parameters.values())
    assert parameters[-1].name == "basis_labels"
    assert parameters[-2].name == "parameter_values"
    assert parameters[-1].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters[-1].default is None


@pytest.mark.parametrize("dimension", [1, 2, 3, 4, 5])
def test_numeric_dimension_never_infers_qubits_or_tensor_factors(
    dimension: int,
) -> None:
    labels = tuple((index,) for index in range(dimension))
    initial = [1.0] + [0.0] * (dimension - 1)
    plan = fql.plan(
        torch.zeros((dimension, dimension)), initial, [0.0, 0.1], basis_labels=labels
    )
    result = fql.run(plan)
    assert plan.dimension == dimension
    assert plan.basis_labels == labels
    assert result.populations.shape == (2, dimension)
    payload = plan.to_dict()
    assert payload["version"] == "1.6"
    assert payload["request"]["n_qubits"] is None
    assert payload["decision"]["n_wires"] is None
    assert "dimensions" not in payload["request"]
    assert "dimensions" not in payload["decision"]
    for attribute in ("n_qubits", "dimensions"):
        with pytest.raises(CapabilityError, match="basis_labels"):
            getattr(plan, attribute)
    summary = engine.plan_density_matrix_evolution(
        torch.zeros((dimension, dimension)),
        initial,
        None,
        [0.0, 0.1],
        basis_labels=labels,
    )
    assert summary.n_wires is None and summary.dimensions is None
    assert summary.basis_labels == labels
    with pytest.raises(CapabilityError, match="basis_labels"):
        _ = summary.n_qubits


@pytest.mark.parametrize(
    "labels",
    [
        [],
        [()],
        [(0,), (0,)],
        [(0,), (1, 0)],
        [(True,)],
        [(1.0,)],
        [("",)],
        [torch.tensor(1)],
        "01",
        {"basis": LABELS},
    ],
)
def test_invalid_labels_fail_before_integration(
    monkeypatch: pytest.MonkeyPatch, labels: object
) -> None:
    monkeypatch.setattr(
        engine,
        "advance_interval",
        lambda *args, **kwargs: pytest.fail("invalid input reached integration"),
    )
    with pytest.raises(EvolutionValidationError) as error:
        fql.run(X, [1.0, 0.0], [0.0, 1.0], basis_labels=labels)
    assert (error.value.code, error.value.field) == (
        "invalid_basis_labels",
        "basis_labels",
    )


@pytest.mark.parametrize(
    "extra",
    [{"dimensions": (2,)}, {"n_qubits": 1}, {"dimensions": (2,), "n_qubits": 1}],
)
def test_labels_are_exclusive_even_when_product_numbers_agree(
    extra: dict[str, object],
) -> None:
    with pytest.raises(EvolutionValidationError) as error:
        fql.plan(X, [1.0, 0.0], [0.0, 1.0], basis_labels=LABELS, **extra)
    assert (error.value.code, error.value.field) == (
        "conflicting_state_space",
        "basis_labels",
    )


@pytest.mark.parametrize(
    "form",
    [
        "pauli",
        "observable",
        "bitstring",
        "named_collapse",
        "local_collapse",
        "local_term",
        "qubit_output",
        "scheduled_pauli",
    ],
)
def test_qubit_forms_are_refused_on_a_two_state_explicit_basis(form: str) -> None:
    h: object = X
    initial: object = [1.0, 0.0]
    kwargs: dict[str, Any] = {}
    field = "hamiltonian"
    if form in ("pauli", "scheduled_pauli"):
        h = [{"operator": "X", "qubits": [0], "coefficient": 0.5}]
        if form == "scheduled_pauli":
            kwargs["parameter_values"] = [{"g": 0.5}]
    elif form == "observable":
        h = fq.X(0)
    elif form == "bitstring":
        initial, field = "0", "initial_state"
    elif form == "named_collapse":
        kwargs["collapse_operators"], field = [
            fql.amplitude_damping(0.1, 0)
        ], "collapse_operators"
    elif form == "local_collapse":
        kwargs["collapse_operators"], field = [
            {"operator": X, "rate": 0.1, "qubits": []}
        ], "collapse_operators"
    elif form == "local_term":
        h = [{"operator": X, "qubits": [0]}]
    else:
        kwargs["outputs"], field = fq.expectation(fq.Z(0)), "observables"
    with pytest.raises(EvolutionValidationError) as error:
        fql.run(h, initial, [0.0, 0.1], basis_labels=LABELS, **kwargs)
    assert (error.value.code, error.value.field) == ("qubit_only_input", field)


@pytest.mark.parametrize("field", ["hamiltonian", "collapse_operators", "observables"])
def test_direct_boson_inputs_require_product_dimensions(field: str) -> None:
    kwargs: dict[str, Any] = {"hamiltonian": X}
    if field == "hamiltonian":
        kwargs[field] = create(0) + annihilate(0)
    elif field == "collapse_operators":
        kwargs[field] = [(annihilate(0), 0.1)]
    else:
        kwargs[field] = {"number": create(0) * annihilate(0)}
    with pytest.raises(EvolutionValidationError) as error:
        engine.evolve_density_matrix(
            initial_state=[1.0, 0.0],
            n_qubits=None,
            times=[0.0, 0.1],
            basis_labels=LABELS,
            **kwargs,
        )
    assert error.value.code == "boson_dimensions_required"


@pytest.mark.parametrize("method", fql.INTEGRATOR_NAMES)
def test_vacuum_and_one_photon_sector_matches_ambient_space_and_loss_law(
    method: str,
) -> None:
    full_h = (create(0) * annihilate(1) + create(1) * annihilate(0)).to_matrix(
        {0: 2, 1: 2}
    )
    full_c = [annihilate(mode).to_matrix({0: 2, 1: 2}) for mode in (0, 1)]
    labels = ((0, 0), (1, 0), (0, 1))
    embedding = torch.eye(4, dtype=torch.complex128)[:, [0, 2, 1]]
    complement = torch.eye(4, dtype=torch.complex128) - embedding @ embedding.mH
    for operator in [full_h, *full_c, *(c.mH @ c for c in full_c)]:
        assert torch.count_nonzero(complement @ operator @ embedding) == 0
    coupling = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    times = torch.linspace(0.0, 0.5, 201, dtype=torch.float64)
    full = fql.run(
        coupling * full_h,
        [0.0, 0.0, 1.0, 0.0],
        times,
        dimensions=(2, 2),
        collapse_operators=[(c, 0.2) for c in full_c],
        method=method,
        return_density_matrices=True,
    )
    result = fql.run(
        coupling * (embedding.mH @ full_h @ embedding),
        [0.0, 1.0, 0.0],
        times,
        basis_labels=labels,
        collapse_operators=[(embedding.mH @ c @ embedding, 0.2) for c in full_c],
        method=method,
        options=fq.ExecutionOptions(require_gradients=True),
        return_density_matrices=True,
    )
    assert full.density_matrices is not None and result.density_matrices is not None
    torch.testing.assert_close(
        embedding @ result.density_matrices @ embedding.mH,
        full.density_matrices,
        atol=1e-10,
        rtol=0,
    )
    survival = math.exp(-0.1)
    tolerance = 1e-6 if method == "crank-nicolson" else 1e-10
    assert result.populations[-1, labels.index((0, 0))].item() == pytest.approx(
        1 - survival, abs=tolerance
    )
    transfer = result.populations[-1, labels.index((0, 1))]
    assert transfer.item() == pytest.approx(
        survival * math.sin(0.35) ** 2, abs=tolerance
    )
    (gradient,) = torch.autograd.grad(transfer, coupling)
    assert gradient.item() == pytest.approx(
        0.5 * survival * math.sin(0.7), abs=tolerance
    )


@pytest.mark.parametrize("scheduled", [False, True])
def test_labelled_plan_roundtrip_retains_axes_and_a_single_numeric_request(
    scheduled: bool,
) -> None:
    plan = _plan(scheduled=scheduled)
    payload = plan.to_dict()
    assert (
        payload["request"]["basis_labels"]
        == payload["decision"]["basis_labels"]
        == [[0, 1], [1, 0]]
    )
    assert ("hamiltonians" in payload["request"]) is scheduled
    assert ("hamiltonian" in payload["request"]) is not scheduled
    restored = fql.LindbladPlan.from_json(plan.to_json())
    assert restored.to_json() == plan.to_json()
    assert restored.basis_labels == LABELS
    result, replay = fql.run(plan), fql.run(restored)
    torch.testing.assert_close(
        result.populations, replay.populations, atol=1e-12, rtol=0
    )
    assert result.density_matrices is not None and replay.density_matrices is not None
    torch.testing.assert_close(
        result.density_matrices, replay.density_matrices, atol=1e-12, rtol=0
    )
    assert "basis_labels" not in result.to_dict()
    assert result.to_dict()["plan_identity"] == plan.identity
    low = (
        engine.evolve_density_matrix(**restored._execution_request())
        if not scheduled
        else engine._evolve_density_matrix(**restored._execution_request())
    )
    torch.testing.assert_close(low.populations, result.populations, atol=1e-12, rtol=0)
    with pytest.raises(TypeError, match="semantic overrides"):
        fql.run(plan, basis_labels=LABELS)


def test_labels_are_frozen_into_plans_and_label_order_enters_identity() -> None:
    labels = [[0, 1], [1, 0]]
    plan = fql.plan(X, [1.0, 0.0], [0.0, 0.1], basis_labels=labels)
    labels[0][0] = 99
    assert plan.basis_labels == LABELS
    other = fql.plan(X, [1.0, 0.0], [0.0, 0.1], basis_labels=tuple(reversed(LABELS)))
    assert other.identity != plan.identity
    public_copy = plan.to_dict()
    public_copy["request"]["basis_labels"][0][0] = 99
    assert plan.basis_labels == LABELS


@pytest.mark.parametrize("scheduled", [False, True])
def test_serialization_does_not_fabricate_a_gradient_graph(scheduled: bool) -> None:
    restored = fql.LindbladPlan.from_json(
        _plan(scheduled=scheduled, gradients=True).to_json()
    )
    with pytest.raises(EvolutionValidationError) as error:
        fql.run(restored)
    assert error.value.code == "gradients_not_available"


@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize(
    "part,key,value",
    [
        ("request", "basis_labels", [[0], [0]]),
        ("request", "basis_labels", [[False], [1]]),
        ("request", "basis_labels", [[0.0], [1]]),
        ("request", "basis_labels", [[""], ["one"]]),
        ("request", "basis_labels", [[1, 0], [0, 1]]),
        ("decision", "basis_labels", [[1, 0], [0, 1]]),
        ("decision", "n_wires", 1),
        ("decision", "n_wires", False),
        ("decision", "dimension", True),
        ("decision", "n_times", 3.0),
        ("decision", "trajectory_bytes", 0),
        ("decision", "batch_size", False),
        ("request", "n_qubits", 1),
        ("request", "n_qubits", "two"),
        ("request", "dimensions", [2]),
        ("decision", "dimensions", None),
        ("request", "sector", "one-photon"),
        ("decision", "unknown", 0),
    ],
)
def test_rehashed_structural_metadata_is_revalidated(
    scheduled: bool, part: str, key: str, value: object
) -> None:
    payload = _plan(scheduled=scheduled).to_dict()
    payload[part][key] = value
    with pytest.raises(SerializationError):
        fql.LindbladPlan.from_dict(_resign(payload))


@pytest.mark.parametrize("field", ["hamiltonian", "initial_density_matrix"])
def test_decoded_numeric_validation_keeps_its_existing_error_category(
    field: str,
) -> None:
    payload = _plan().to_dict()
    payload["request"][field] = _complex_tensor_payload(
        torch.eye(3, dtype=torch.complex128)
    )
    with pytest.raises(EvolutionValidationError) as error:
        fql.LindbladPlan.from_dict(_resign(payload))
    assert error.value.code == "invalid_shape"


@pytest.mark.parametrize("part", ["request", "decision", "top"])
@pytest.mark.parametrize("version", ["1.3", "1.4", "1.5"])
def test_old_versions_refuse_disguised_labelled_semantics(
    part: str, version: str
) -> None:
    dimension = 3 if version == "1.4" else 2
    h = torch.zeros((dimension, dimension), dtype=torch.complex128)
    kwargs: dict[str, Any] = {"dimensions": (dimension,)}
    if version == "1.5":
        h = [{"operator": h, "coefficient": fq.Parameter("g")}]
        kwargs["parameter_values"] = [{"g": 0.3}]
    payload = fql.plan(
        h, [1.0] + [0.0] * (dimension - 1), [0.0, 0.1], **kwargs
    ).to_dict()
    assert payload["version"] == version
    target = payload if part == "top" else payload[part]
    target["basis_labels"] = [[index] for index in range(dimension)]
    with pytest.raises(SerializationError):
        fql.LindbladPlan.from_dict(_resign(payload))


@pytest.mark.parametrize("method", fql.INTEGRATOR_NAMES)
@pytest.mark.parametrize("precision", ["complex64", "complex128"])
@pytest.mark.parametrize("scheduled", [False, True])
def test_initial_density_adjoint_matches_independent_discrete_map_and_preserves_legacy_graph(
    method: str, precision: str, scheduled: bool
) -> None:
    dtype = torch.complex64 if precision == "complex64" else torch.complex128
    rho = torch.tensor([[0.6, 0.1j], [-0.1j, 0.4]], dtype=dtype, requires_grad=True)
    h: object = 0.3 * X
    values = None
    if scheduled:
        h = [{"operator": X, "coefficient": fq.Parameter("g")}]
        values = [{"g": 0.2}, {"g": 0.7}]
    collapse = [(torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=dtype), 0.2)]
    times = [0.0, 0.04, 0.1]

    def cost(trajectory: torch.Tensor) -> torch.Tensor:
        return trajectory[-1, 1, 1].real + 0.2 * trajectory[1, 0, 1].imag

    result = fql.run(
        h,
        rho,
        times,
        basis_labels=LABELS,
        collapse_operators=collapse,
        parameter_values=values,
        method=method,
        options=fq.ExecutionOptions(precision=precision, require_gradients=True),
        return_density_matrices=True,
    )
    assert result.density_matrices is not None
    (graph_gradient,) = torch.autograd.grad(cost(result.density_matrices), rho)
    # A differentiated finite-precision Krylov basis has conditioning limits
    # independent of the state-space spelling. Keep that legacy comparison
    # separate from the exact discrete-map reference for the initial adjoint.
    legacy_rho = rho.detach().clone().requires_grad_(True)
    legacy = fql.run(
        h,
        legacy_rho,
        times,
        dimensions=(2,),
        collapse_operators=collapse,
        parameter_values=values,
        method=method,
        options=fq.ExecutionOptions(precision=precision, require_gradients=True),
        return_density_matrices=True,
    )
    assert legacy.density_matrices is not None
    (legacy_gradient,) = torch.autograd.grad(cost(legacy.density_matrices), legacy_rho)
    parity_tolerance = 1e-7 if precision == "complex64" else 1e-12
    torch.testing.assert_close(
        graph_gradient, legacy_gradient, atol=parity_tolerance, rtol=0
    )

    reference_rho = rho.detach().clone().requires_grad_(True)
    factors = (0.2, 0.7) if scheduled else (0.3, 0.3)
    reference = _dense_two_state_trajectory(
        [(factor * X).to(dtype=dtype) for factor in factors],
        reference_rho,
        times,
        collapse[0][0],
        collapse[0][1],
        method,
    )
    (expected,) = torch.autograd.grad(cost(reference), reference_rho)
    actual = fql.adjoint_gradient(
        h,
        rho.detach(),
        times,
        cost,
        basis_labels=LABELS,
        collapse_operators=collapse,
        parameter_values=values,
        method=method,
        options=fq.ExecutionOptions(precision=precision),
    )
    low = initial_adjoint(
        h,
        rho.detach(),
        None,
        times,
        cost,
        basis_labels=LABELS,
        collapse_operators=collapse,
        parameter_values=values,
        method=method,
        dtype=dtype,
    )
    tolerance = 2e-5 if precision == "complex64" else 1e-9
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0)
    torch.testing.assert_close(low, expected, atol=tolerance, rtol=0)


@pytest.mark.parametrize("method", fql.INTEGRATOR_NAMES)
def test_zero_first_interval_keeps_the_live_later_control_sensitivity(
    method: str,
) -> None:
    values = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
    terms = [{"operator": X, "coefficient": 0.5 * fq.Parameter("g")}]

    def probability(controls: torch.Tensor) -> torch.Tensor:
        return fql.run(
            terms,
            [1.0, 0.0],
            [0.0, 0.1, 0.3],
            basis_labels=LABELS,
            parameter_values=[{"g": value} for value in controls.unbind()],
            method=method,
        ).populations[-1, 1]

    (gradient,) = torch.autograd.grad(probability(values), values)
    for index in range(2):
        delta = torch.zeros_like(values)
        delta[index] = 1e-5
        stencil = (
            probability(values.detach() + delta) - probability(values.detach() - delta)
        ) / 2e-5
        torch.testing.assert_close(gradient[index], stencil, atol=2e-9, rtol=0)
    assert gradient[0].item() > 0.009


@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize(
    "field",
    [
        "hamiltonian",
        "initial_density_matrix",
        "basis_labels",
        "decision_labels",
        "numerics",
        "intervals",
        "top",
        "tensor_shape",
    ],
)
def test_required_fields_and_nested_semantics_are_revalidated(
    scheduled: bool, field: str
) -> None:
    payload = _plan(scheduled=scheduled).to_dict()
    hkey = "hamiltonians" if scheduled else "hamiltonian"
    tensor = payload["request"][hkey][0] if scheduled else payload["request"][hkey]
    if field == "hamiltonian":
        tensor["unexpected"] = 1
    elif field == "initial_density_matrix":
        payload["request"][field]["sector"] = "one-photon"
    elif field == "basis_labels":
        del payload["request"][field]
    elif field == "decision_labels":
        del payload["decision"]["basis_labels"]
    elif field == "numerics":
        payload["decision"]["numerics"]["unexpected"] = 1
    elif field == "intervals":
        payload["decision"]["numerics"]["n_intervals"] = 99
    elif field == "top":
        payload["basis_labels"] = [[0], [1]]
    else:
        tensor["shape"] = [True, 2]
    with pytest.raises(SerializationError):
        fql.LindbladPlan.from_dict(_resign(payload))


def test_a_consistent_relabelling_is_a_distinct_mathematical_request() -> None:
    original = _plan()
    payload = original.to_dict()
    labels = [["left"], ["right"]]
    payload["request"]["basis_labels"] = labels
    payload["decision"]["basis_labels"] = labels
    other = fql.LindbladPlan.from_dict(_resign(payload))
    assert other.identity != original.identity
    assert other.basis_labels == (("left",), ("right",))
    # Literal labels carry coordinates, not a certificate of a physical sector.
    torch.testing.assert_close(
        fql.run(other).populations, fql.run(original).populations, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    "dimension,scheduled", [(2, False), (3, False), (2, True), (3, True)]
)
def test_none_keeps_the_existing_product_payload_exactly(
    dimension: int, scheduled: bool
) -> None:
    h: object = torch.zeros((dimension, dimension), dtype=torch.complex128)
    kwargs: dict[str, Any] = {"dimensions": (dimension,)}
    if scheduled:
        h = [{"operator": h, "coefficient": fq.Parameter("g")}]
        kwargs["parameter_values"] = [{"g": 0.3}]
    initial = [1.0] + [0.0] * (dimension - 1)
    original = fql.plan(h, initial, [0.0, 0.1], **kwargs)
    explicit_none = fql.plan(h, initial, [0.0, 0.1], basis_labels=None, **kwargs)
    assert original.to_json() == explicit_none.to_json()
    assert original.basis_labels is None
    assert "basis_labels" not in original.to_dict()["request"]
    assert "basis_labels" not in original.to_dict()["decision"]


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
@pytest.mark.parametrize("scheduled", [False, True])
def test_product_writer_matches_captured_preimplementation_payloads(
    precision: str, scheduled: bool
) -> None:
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "fixtures/lindblad_product_plan_compatibility.json"
        ).read_text()
    )
    operator = torch.zeros((3, 3), dtype=torch.complex128)
    operator[0, 1] = operator[1, 0] = 1
    collapse = torch.zeros_like(operator)
    collapse[0, 1] = 1
    h: object = 0.7 * operator
    values = None
    if scheduled:
        h = [{"operator": operator, "coefficient": fq.Parameter("g")}]
        values = [{"g": 0.7}, {"g": 0.2}]
    plan = fql.plan(
        h,
        [0.0, 1.0, 0.0],
        [0.0, 0.07, 0.13],
        dimensions=(3,),
        collapse_operators=[(collapse, 0.1)],
        parameter_values=values,
        options=fq.ExecutionOptions(precision=precision),
        return_density_matrices=True,
        method="runge-kutta",
    )
    expected = fixture["plans"][f"{precision}/(3,)/{scheduled}"]
    assert plan.to_dict() == expected
    assert fql.LindbladPlan.from_dict(expected).to_dict() == expected
