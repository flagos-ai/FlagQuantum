"""Replaceable proposal 080 consumers using the existing numerical engine.

Only the four fixtures supply the temporary output boundary. Replace them with
OutputRequest, expectation, fql.plan and lower_outputs after implementation,
then remove the fake. Physical and compatibility assertions must not change.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from typing import TypeAlias

import pytest
import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.core.ir import MeasurementNode
from flagquantum.errors import CapabilityError
from flagquantum.lindblad._plan import build_lindblad_plan
from flagquantum.observables import Observable, OutputRequest, lower_outputs
from flagquantum.simulation.lindblad import EvolutionValidationError

pytestmark = pytest.mark.unit


@dataclass(frozen=True, slots=True, eq=False)
class _OutputFake:
    kind: str
    qubits: tuple[int, ...] = ()
    observable: Observable | torch.Tensor | None = None
    name: str | None = None
    log_base: float | None = None

    def __post_init__(self) -> None:
        matrix = isinstance(self.observable, torch.Tensor)
        if matrix and self.kind != "expectation":
            raise TypeError("matrix observables require expectation output")
        # Common names, kinds and selectors remain the existing owner's job.
        checked = fq.OutputRequest(
            self.kind,
            self.qubits,
            fq.I(0) if isinstance(self.observable, torch.Tensor) else self.observable,
            self.name,
            self.log_base,
        )
        object.__setattr__(self, "qubits", checked.qubits)
        if isinstance(self.observable, torch.Tensor):
            operator = self.observable
            if (
                operator.layout != torch.strided
                or operator.is_quantized
                or operator.device.type == "meta"
                or operator.dtype == torch.bool
            ):
                raise TypeError("expectation requires an ordinary numeric dense tensor")
            if (
                operator.ndim != 2
                or not operator.shape[0]
                or (operator.shape[0] != operator.shape[1])
            ):
                raise ValueError("expectation matrix must be nonempty and square")
            if self.qubits:
                raise ValueError(
                    "full-space expectations do not accept qubit selectors"
                )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _OutputFake) or type(self) is not type(other):
            return NotImplemented
        if (self.kind, self.qubits, self.name, self.log_base) != (
            other.kind,
            other.qubits,
            other.name,
            other.log_base,
        ):
            return False
        if isinstance(self.observable, torch.Tensor) or isinstance(
            other.observable, torch.Tensor
        ):
            return self.observable is other.observable
        return self.observable == other.observable

    def __hash__(self) -> int:
        operator = (
            id(self.observable)
            if isinstance(self.observable, torch.Tensor)
            else self.observable
        )
        return hash((self.kind, self.qubits, operator, self.name, self.log_base))


_Output: TypeAlias = _OutputFake | OutputRequest


def _expectation_fake(
    observable: Observable | torch.Tensor, *, name: str | None = None
) -> _OutputFake:
    return _OutputFake("expectation", observable=observable, name=name)


def _requests(outputs: _Output | Sequence[_Output] | None) -> tuple[_Output, ...]:
    if outputs is None:
        return ()
    values = (
        (outputs,)
        if isinstance(outputs, (_OutputFake, OutputRequest))
        else tuple(outputs)
    )
    if not values or any(
        not isinstance(value, (_OutputFake, OutputRequest)) for value in values
    ):
        raise TypeError("outputs must contain output requests")
    return values


def _named_matrices(outputs: _Output | Sequence[_Output] | None) -> dict[str, object]:
    matrices: dict[str, object] = {}
    for index, request in enumerate(_requests(outputs)):
        if request.kind != "expectation":
            raise CapabilityError("Lindblad outputs support only expectations")
        name = request.name or f"expectation_{index}"
        if name in matrices:
            raise EvolutionValidationError(
                "duplicate_output_name", "output names must be unique", field="outputs"
            )
        matrices[name] = (
            request.observable.clone()
            if isinstance(request.observable, torch.Tensor)
            else request.observable
        )
    return matrices


def _plan_fake(
    hamiltonian: object,
    initial_state: object,
    times: object,
    *,
    outputs: _Output | Sequence[_Output] | None = None,
    options: fq.ExecutionOptions | None = None,
    dimensions: Sequence[int] | None = None,
    basis_labels: Sequence[Sequence[int | str]] | None = None,
    parameter_values: Sequence[Mapping[str | fq.Parameter, object]] | None = None,
    n_qubits: int | None = None,
    collapse_operators: Sequence[object] | None = None,
    method: str = "runge-kutta",
    return_density_matrices: bool = False,
) -> fql.LindbladPlan:
    matrices = _named_matrices(outputs)
    device, dtype, batch, gradients = fql._execution_settings(options)
    qubits = (
        fql._infer_n_qubits(initial_state)
        if n_qubits is None and dimensions is None and basis_labels is None
        else n_qubits
    )
    return build_lindblad_plan(
        hamiltonian,
        initial_state,
        qubits,
        times,
        collapse_operators,
        matrices,
        device=device,
        dtype=dtype,
        batch_size=batch,
        require_gradients=gradients,
        dimensions=dimensions,
        basis_labels=basis_labels,
        parameter_values=parameter_values,
        method=method,
        return_density_matrices=return_density_matrices,
    )


def _lower_fake(
    outputs: _Output | Sequence[_Output] | None,
    *,
    n_qubits: int,
    shots: int | None,
) -> tuple[MeasurementNode, ...] | None:
    old: list[fq.OutputRequest] = []
    for request in _requests(outputs):
        if isinstance(request.observable, torch.Tensor):
            raise CapabilityError("matrix expectations require the Lindblad path")
        old.append(
            fq.OutputRequest(
                request.kind,
                request.qubits,
                request.observable,
                request.name,
                request.log_base,
            )
        )
    return lower_outputs(old or None, n_qubits=n_qubits, shots=shots)


@pytest.fixture
def request_factory() -> Callable[..., _Output]:
    return _OutputFake


@pytest.fixture
def expectation_factory() -> Callable[..., _Output]:
    return _expectation_fake


@pytest.fixture
def plan_factory() -> Callable[..., fql.LindbladPlan]:
    return _plan_fake


@pytest.fixture
def output_lowering() -> Callable[..., tuple[MeasurementNode, ...] | None]:
    return _lower_fake


def _transition() -> torch.Tensor:
    matrix = torch.zeros((3, 3), dtype=torch.complex128)
    matrix[1, 2] = matrix[2, 1] = 1
    return matrix


def _projector() -> torch.Tensor:
    return torch.diag(torch.tensor([0.0, 0.0, 1.0], dtype=torch.complex128))


@pytest.mark.parametrize("method", fql.INTEGRATOR_NAMES)
def test_qutrit_expectation_and_gradient_use_existing_engine(
    expectation_factory: Callable[..., _Output],
    plan_factory: Callable[..., fql.LindbladPlan],
    method: str,
) -> None:
    g = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    times = torch.linspace(0.0, 0.5, 201, dtype=torch.float64)
    plan = plan_factory(
        g * _transition(),
        [0.0, 1.0, 0.0],
        times,
        dimensions=(3,),
        method=method,
        outputs=expectation_factory(_projector(), name="leakage"),
        options=fq.ExecutionOptions(require_gradients=True),
    )
    result = fql.run(plan)
    (gradient,) = torch.autograd.grad(result.expectation("leakage")[-1], g)
    tolerance = 1e-6 if method == "crank-nicolson" else 1e-10
    torch.testing.assert_close(
        result.expectation("leakage"),
        torch.sin(0.7 * times) ** 2,
        atol=tolerance,
        rtol=0,
    )
    assert gradient.item() == pytest.approx(0.5 * math.sin(0.7), abs=tolerance)
    assert result.plan is plan
    assert result.expectation().shape == (201,)
    assert torch.equal(result.expectation(0), result.populations[:, 2])
    with pytest.raises(EvolutionValidationError) as caught:
        fql.run(fql.LindbladPlan.from_json(plan.to_json()))
    assert caught.value.code == "gradients_not_available"


def test_crank_nicolson_gradient_converges_without_relaxing_error_budget(
    expectation_factory: Callable[..., _Output],
    plan_factory: Callable[..., fql.LindbladPlan],
) -> None:
    errors = []
    for count in (51, 101, 201):
        g = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
        result = fql.run(
            plan_factory(
                g * _transition(),
                [0.0, 1.0, 0.0],
                torch.linspace(0.0, 0.5, count, dtype=torch.float64),
                dimensions=(3,),
                method="crank-nicolson",
                outputs=expectation_factory(_projector()),
                options=fq.ExecutionOptions(require_gradients=True),
            )
        )
        (gradient,) = torch.autograd.grad(result.expectation()[-1], g)
        steps = 2 * (count - 1)
        dt = 0.5 / steps
        angle = steps * math.atan(0.7 * dt)
        # The exact CN rotation of this Bloch vector is independent of GMRES AD.
        discrete = math.sin(2 * angle) * steps * dt / (1 + (0.7 * dt) ** 2)
        assert gradient.item() == pytest.approx(discrete, abs=1e-10)
        errors.append(abs(gradient.item() - 0.5 * math.sin(0.7)))
    assert 3.9 < errors[0] / errors[1] < 4.1
    assert 3.9 < errors[1] / errors[2] < 4.1
    assert errors[-1] < 1e-6


@pytest.mark.integration
def test_optimizer_and_finite_difference_use_the_same_output_boundary(
    expectation_factory: Callable[..., _Output],
    plan_factory: Callable[..., fql.LindbladPlan],
) -> None:
    coupling = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
    optimizer = torch.optim.Adam([coupling], lr=0.04)
    target = math.sin(0.35) ** 2
    times = torch.linspace(0.0, 0.5, 51, dtype=torch.float64)
    losses = []
    for step in range(40):
        optimizer.zero_grad()
        plan = plan_factory(
            coupling * _transition(),
            [0.0, 1.0, 0.0],
            times,
            dimensions=(3,),
            outputs=expectation_factory(_projector(), name="leakage"),
            options=fq.ExecutionOptions(require_gradients=True),
        )
        value = fql.run(plan).expectation("leakage")[-1]
        if step == 0:
            (gradient,) = torch.autograd.grad(value, coupling, retain_graph=True)
            epsilon = 1e-5
            samples = []
            for offset in (-epsilon, epsilon):
                probe = plan_factory(
                    (coupling.detach() + offset) * _transition(),
                    [0.0, 1.0, 0.0],
                    times,
                    dimensions=(3,),
                    outputs=expectation_factory(_projector()),
                )
                samples.append(fql.run(probe).expectation()[-1].item())
            difference = (samples[1] - samples[0]) / (2 * epsilon)
            assert gradient.item() == pytest.approx(difference, abs=1e-8)
        loss = (value - target).square()
        torch.autograd.backward(loss)
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] / 10


@pytest.mark.parametrize("batch_size", [1, 2])
def test_complex_labelled_outputs_match_independent_unitary_and_coordinates(
    expectation_factory: Callable[..., _Output],
    plan_factory: Callable[..., fql.LindbladPlan],
    batch_size: int,
) -> None:
    g = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    alpha = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, -1j], [0.0, 1j, 0.0]], dtype=torch.complex128
    )
    operator = alpha * _projector() + 0.1 * y
    initial = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=torch.complex128)
    initial[1] /= math.sqrt(2)
    initial = initial[:batch_size]
    times = torch.linspace(0.0, 0.5, 51, dtype=torch.float64)
    labels = (("vac",), ("left",), ("right",))
    output = expectation_factory(operator, name="reading")
    plan = plan_factory(
        g * y,
        initial,
        times,
        basis_labels=labels,
        outputs=output,
        options=fq.ExecutionOptions(batch_size=batch_size, require_gradients=True),
    )
    result = fql.run(plan)
    states = torch.einsum(
        "tij,bj->tbi", torch.matrix_exp(-1j * times[:, None, None] * g * y), initial
    )
    expected = torch.einsum("tbi,ij,tbj->tb", states.conj(), operator, states).real
    actual = result.expectation("reading")
    assert actual.shape == (51, batch_size)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)
    gradients = torch.autograd.grad(actual[-1].sum(), (g, alpha), retain_graph=True)
    references = torch.autograd.grad(expected[-1].sum(), (g, alpha))
    for gradient, reference in zip(gradients, references, strict=True):
        torch.testing.assert_close(gradient, reference, atol=1e-9, rtol=0)
    order = torch.tensor([2, 0, 1])
    permuted = plan_factory(
        (g * y)[order][:, order],
        initial[:, order],
        times,
        basis_labels=tuple(labels[index] for index in order.tolist()),
        outputs=expectation_factory(operator[order][:, order]),
        options=fq.ExecutionOptions(batch_size=batch_size),
    )
    torch.testing.assert_close(
        fql.run(permuted).expectation(), actual, atol=1e-10, rtol=0
    )
    wrong = plan_factory(
        (g * y)[order][:, order],
        initial[:, order],
        times,
        basis_labels=tuple(labels[index] for index in order.tolist()),
        outputs=output,
        options=fq.ExecutionOptions(batch_size=batch_size),
    )
    assert not torch.allclose(fql.run(wrong).expectation(), actual, atol=1e-3, rtol=0)


def test_spin_mode_expectation_reuses_product_order(
    expectation_factory: Callable[..., _Output],
    plan_factory: Callable[..., fql.LindbladPlan],
) -> None:
    from flagquantum.observables.boson import annihilate

    lower = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex128)
    ladder = annihilate(0).to_matrix({0: 3})
    h = 0.7 * (
        torch.kron(lower.T.contiguous(), ladder)
        + torch.kron(lower, ladder.T.contiguous())
    )
    operator = torch.kron(torch.eye(2), torch.diag(torch.tensor([0.0, 1.0, 2.0])))
    times = torch.linspace(0.0, 0.5, 51, dtype=torch.float64)
    result = fql.run(
        plan_factory(
            h,
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            times,
            dimensions=(2, 3),
            outputs=expectation_factory(operator, name="occupation"),
        )
    )
    torch.testing.assert_close(
        result.expectation("occupation"),
        torch.sin(0.7 * times) ** 2,
        atol=1e-10,
        rtol=0,
    )


def test_schedule_binding_and_initial_state_gradients(
    expectation_factory: Callable[..., _Output],
    plan_factory: Callable[..., fql.LindbladPlan],
) -> None:
    drive = fq.Parameter("drive")
    amplitudes = torch.tensor([0.3, 0.7, 0.4], dtype=torch.float64, requires_grad=True)
    beta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    rho = torch.diag(torch.stack((beta * 0, 1 - beta, beta))).to(torch.complex128)
    times = [0.0, 0.1, 0.25, 0.5]
    plan = plan_factory(
        [{"operator": _transition(), "coefficient": drive}],
        rho,
        times,
        dimensions=(3,),
        parameter_values=[{drive: value} for value in amplitudes],
        outputs=expectation_factory(_projector()),
        options=fq.ExecutionOptions(require_gradients=True),
    )
    value = fql.run(plan).expectation()[-1]
    duration = amplitudes.new_tensor([0.1, 0.15, 0.25])
    angle = (amplitudes.detach() * duration).sum()
    expected = beta.detach() + (1 - 2 * beta.detach()) * torch.sin(angle) ** 2
    gradient, initial_gradient = torch.autograd.grad(value, (amplitudes, beta))
    torch.testing.assert_close(value, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        gradient,
        (1 - 2 * beta.detach()) * duration * torch.sin(2 * angle),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        initial_gradient, torch.cos(2 * angle), atol=1e-6, rtol=0
    )


def test_matrix_snapshot_preserves_values_and_output_only_gradient(
    expectation_factory: Callable[..., _Output],
    plan_factory: Callable[..., fql.LindbladPlan],
) -> None:
    alpha = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    source = alpha * _projector()
    plan = plan_factory(
        0.7 * _transition(),
        [0.0, 1.0, 0.0],
        torch.linspace(0.0, 0.5, 51),
        dimensions=(3,),
        outputs=expectation_factory(source, name="reading"),
    )
    encoded, identity = plan.to_json(), plan.identity
    with torch.no_grad():
        source.zero_()
    result = fql.run(plan)
    (gradient,) = torch.autograd.grad(result.expectation()[-1], alpha)
    assert result.expectation()[-1].item() == pytest.approx(
        0.6 * math.sin(0.35) ** 2, abs=1e-10
    )
    assert gradient.item() == pytest.approx(math.sin(0.35) ** 2, abs=1e-10)
    assert plan.identity == identity and plan.to_json() == encoded
    replay = fql.run(fql.LindbladPlan.from_json(encoded))
    assert torch.equal(result.expectation(), replay.expectation())
    with pytest.raises(EvolutionValidationError) as caught:
        fql.run(
            plan_factory(
                0.7 * _transition(),
                [0.0, 1.0, 0.0],
                [0.0, 0.1],
                dimensions=(3,),
                outputs=expectation_factory(alpha * _projector()),
                options=fq.ExecutionOptions(require_gradients=True),
            )
        )
    assert caught.value.code == "gradients_not_available"


@pytest.mark.parametrize(
    "kind", ["samples", "counts", "probabilities", "density_matrix", "vn_entropy"]
)
def test_tensor_is_refused_on_other_request_kinds(
    request_factory: Callable[..., _Output],
    kind: str,
) -> None:
    with pytest.raises(TypeError):
        request_factory(kind, observable=torch.eye(3))


@pytest.mark.parametrize(
    "operator",
    [torch.zeros(3), torch.zeros(2, 3), torch.zeros(0, 0), torch.zeros(1, 2, 2)],
)
def test_invalid_matrix_rank_and_shape_fail_at_construction(
    expectation_factory: Callable[..., _Output],
    operator: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        expectation_factory(operator)


@pytest.mark.parametrize(
    "operator",
    [
        torch.eye(3, dtype=torch.bool),
        torch.eye(3).to_sparse(),
        torch.empty(3, 3, device="meta"),
    ],
)
def test_unsupported_matrix_representations_fail_at_construction(
    expectation_factory: Callable[..., _Output],
    operator: torch.Tensor,
) -> None:
    with pytest.raises(TypeError):
        expectation_factory(operator)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
        torch.int64,
        torch.uint8,
    ],
)
def test_ordinary_numeric_matrix_keeps_the_live_input(
    expectation_factory: Callable[..., _Output],
    dtype: torch.dtype,
) -> None:
    operator = torch.eye(3, dtype=dtype)
    request = expectation_factory(operator, name="reading")
    assert request.observable is operator
    assert request.kind == "expectation" and request.qubits == ()


def test_quantized_matrix_and_nonmatrix_values_are_refused(
    expectation_factory: Callable[..., _Output],
) -> None:
    from flagquantum.observables.boson import annihilate

    operator = torch.quantize_per_tensor(torch.eye(3), 0.1, 0, torch.quint8)
    for value in (operator, None, [[1.0, 0.0], [0.0, 1.0]], annihilate(0)):
        with pytest.raises(TypeError):
            expectation_factory(value)


def test_matrix_selector_and_numerical_refusals_precede_execution(
    request_factory: Callable[..., _Output],
    expectation_factory: Callable[..., _Output],
    plan_factory: Callable[..., fql.LindbladPlan],
) -> None:
    with pytest.raises(ValueError):
        request_factory("expectation", qubits=(0,), observable=torch.eye(3))
    for operator, code in [
        (torch.eye(2), "invalid_shape"),
        (torch.full((3, 3), math.nan), "non_finite_value"),
        (torch.triu(torch.ones(3, 3), diagonal=1), "non_hermitian_observable"),
    ]:
        with pytest.raises(EvolutionValidationError) as caught:
            plan_factory(
                _transition(),
                [0.0, 1.0, 0.0],
                [0.0, 0.1],
                dimensions=(3,),
                outputs=expectation_factory(operator, name="reading"),
            )
        assert caught.value.code == code
        assert "observables" in caught.value.field


def test_names_order_identity_and_closed_plan_behavior(
    expectation_factory: Callable[..., _Output],
    plan_factory: Callable[..., fql.LindbladPlan],
) -> None:
    first = expectation_factory(_projector(), name="reading")
    second = expectation_factory(torch.eye(3))
    plan = plan_factory(
        _transition(),
        [0.0, 1.0, 0.0],
        [0.0, 0.1],
        dimensions=(3,),
        outputs=[first, second],
    )
    result = fql.run(plan)
    assert tuple(result.observables) == ("reading", "expectation_1")
    assert torch.equal(result.expectation(0), result.expectation("reading"))
    assert torch.equal(result.expectations[1], result.expectation("expectation_1"))
    for outputs in [
        [expectation_factory(_projector(), name="other"), second],
        [expectation_factory(2 * _projector(), name="reading"), second],
        [second, first],
    ]:
        changed = plan_factory(
            _transition(), [0.0, 1.0, 0.0], [0.0, 0.1], dimensions=(3,), outputs=outputs
        )
        assert changed.identity != plan.identity
    with pytest.raises(TypeError, match="semantic overrides"):
        fql.run(plan, outputs=fq.expectation(fq.Z(0)))
    for outputs in [
        [expectation_factory(_projector(), name="expectation_1"), second],
        [first, expectation_factory(torch.eye(3), name="reading")],
        [
            expectation_factory(fq.Z(0), name="expectation_1"),
            expectation_factory(fq.X(0)),
        ],
    ]:
        with pytest.raises(EvolutionValidationError) as caught:
            plan_factory(
                _transition(),
                [0.0, 1.0, 0.0],
                [0.0, 0.1],
                dimensions=(3,),
                outputs=outputs,
            )
        assert caught.value.code == "duplicate_output_name"
        assert caught.value.field == "outputs"


def test_request_equality_hash_and_frozen_fields_do_not_read_tensor_values(
    request_factory: Callable[..., _Output],
    expectation_factory: Callable[..., _Output],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = torch.eye(3)
    first = expectation_factory(matrix, name="reading")
    same = expectation_factory(matrix, name="reading")
    clone = expectation_factory(matrix.clone(), name="reading")
    different_name = expectation_factory(matrix, name="other")
    pauli = expectation_factory(fq.Z(0), name="reading")
    old = fq.expectation(fq.Z(0), name="reading")
    old_copy = request_factory("expectation", observable=fq.Z(0), name="reading")
    subclass = type("DistinctOutputRequest", (type(first),), {})
    other_class = subclass(
        first.kind, first.qubits, first.observable, first.name, first.log_base
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("comparison must not read tensor values")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__eq__", forbidden)
        patch.setattr(torch, "equal", forbidden)
        patch.setattr(torch, "allclose", forbidden)
        assert first == same and first != clone and first != different_name
        assert first != pauli and pauli != first
        assert first != other_class and other_class != first
        assert (
            hash(first)
            == hash(same)
            == hash(("expectation", (), id(matrix), "reading", None))
        )
        assert hash(old_copy) == hash(old)
    before = hash(first)
    matrix.zero_()
    assert hash(first) == before
    assert old_copy == request_factory(
        "expectation", observable=fq.Z(0), name="reading"
    )
    with pytest.raises(FrozenInstanceError):
        setattr(first, "name", "changed")


def test_unsupported_circuit_lowering_refuses_matrix_before_pauli_access(
    expectation_factory: Callable[..., _Output],
    output_lowering: Callable[..., tuple[MeasurementNode, ...] | None],
) -> None:
    with pytest.raises(CapabilityError, match="Lindblad"):
        output_lowering(expectation_factory(torch.eye(2)), n_qubits=1, shots=None)
    pauli = expectation_factory(fq.Z(0), name="z")
    expected = lower_outputs(fq.expectation(fq.Z(0), name="z"), n_qubits=1, shots=None)
    assert output_lowering(pauli, n_qubits=1, shots=None) == expected


def _legacy_output_plan(kind: str, precision: str) -> fql.LindbladPlan:
    options = fq.ExecutionOptions(precision=precision)
    if kind == "qubit":
        return fql.plan(
            torch.tensor([[0.0, 0.7], [0.7, 0.0]]),
            [1.0, 0.0],
            [0.0, 0.1],
            outputs=fq.expectation(fq.Z(0), name="z"),
            options=options,
        )
    scheduled = kind.endswith("scheduled")
    labelled = kind.startswith("labelled")
    drive = fq.Parameter("drive")
    hamiltonian = (
        [{"operator": _transition(), "coefficient": drive}]
        if scheduled
        else 0.7 * _transition()
    )
    dtype = {"complex64": torch.complex64, "complex128": torch.complex128}[precision]
    return build_lindblad_plan(
        hamiltonian,
        [0.0, 1.0, 0.0],
        None,
        [0.0, 0.1],
        None,
        {"reading": _projector()},
        device="cpu",
        dtype=dtype,
        return_density_matrices=False,
        dimensions=None if labelled else (3,),
        basis_labels=(("vac",), ("left",), ("right",)) if labelled else None,
        parameter_values=[{drive: 0.7}] if scheduled else None,
    )


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
@pytest.mark.parametrize(
    "kind,version",
    [
        ("qubit", "1.3"),
        ("product", "1.4"),
        ("product-scheduled", "1.5"),
        ("labelled", "1.6"),
        ("labelled-scheduled", "1.6"),
    ],
)
def test_preimplementation_output_plans_retain_exact_bytes(
    kind: str,
    version: str,
    precision: str,
) -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "fixtures/lindblad_output_plan_compatibility.json"
    )
    fixture = json.loads(path.read_text())
    expected = fixture["plans"][f"{kind}/{precision}"]
    plan = _legacy_output_plan(kind, precision)
    assert plan.to_dict()["version"] == version
    assert plan.to_dict() == expected
    assert plan.to_json() == json.dumps(
        expected, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    restored = fql.LindbladPlan.from_dict(expected)
    assert restored.identity == plan.identity
    assert torch.equal(fql.run(plan).expectation(), fql.run(restored).expectation())
