"""`fq.gradient`: the resolved method, the exact rules, and the approximations.

The entry point exists because a user had to know in advance which gradient route
their program supported and then call a different function or method for each
one. These tests check the two things that makes that claim true: the resolved
method is reported rather than assumed, and the methods agree with each other
wherever they are both available.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from torch import Tensor

import flagquantum as fq
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.errors import CapabilityError, ValidationError
from flagquantum.gradients import GradientResult
from flagquantum.gradients import gradient as module_gradient

pytestmark = pytest.mark.unit

#: Three parameters that all move the loss, so a relative-error comparison says
#: something. A parameter whose derivative happens to vanish would make the
#: comparison vacuous instead.
_PARAMETERS = torch.tensor([0.3, 0.7, -0.4], dtype=torch.float64)


def _circuit(parameters: torch.Tensor) -> fq.Circuit:
    """Build the shared reference circuit: three angles, one entangler."""

    return (
        fq.Circuit(2)
        .ry(0, theta=parameters[0])
        .rx(0, theta=parameters[1])
        .ry(1, theta=parameters[2])
        .cx(1, 0)
    )


def _loss(circuit: fq.Circuit) -> torch.Tensor:
    return circuit.expectation_z(0)


def _value(parameters: torch.Tensor) -> torch.Tensor:
    return _loss(_circuit(parameters))


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(((actual - expected).abs() / expected.abs()).max())


def test_the_result_reports_autograd_as_the_resolved_auto_method() -> None:
    result = fq.gradient(_value, _PARAMETERS)

    assert result.method == "autograd"
    assert result.exact is True
    assert result.step is None
    assert result.gradient.shape == _PARAMETERS.shape
    assert result.gradient.dtype == _PARAMETERS.dtype
    assert result.gradient.device == _PARAMETERS.device


def test_the_result_never_carries_the_autograd_graph() -> None:
    """A gradient is the answer, not an intermediate step of a longer graph."""

    assert fq.gradient(_value, _PARAMETERS).gradient.requires_grad is False
    assert fq.gradient(_circuit, _PARAMETERS, _loss).gradient.requires_grad is False


def test_auto_selects_parameter_shift_when_the_loss_carries_no_graph() -> None:
    def detached(circuit: fq.Circuit) -> torch.Tensor:
        return _loss(circuit).detach()

    result = fq.gradient(_circuit, _PARAMETERS, detached)

    assert result.method == "parameter_shift"
    assert result.exact is True
    assert result.step is None


def test_auto_selects_finite_difference_without_a_circuit_or_a_graph() -> None:
    def scalar(parameters: torch.Tensor) -> torch.Tensor:
        angle = parameters[0].detach()
        return torch.tensor(
            math.sin(float(angle)) * float(parameters[1].detach()),
            dtype=torch.float64,
        )

    parameters = torch.tensor([0.3, 0.7], dtype=torch.float64)
    result = fq.gradient(scalar, parameters)

    assert result.method == "finite_difference"
    assert result.exact is False
    assert result.step is not None
    expected = torch.tensor([math.cos(0.3) * 0.7, math.sin(0.3)], dtype=torch.float64)
    assert _relative_error(result.gradient, expected) < 1e-6


def test_the_single_callable_and_loss_forms_agree() -> None:
    """`fq.gradient(fn, p)` and `fq.gradient(build, p, loss)` are one computation."""

    direct = fq.gradient(_value, _PARAMETERS)
    with_loss = fq.gradient(_circuit, _PARAMETERS, _loss)

    assert torch.allclose(direct.gradient, with_loss.gradient, rtol=1e-6, atol=1e-9)
    assert direct.method == with_loss.method == "autograd"


@pytest.mark.parametrize("method", ["autograd", "parameter_shift"])
def test_the_public_entry_point_is_the_module_function(method: str) -> None:
    public = fq.gradient(_circuit, _PARAMETERS, _loss, method=method)
    internal = module_gradient(_circuit, _PARAMETERS, _loss, method=method)

    assert (public.method, public.exact, public.step) == (
        internal.method,
        internal.exact,
        internal.step,
    )
    assert torch.equal(public.gradient, internal.gradient)


def test_parameter_shift_matches_autograd_on_the_reference_circuit() -> None:
    exact = fq.gradient(_value, _PARAMETERS)
    shifted = fq.gradient(_circuit, _PARAMETERS, _loss, method="parameter_shift")

    assert _relative_error(shifted.gradient, exact.gradient) < 1e-5


#: Values of every parameter that is not the one being differentiated. They keep
#: the multi-angle opcodes on a point of the parameter space where each
#: parameter's own derivative is well separated from the others'.
_FIXED_PARAMETER_VALUE = 0.9


def _gate_probe(
    opcode: str, index: int, *, superposed: bool
) -> tuple[Callable[[torch.Tensor], fq.Circuit], Callable[[fq.Circuit], Tensor]]:
    """Build one circuit whose ``index``-th declared parameter is the free one.

    A single probe cannot see every gate. A rotation about an axis leaves the
    expectation value along that same axis untouched, and a diagonal gate is a
    global phase on a computational basis state and so is invisible without a
    coherent superposition. Two probes -- one on the basis state, one behind a
    uniform superposition -- are enough that every declared parameter of every
    differentiable opcode moves the loss, which is what makes the comparison
    below non-vacuous.
    """

    schema = OPERATOR_SCHEMAS[opcode]
    arity = schema.arity
    names = schema.parameters

    def circuit(parameters: torch.Tensor) -> fq.Circuit:
        built = fq.Circuit(arity)
        if superposed:
            for qubit in range(arity):
                built.h(qubit)
        angles = {
            name: (parameters[0] if position == index else _FIXED_PARAMETER_VALUE)
            for position, name in enumerate(names)
        }
        getattr(built, opcode)(*range(arity), **angles)
        return built

    def loss(built: fq.Circuit) -> Tensor:
        return sum(
            fq.run(
                built,
                outputs=[
                    fq.expectation(observable(qubit))
                    for observable in (fq.X, fq.Y, fq.Z)
                    for qubit in range(arity)
                ],
            ).expectations
        )

    return circuit, loss


def test_every_differentiable_gate_parameter_agrees_between_shift_and_autograd() -> (
    None
):
    """The exact rule is declared per opcode, so check every declared parameter."""

    differentiable = sorted(
        name for name, schema in OPERATOR_SCHEMAS.items() if schema.differentiable
    )
    assert len(differentiable) == 14

    compared = 0
    for opcode in differentiable:
        names = OPERATOR_SCHEMAS[opcode].parameters
        for index, name in enumerate(names):
            probes = [
                _gate_probe(opcode, index, superposed=superposed)
                for superposed in (False, True)
            ]
            for theta in (0.37, -1.11):
                parameters = torch.tensor([theta], dtype=torch.float64)
                moved = 0.0
                for circuit, loss in probes:
                    exact = fq.gradient(
                        lambda p, c=circuit, score=loss: score(c(p)), parameters
                    )
                    shifted = fq.gradient(
                        circuit, parameters, loss, method="parameter_shift"
                    )
                    assert torch.allclose(
                        shifted.gradient, exact.gradient, rtol=1e-4, atol=1e-7
                    ), f"{opcode}.{name} at theta={theta}"
                    moved += abs(exact.gradient.item())
                # A parameter that moved nothing would agree for the wrong reason.
                assert moved > 1e-6, f"{opcode}.{name} at theta={theta} moved nothing"
                compared += 1

    assert compared == 34


def test_finite_difference_is_close_but_not_exact() -> None:
    exact = fq.gradient(_value, _PARAMETERS)
    result = fq.gradient(_circuit, _PARAMETERS, _loss, method="finite_difference")

    assert result.exact is False
    assert result.step is not None and result.step > 0.0
    error = _relative_error(result.gradient, exact.gradient)
    assert error < 1e-3
    assert error > 0.0


def test_the_derived_step_follows_the_precision_the_loss_carries() -> None:
    """The default step is measured from the loss dtype, not from the parameters."""

    result = fq.gradient(_circuit, _PARAMETERS, _loss, method="finite_difference")
    float32_step = math.pow(float(torch.finfo(torch.float32).eps), 1.0 / 3.0)

    assert result.step == pytest.approx(float32_step, rel=1e-12)


def test_an_explicit_step_is_reported_back_unchanged() -> None:
    result = fq.gradient(
        _circuit, _PARAMETERS, _loss, method="finite_difference", step=1e-2
    )

    assert result.step == 1e-2


def test_spsa_is_reproducible_for_one_generator_seed() -> None:
    first = fq.gradient(
        _circuit,
        _PARAMETERS,
        _loss,
        method="spsa",
        directions=8,
        generator=torch.Generator().manual_seed(4),
    )
    second = fq.gradient(
        _circuit,
        _PARAMETERS,
        _loss,
        method="spsa",
        directions=8,
        generator=torch.Generator().manual_seed(4),
    )

    assert torch.equal(first.gradient, second.gradient)


def test_spsa_is_not_exact_and_converges_with_the_direction_count() -> None:
    """The estimator's error is reported as an error, then shown to shrink."""

    exact = fq.gradient(_value, _PARAMETERS).gradient
    seed = torch.Generator().manual_seed(0)

    def estimate(directions: int) -> torch.Tensor:
        return fq.gradient(
            _circuit,
            _PARAMETERS,
            _loss,
            method="spsa",
            directions=directions,
            generator=torch.Generator().manual_seed(int(seed.seed())),
        ).gradient

    one = estimate(1)
    many = estimate(512)
    result = fq.gradient(
        _circuit,
        _PARAMETERS,
        _loss,
        method="spsa",
        directions=1,
        generator=torch.Generator().manual_seed(0),
    )

    assert result.exact is False
    assert not torch.allclose(one, exact, rtol=1e-3, atol=1e-3)
    assert (many - exact).abs().max() < (one - exact).abs().max()


@pytest.mark.parametrize("mode", ["statevector", "mps", "tensor_network"])
def test_auto_is_autograd_in_every_differentiable_execution_mode(mode: str) -> None:
    options = fq.ExecutionOptions(mode=mode)

    def value(parameters: torch.Tensor) -> torch.Tensor:
        result = fq.run(
            _circuit(parameters),
            options=options,
            outputs=fq.expectation(fq.Z(0)),
        )
        return result.expectations[0]

    result = fq.gradient(value, _PARAMETERS)
    expected = fq.gradient(_value, _PARAMETERS)

    assert result.method == "autograd"
    assert torch.allclose(result.gradient, expected.gradient, rtol=1e-5, atol=1e-9)


def test_a_sampling_only_mode_fails_closed_through_the_same_call() -> None:
    options = fq.ExecutionOptions(mode="stabilizer")

    def value(parameters: torch.Tensor) -> torch.Tensor:
        result = fq.run(
            _circuit(parameters),
            options=options,
            outputs=fq.expectation(fq.Z(0)),
        )
        return result.expectations[0]

    with pytest.raises(CapabilityError, match="stabilizer"):
        fq.gradient(value, _PARAMETERS)


def test_adjoint_is_refused_by_name_and_points_at_the_supported_routes() -> None:
    with pytest.raises(CapabilityError) as raised:
        fq.gradient(_value, _PARAMETERS, method="adjoint")

    message = str(raised.value)
    assert "adjoint" in message
    assert "autograd" in message
    assert "backward" in message


def test_parameter_shift_without_a_circuit_is_refused_by_name() -> None:
    with pytest.raises(CapabilityError) as raised:
        fq.gradient(_value, _PARAMETERS, method="parameter_shift")

    assert "parameter_shift" in str(raised.value)
    assert "finite_difference" in str(raised.value)


def test_a_program_that_breaks_the_shift_precondition_is_a_capability_error() -> None:
    """A parameter that controls no gate angle is a boundary, not a wrong value."""

    def constant(parameters: torch.Tensor) -> fq.Circuit:
        return fq.Circuit(1).h(0)

    def loss(circuit: fq.Circuit) -> Tensor:
        return circuit.expectation_z(0)

    with pytest.raises(CapabilityError, match="does not apply"):
        fq.gradient(constant, torch.tensor([0.3]), loss, method="parameter_shift")


def test_an_unknown_method_lists_the_accepted_values_and_excludes_adjoint() -> None:
    with pytest.raises(ValidationError) as raised:
        fq.gradient(_value, _PARAMETERS, method="backprop")

    message = str(raised.value)
    for accepted in (
        "auto",
        "autograd",
        "parameter_shift",
        "finite_difference",
        "spsa",
    ):
        assert repr(accepted) in message
    assert "adjoint" not in message


@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param(torch.tensor([], dtype=torch.float64), id="empty"),
        pytest.param(torch.tensor([1, 2]), id="integer"),
        pytest.param(torch.tensor([1.0 + 2.0j]), id="complex"),
        pytest.param(torch.tensor([float("nan")]), id="not_finite"),
    ],
)
def test_an_unusable_parameter_tensor_is_refused(parameters: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValidationError)):
        fq.gradient(_value, parameters)


def test_a_non_tensor_parameter_argument_is_refused() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        fq.gradient(_value, [0.3, 0.7])  # type: ignore[arg-type]


@pytest.mark.parametrize("step", [0.0, -1.0, float("inf"), float("nan")])
def test_an_unusable_step_is_refused(step: float) -> None:
    with pytest.raises(ValidationError, match="step"):
        fq.gradient(_value, _PARAMETERS, method="finite_difference", step=step)


def test_directions_and_generator_are_refused_outside_spsa() -> None:
    with pytest.raises(ValidationError, match="spsa"):
        fq.gradient(_value, _PARAMETERS, method="autograd", directions=2)
    with pytest.raises(ValidationError, match="spsa"):
        fq.gradient(
            _value,
            _PARAMETERS,
            method="finite_difference",
            generator=torch.Generator(),
        )


@pytest.mark.parametrize("directions", [0, -3])
def test_an_unusable_direction_count_is_refused(directions: int) -> None:
    with pytest.raises(ValidationError, match="directions"):
        fq.gradient(_value, _PARAMETERS, method="spsa", directions=directions)


def test_a_non_integer_direction_count_is_refused() -> None:
    with pytest.raises(TypeError, match="directions"):
        fq.gradient(_value, _PARAMETERS, method="spsa", directions=2.0)  # type: ignore[arg-type]


def test_a_program_that_returns_no_scalar_is_refused() -> None:
    def vector(parameters: torch.Tensor) -> torch.Tensor:
        return torch.stack([parameters[0], parameters[1]])

    with pytest.raises(ValidationError, match="one scalar tensor"):
        fq.gradient(vector, torch.tensor([0.3, 0.7], dtype=torch.float64))


def test_gradient_is_a_root_export_and_the_result_type_is_not() -> None:
    """The authorized addition is one name; the return type stays documented."""

    assert "gradient" in fq.__all__
    assert "GradientResult" not in fq.__all__
    assert fq.gradient.__module__ == "flagquantum._api"

    result = fq.gradient(_value, _PARAMETERS)
    assert isinstance(result, GradientResult)
