"""One central difference, and one step policy, for both of its callers.

Two surfaces need a numerical derivative: `fq.gradient` when a program has no
analytically differentiable route, and the P5 CPU conformance oracle that checks
the parameter-shift route against an independent method. They were two copies of
the same quotient with different signatures. These tests hold the single
implementation in place *and* hold the behaviour a user observes, so a future
edit cannot silently change the derivative the public route reports or let the
oracle drift away from it.
"""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum as fq
from flagquantum.core.finite_differences import (
    central_difference_gradient,
    default_difference_step,
)
from flagquantum.errors import ValidationError

pytestmark = pytest.mark.unit

_PARAMETERS = torch.tensor([0.3, 0.7, -0.4], dtype=torch.float64)


def _circuit(parameters: torch.Tensor) -> fq.Circuit:
    return (
        fq.Circuit(2)
        .ry(0, theta=parameters[0])
        .rx(0, theta=parameters[1])
        .ry(1, theta=parameters[2])
        .cx(1, 0)
    )


def _value(parameters: torch.Tensor) -> torch.Tensor:
    return _circuit(parameters).expectation_z(0)


def test_the_default_displacement_is_derived_from_the_delivered_precision() -> None:
    """The step balances truncation against the roundoff of the arithmetic used.

    The dtype a caller's parameter tensor carries is not necessarily the dtype
    the program computes in, so the step has to come from a delivered sample.
    """

    for dtype in (torch.float32, torch.float64):
        sample = torch.zeros(1, dtype=dtype)
        expected = math.pow(float(torch.finfo(dtype).eps), 1.0 / 3.0)
        assert default_difference_step(sample) == expected

    # A narrower precision needs a wider step, which is the whole point of
    # measuring the delivered sample instead of assuming one.
    assert default_difference_step(torch.zeros(1, dtype=torch.float32)) > (
        default_difference_step(torch.zeros(1, dtype=torch.float64))
    )


def test_the_reported_step_is_the_displacement_the_derivative_used() -> None:
    """`result.step` is published evidence, so it must describe the real quotient."""

    result = fq.gradient(_value, _PARAMETERS, method="finite_difference")

    assert result.method == "finite_difference"
    assert result.exact is False
    assert result.step == default_difference_step(_value(_PARAMETERS))
    # The quotient returns the dtype its own arithmetic produced; the public
    # route restores the parameter dtype and device it was handed. Recomputed at
    # the reported step, the two must agree exactly, not merely closely.
    assert torch.equal(
        result.gradient,
        central_difference_gradient(_value, _PARAMETERS, step=result.step)
        .reshape_as(_PARAMETERS)
        .to(device=_PARAMETERS.device, dtype=_PARAMETERS.dtype),
    )


def test_a_caller_supplied_step_is_the_one_used() -> None:
    """An explicit `step=` must move the derivative, not just the reported number."""

    coarse = fq.gradient(_value, _PARAMETERS, method="finite_difference", step=1e-1)
    fine = fq.gradient(_value, _PARAMETERS, method="finite_difference", step=1e-5)

    assert coarse.step == 1e-1
    assert fine.step == 1e-5
    assert not torch.allclose(coarse.gradient, fine.gradient)


def test_the_input_shape_dtype_and_device_survive_the_round_trip() -> None:
    """Every caller owes its own caller the shape it was handed back."""

    for dtype in (torch.float32, torch.float64):
        parameters = _PARAMETERS.to(dtype)
        result = fq.gradient(_value, parameters, method="finite_difference")

        assert result.gradient.shape == parameters.shape
        assert result.gradient.dtype == parameters.dtype
        assert result.gradient.device == parameters.device


@pytest.mark.parametrize("step", [0.0, -1.0, float("inf"), float("nan")])
def test_a_displacement_that_cannot_be_divided_by_is_refused(step: float) -> None:
    """The quotient raises before it divides, so no caller can produce a nan silently."""

    with pytest.raises(ValueError, match="positive finite"):
        central_difference_gradient(_value, _PARAMETERS, step=step)


@pytest.mark.parametrize("step", [0.0, -1.0, float("inf"), float("nan")])
def test_the_public_route_refuses_an_unusable_displacement_before_the_quotient(
    step: float,
) -> None:
    """The public entry point owns its documented error type, not ValueError."""

    with pytest.raises(ValidationError, match="step"):
        fq.gradient(_value, _PARAMETERS, method="finite_difference", step=step)


def test_the_p5_conformance_oracle_is_this_same_difference() -> None:
    """The oracle that checks the exact route must not be a second quotient.

    The two implementations this file replaced differed only in signature while
    sharing the formula and the denominator, which is exactly the drift this
    assertion makes expensive: reintroducing a local quotient with its own
    arithmetic fails here rather than only in a numerical tolerances report.
    """

    from flagquantum.runtime.executors.statevector.split_real_imag import (
        _complex128_pauli_expectation,
        _normalized_observables,
        _parameter_occurrences,
        _training_conformance_ir,
        _training_conformance_observable,
    )
    from flagquantum.runtime.executors.statevector.split_real_imag_autograd_conformance import (
        _finite_difference_gradient,
    )
    from flagquantum.runtime.executors.statevector.split_real_imag_device_double_single_conformance import (
        _p4_reference_ir,
    )
    from flagquantum.runtime.executors.statevector.split_real_imag_double_single import (
        _bind_p3_ir,
        _normalized_p3_bindings,
    )

    epsilon = 1e-3
    ir = _p4_reference_ir(_training_conformance_ir(8, 0))
    terms = _normalized_observables(ir, _training_conformance_observable())
    bindings = _normalized_p3_bindings(
        {
            "alpha": torch.tensor(0.17, dtype=torch.float32, requires_grad=True),
            "beta": torch.tensor(-0.29, dtype=torch.float32, requires_grad=True),
            "gamma": torch.tensor(0.11, dtype=torch.float32, requires_grad=True),
        }
    )
    order = tuple(sorted(_parameter_occurrences(ir)))
    assert set(order) == set(bindings)

    expected = central_difference_gradient(
        lambda flat: _complex128_pauli_expectation(
            _bind_p3_ir(
                ir,
                {
                    **{name: bindings[name] for name in bindings if name not in order},
                    **{name: flat[index] for index, name in enumerate(order)},
                },
            ),
            terms,
        ),
        torch.stack([bindings[name] for name in order]),
        step=epsilon,
    )
    actual = _finite_difference_gradient(ir, terms, bindings, order, epsilon=epsilon)

    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)
