"""The contract of the vector-valued derivative entry points.

`fq.gradient` answers "how does one scalar move". These tests state the contract
of the three entry points that answer for a program returning several values:
`fq.jacobian` returns every partial derivative, `fq.jvp` applies the derivative
to a parameter direction, and `fq.vjp` applies it to an output direction.

The tests are written against a real circuit rather than a synthetic function
wherever the shape rules are what is under test, because the failure this family
was designed around -- a reverse seed allocated in the program's own output dtype
instead of the parameter dtype -- only appears once a FlagQuantum executor is
carrying the chain rule. The circuits here are `complex128` so that the analytic
comparisons are limited by the formula and not by the default single precision.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import flagquantum as fq
from flagquantum.errors import CapabilityError, ValidationError
from flagquantum.gradients import jacobian as module_jacobian
from flagquantum.gradients import jvp as module_jvp
from flagquantum.gradients import vjp as module_vjp

pytestmark = pytest.mark.unit

PARAMETERS = torch.tensor([0.3, 0.7, -0.4], dtype=torch.float64)


def _two_qubit_probabilities(parameters: torch.Tensor) -> torch.Tensor:
    """Return the four basis probabilities of a two-qubit circuit."""

    circuit = (
        fq.Circuit(2, dtype=torch.complex128)
        .ry(0, theta=parameters[0])
        .rx(0, theta=parameters[1])
        .ry(1, theta=parameters[2])
        .cx(1, 0)
    )
    return circuit.probabilities()


def _two_expectations(parameters: torch.Tensor) -> torch.Tensor:
    """Return one expectation value per qubit."""

    circuit = (
        fq.Circuit(2, dtype=torch.complex128)
        .ry(0, theta=parameters[0])
        .rx(0, theta=parameters[1])
        .ry(1, theta=parameters[2])
        .cx(1, 0)
    )
    result = fq.run(
        circuit,
        outputs=[fq.expectation(fq.Z(0)), fq.expectation(fq.Z(1))],
    )
    return torch.stack(list(result.expectations))


def _matrix_and_vector(
    matrix: torch.Tensor, parameters: torch.Tensor, output: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a Jacobian and an output for the matrix identities below."""

    return matrix.reshape(output.numel(), parameters.numel()), output.reshape(-1)


def test_the_root_exports_delegate_to_the_gradient_module() -> None:
    """The root name is the facade entry; the algorithm lives in one module."""

    assert fq.jacobian.__module__ == "flagquantum._api"
    assert fq.jvp.__module__ == "flagquantum._api"
    assert fq.vjp.__module__ == "flagquantum._api"
    assert module_jacobian.__module__ == "flagquantum.gradients"
    assert module_jvp.__module__ == "flagquantum.gradients"
    assert module_vjp.__module__ == "flagquantum.gradients"
    assert callable(fq.jacobian) and callable(fq.jvp) and callable(fq.vjp)


def test_the_jacobian_has_one_axis_per_parameter_element() -> None:
    output = _two_expectations(PARAMETERS)
    result = fq.jacobian(_two_expectations, PARAMETERS)

    assert result.shape == (*output.shape, *PARAMETERS.shape)
    assert result.dtype == PARAMETERS.dtype
    assert result.device == PARAMETERS.device


def test_the_jacobian_keeps_a_multi_axis_parameter_shape() -> None:
    """A caller whose parameters are a matrix gets that matrix back as an axis."""

    grid = torch.tensor([[0.3, 0.7], [-0.4, 0.1]], dtype=torch.float64)

    def program(parameters: torch.Tensor) -> torch.Tensor:
        circuit = (
            fq.Circuit(2, dtype=torch.complex128)
            .ry(0, theta=parameters[0, 0])
            .rx(0, theta=parameters[0, 1])
            .ry(1, theta=parameters[1, 0])
            .cx(1, 0)
        )
        return fq.run(circuit, outputs=fq.expectation(fq.Z(0))).expectations[0]

    output = program(grid)
    result = fq.jacobian(program, grid)

    assert result.shape == (*output.shape, *grid.shape)


def test_the_jacobian_of_a_probability_vector_is_exact() -> None:
    """Every entry is checked against an independent central difference."""

    result = fq.jacobian(_two_qubit_probabilities, PARAMETERS)
    rows = result.reshape(-1, PARAMETERS.numel())
    step = 1e-5
    independent = torch.zeros_like(rows)
    for row in range(rows.shape[0]):
        for column in range(PARAMETERS.numel()):
            shifted = []
            for sign in (1.0, -1.0):
                moved = PARAMETERS.clone()
                moved[column] += sign * step
                shifted.append(float(_two_qubit_probabilities(moved)[0, row]))
            independent[row, column] = (shifted[0] - shifted[1]) / (2.0 * step)

    assert torch.allclose(rows, independent, rtol=1e-7, atol=1e-10)
    # Every entry moving is what makes the agreement above meaningful: an
    # all-zero Jacobian would also match a difference of an all-zero program.
    assert float(rows.abs().min()) > 1e-3


def test_the_jacobian_rows_are_the_scalar_gradients_of_each_output() -> None:
    """The vector entry point and the scalar entry point must not disagree."""

    output = _two_expectations(PARAMETERS)
    reference = fq.jacobian(_two_expectations, PARAMETERS)
    rows = []

    def output_element(position: int) -> Callable[[torch.Tensor], torch.Tensor]:
        def program(parameters: torch.Tensor) -> torch.Tensor:
            return _two_expectations(parameters)[position]

        return program

    for position in range(output.numel()):
        scalar = fq.gradient(output_element(position), PARAMETERS)
        assert scalar.method == "autograd"
        rows.append(scalar.gradient)

    assert torch.allclose(
        reference.reshape(output.numel(), PARAMETERS.numel()), torch.stack(rows)
    )


def test_the_jacobian_matches_the_closed_form_on_one_qubit() -> None:
    """A route that agreed with itself could still be wrong, so check analysis."""

    theta = torch.tensor([0.3], dtype=torch.float64)

    def probabilities(parameters: torch.Tensor) -> torch.Tensor:
        circuit = fq.Circuit(1, dtype=torch.complex128).ry(0, theta=parameters[0])
        return circuit.probabilities()

    # The probabilities of Ry(t)|0> are cos^2(t/2) and sin^2(t/2), whose
    # derivatives are -sin(t)/2 and +sin(t)/2.
    expected = torch.stack([-torch.sin(theta) / 2.0, torch.sin(theta) / 2.0]).reshape(
        2, 1
    )
    result = fq.jacobian(probabilities, theta)

    assert result.shape == (1, 2, 1)
    assert torch.allclose(result[0], expected, rtol=1e-12, atol=1e-15)
    assert float(expected.abs().min()) > 0.1


def test_the_jacobian_vector_product_is_the_jacobian_applied_to_the_direction() -> None:
    tangent = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    output = _two_expectations(PARAMETERS)
    matrix, _ = _matrix_and_vector(
        fq.jacobian(_two_expectations, PARAMETERS), PARAMETERS, output
    )

    product = fq.jvp(_two_expectations, PARAMETERS, tangent)

    assert product.shape == output.shape
    assert torch.allclose(product.reshape(-1), matrix @ tangent, rtol=1e-10, atol=1e-13)


def test_the_vector_jacobian_product_is_the_cotangent_applied_to_the_jacobian() -> None:
    output = _two_expectations(PARAMETERS)
    cotangent = torch.tensor([0.25, -1.5], dtype=torch.float64).reshape(output.shape)
    matrix, _ = _matrix_and_vector(
        fq.jacobian(_two_expectations, PARAMETERS), PARAMETERS, output
    )

    product = fq.vjp(_two_expectations, PARAMETERS, cotangent)

    assert product.shape == PARAMETERS.shape
    expected = matrix.T @ cotangent.reshape(-1)
    assert torch.allclose(
        product, expected.reshape_as(PARAMETERS), rtol=1e-12, atol=1e-15
    )


def test_the_two_products_are_adjoint() -> None:
    """<Jv, c> must equal <v, J^T c> for every direction pair."""

    output = _two_expectations(PARAMETERS)
    tangent = torch.tensor([0.4, 1.1, -0.9], dtype=torch.float64)
    cotangent = torch.tensor([-0.6, 2.0], dtype=torch.float64).reshape(output.shape)

    forward = float((fq.jvp(_two_expectations, PARAMETERS, tangent) * cotangent).sum())
    reverse = float((fq.vjp(_two_expectations, PARAMETERS, cotangent) * tangent).sum())

    assert forward == pytest.approx(reverse, rel=1e-10, abs=1e-13)
    assert abs(forward) > 1e-3


def test_the_products_match_a_finite_difference_of_the_scalar_objective() -> None:
    """The adjoint identity is also checked against an outside measurement."""

    output = _two_expectations(PARAMETERS)
    tangent = torch.tensor([0.4, 1.1, -0.9], dtype=torch.float64)
    cotangent = torch.tensor([-0.6, 2.0], dtype=torch.float64).reshape(output.shape)

    def objective(parameters: torch.Tensor) -> float:
        return float((_two_expectations(parameters) * cotangent).sum())

    step = 1e-5
    plus = objective(PARAMETERS + step * tangent)
    minus = objective(PARAMETERS - step * tangent)
    measured = (plus - minus) / (2.0 * step)

    reverse = float((fq.vjp(_two_expectations, PARAMETERS, cotangent) * tangent).sum())

    assert reverse == pytest.approx(measured, rel=1e-8, abs=1e-11)


@pytest.mark.parametrize("entry_point", ["jacobian", "jvp", "vjp"])
def test_every_entry_point_evaluates_the_program_once(entry_point: str) -> None:
    """One forward pass serves the derivative, so the cost is in the sweeps."""

    calls = []

    def program(parameters: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return _two_expectations(parameters)

    output = _two_expectations(PARAMETERS)
    if entry_point == "jacobian":
        fq.jacobian(program, PARAMETERS)
    elif entry_point == "jvp":
        fq.jvp(program, PARAMETERS, torch.ones(3, dtype=torch.float64))
    else:
        fq.vjp(program, PARAMETERS, torch.ones_like(output))

    assert len(calls) == 1


@pytest.mark.parametrize("entry_point", ["jacobian", "jvp", "vjp"])
def test_every_result_is_detached(entry_point: str) -> None:
    """A training loop reuses the value, so it must not extend the tape."""

    output = _two_expectations(PARAMETERS)
    if entry_point == "jacobian":
        result = fq.jacobian(_two_expectations, PARAMETERS)
    elif entry_point == "jvp":
        result = fq.jvp(
            _two_expectations, PARAMETERS, torch.ones(3, dtype=torch.float64)
        )
    else:
        result = fq.vjp(_two_expectations, PARAMETERS, torch.ones_like(output))

    assert result.requires_grad is False
    assert result.grad_fn is None
    assert PARAMETERS.requires_grad is False


def test_a_jacobian_cannot_be_differentiated_a_second_time() -> None:
    """Detachment is deliberate, so nesting fails closed instead of half-working."""

    with pytest.raises(CapabilityError, match="does not depend on the supplied"):
        fq.jacobian(
            lambda parameters: fq.jacobian(_two_expectations, parameters), PARAMETERS
        )


@pytest.mark.parametrize(
    ("circuit_dtype", "parameter_dtype"),
    [
        (None, torch.float32),
        (None, torch.float64),
        (torch.complex64, torch.float32),
        (torch.complex64, torch.float64),
        (torch.complex128, torch.float32),
        (torch.complex128, torch.float64),
    ],
)
def test_the_result_dtype_follows_the_parameters_not_the_circuit(
    circuit_dtype: torch.dtype | None, parameter_dtype: torch.dtype
) -> None:
    """The reverse seed must not decide the dtype of the answer.

    A seed allocated in the program's own output dtype raises a dtype mismatch
    inside the executor whenever the amplitudes are narrower than the
    parameters -- the default circuit is `complex64` while parameters are
    commonly `float64`. Only a seed in the parameter dtype passes every
    combination, and it is also what makes the result's dtype predictable.
    """

    parameters = PARAMETERS.to(parameter_dtype)

    def program(values: torch.Tensor) -> torch.Tensor:
        circuit = (
            fq.Circuit(2, dtype=circuit_dtype)
            .ry(0, theta=values[0])
            .rx(0, theta=values[1])
            .ry(1, theta=values[2])
            .cx(1, 0)
        )
        result = fq.run(
            circuit,
            outputs=[fq.expectation(fq.Z(0)), fq.expectation(fq.Z(1))],
        )
        return torch.stack(list(result.expectations))

    output = program(parameters)
    jacobian = fq.jacobian(program, parameters)
    forward = fq.jvp(program, parameters, torch.ones(3, dtype=parameter_dtype))
    reverse = fq.vjp(program, parameters, torch.ones_like(output))

    assert jacobian.dtype == parameter_dtype
    assert forward.dtype == parameter_dtype
    assert reverse.dtype == parameter_dtype
    assert torch.allclose(
        forward.reshape(-1),
        jacobian.reshape(-1, 3) @ torch.ones(3, dtype=parameter_dtype),
        rtol=1e-4,
        atol=1e-6,
    )


@pytest.mark.parametrize("entry_point", ["jacobian", "jvp", "vjp"])
def test_a_program_that_ignores_the_parameters_is_refused(entry_point: str) -> None:
    """A zero Jacobian would hide a program that never read the parameters."""

    def constant(parameters: torch.Tensor) -> torch.Tensor:
        return torch.ones(2, dtype=torch.float64)

    with pytest.raises(CapabilityError, match="does not depend on the supplied"):
        if entry_point == "jacobian":
            fq.jacobian(constant, PARAMETERS)
        elif entry_point == "jvp":
            fq.jvp(constant, PARAMETERS, torch.ones(3, dtype=torch.float64))
        else:
            fq.vjp(constant, PARAMETERS, torch.ones(2, dtype=torch.float64))


def test_a_complex_program_output_is_refused_with_its_reason() -> None:
    """No entry point defines a Wirtinger convention, so none may guess one."""

    def amplitudes(parameters: torch.Tensor) -> torch.Tensor:
        return fq.Circuit(1).ry(0, theta=parameters[0]).state()

    with pytest.raises(ValidationError, match="Wirtinger"):
        fq.jacobian(amplitudes, PARAMETERS[:1])


def test_a_direction_that_only_broadcasts_is_refused() -> None:
    """Broadcasting would differentiate along a different parameterization."""

    output = _two_expectations(PARAMETERS)

    with pytest.raises(ValidationError, match="tangents must be shaped like"):
        fq.jvp(_two_expectations, PARAMETERS, torch.ones(1, dtype=torch.float64))
    with pytest.raises(ValidationError, match="cotangents must be shaped like"):
        fq.vjp(_two_expectations, PARAMETERS, torch.ones(1, dtype=torch.float64))

    assert output.shape == (2, 1)
    with pytest.raises(ValidationError, match="cotangents must be shaped like"):
        fq.vjp(
            _two_expectations,
            PARAMETERS,
            torch.ones(output.numel(), dtype=torch.float64),
        )


@pytest.mark.parametrize(
    ("direction", "message"),
    [
        (torch.ones(3, dtype=torch.int64), "real floating-point"),
        (torch.ones(3, dtype=torch.complex64), "real floating-point"),
        (torch.tensor([1.0, float("nan"), 1.0]), "finite"),
        (torch.tensor([1.0, float("inf"), 1.0]), "finite"),
    ],
)
def test_a_direction_that_is_not_a_real_finite_vector_is_refused(
    direction: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        fq.jvp(_two_expectations, PARAMETERS, direction)


def test_a_non_tensor_output_is_refused() -> None:
    with pytest.raises(ValidationError, match="must return a torch.Tensor"):
        fq.jacobian(lambda parameters: 1.0, PARAMETERS)


def test_an_empty_output_is_refused() -> None:
    with pytest.raises(ValidationError, match="non-empty tensor"):
        fq.jacobian(lambda parameters: torch.zeros(0), PARAMETERS)


def test_the_shared_parameter_rules_still_apply() -> None:
    with pytest.raises(TypeError, match="parameters must be a torch.Tensor"):
        fq.jacobian(_two_expectations, [0.3, 0.7, -0.4])  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="parameters must not be empty"):
        fq.jacobian(_two_expectations, torch.zeros(0))
    with pytest.raises(ValidationError, match="parameters must be real"):
        fq.jacobian(_two_expectations, torch.ones(3, dtype=torch.complex64))
    with pytest.raises(ValidationError, match="parameters must be finite"):
        fq.jacobian(_two_expectations, torch.tensor([1.0, float("nan"), 1.0]))


def test_a_non_tensor_direction_is_refused() -> None:
    with pytest.raises(TypeError, match="tangents must be a torch.Tensor"):
        fq.jvp(_two_expectations, PARAMETERS, [1.0, 1.0, 1.0])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="cotangents must be a torch.Tensor"):
        fq.vjp(_two_expectations, PARAMETERS, [1.0, 1.0])  # type: ignore[arg-type]
