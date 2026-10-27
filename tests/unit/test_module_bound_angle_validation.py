"""Refuse a bound parameter value that is not a finite angle.

`Circuit.ry(0, float("nan"))` already refuses with `IRValidationError`, because
the gate constructor checks the value it is given. A `Module` builds its gates
from a builder that is *traced* rather than executed, so the constructor sees a
tracer and defers the magnitude check; the value arrives later, when the reusable
circuit is rebound to concrete parameters. Nothing re-checks it there, and the
run completes with `tensor([nan])`.

These tests pin the refusal to the bind boundary and to the canonical exception,
so every representation inherits it instead of each simulator inventing its own
finite check.
"""

from __future__ import annotations

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import IRValidationError
from flagquantum.errors import ExecutionError

pytestmark = pytest.mark.unit

MODES = ("statevector", "mps", "tensor_network")


def _square_builder(parameters: torch.Tensor) -> fq.Circuit:
    """A nonlinear binding, so an overflowing square is a value-level mistake."""

    return fq.Circuit(1, dtype=torch.complex128).ry(
        0, theta=parameters[0] * parameters[0]
    )


def _module(**options: object) -> fq.Module:
    return fq.Module(
        _square_builder,
        1,
        init=torch.tensor([0.2], dtype=torch.float64),
        dtype=torch.float64,
        policy=fq.RuntimePolicy(execution_options=fq.ExecutionOptions(**options)),
    )


def _fill(module: fq.Module, value: float) -> None:
    with torch.no_grad():
        module.parameters_tensor.fill_(value)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_a_bound_non_finite_angle_is_refused(mode: str, value: float) -> None:
    module = _module(mode=mode)
    module()
    _fill(module, value)

    with pytest.raises(IRValidationError) as raised:
        module()

    message = str(raised.value)
    assert "ry" in message
    assert "finite" in message


def test_an_overflowing_binding_is_refused() -> None:
    """`1e200` is a finite input whose square is not an angle."""

    module = _module()
    module()
    _fill(module, 1e200)

    with pytest.raises(IRValidationError):
        module()


def test_the_refusal_covers_a_cold_module_and_a_warmed_one() -> None:
    """The reusable circuit keeps its bound values live, so both paths must check."""

    cold = _module()
    _fill(cold, float("nan"))
    with pytest.raises(IRValidationError):
        cold()

    warm = _module()
    warm()
    warm()
    _fill(warm, float("nan"))
    with pytest.raises(IRValidationError):
        warm()


def test_the_refusal_reaches_the_execute_path_too() -> None:
    module = _module()
    module.execute()
    _fill(module, float("nan"))

    with pytest.raises((IRValidationError, ExecutionError)) as raised:
        module.execute()

    cause = raised.value
    while cause is not None:
        if isinstance(cause, IRValidationError):
            return
        cause = cause.__cause__
    raise AssertionError("execute did not surface the canonical refusal")


def test_a_direct_circuit_refuses_the_same_way() -> None:
    """The control: the constructor path this fix must match."""

    with pytest.raises(IRValidationError):
        fq.Circuit(1, dtype=torch.complex128).ry(0, theta=float("nan"))


def test_a_finite_parameter_still_runs_and_keeps_its_gradient() -> None:
    module = _module()
    value = module()
    value.backward()

    assert torch.isfinite(value).all()
    assert module.parameters_tensor.grad is not None
    assert torch.isfinite(module.parameters_tensor.grad).all()


def test_a_feature_input_that_is_not_finite_is_refused() -> None:
    """The bound value can arrive through a feature tensor, not only a weight."""

    def builder(parameters: torch.Tensor, features: torch.Tensor) -> fq.Circuit:
        circuit = fq.Circuit(2, dtype=torch.complex128).rx(0, features)
        return circuit.ry(1, parameters[0]).cx(0, 1)

    module = fq.Module(
        builder,
        1,
        init=torch.tensor([0.2], dtype=torch.float64),
        dtype=torch.float64,
    )
    module(torch.tensor(0.3, dtype=torch.float64))

    with pytest.raises((IRValidationError, ExecutionError)):
        module(torch.tensor(float("nan"), dtype=torch.float64))


def test_a_refused_forward_leaves_the_parameter_and_optimizer_untouched() -> None:
    """A loop that guards on the refusal never reaches a step on a poisoned value.

    The parameter is deliberately poisoned here, so the comparison is against the
    state immediately after that poisoning: what the refusal must not do is move
    the parameter or the optimizer slots any further.
    """

    module = _module()
    optimizer = torch.optim.Adam(module.parameters(), lr=0.1)
    module().backward()
    optimizer.step()
    _fill(module, float("nan"))
    poisoned = module.parameters_tensor.detach().clone()
    slot = optimizer.state[module.parameters_tensor]["exp_avg"].detach().clone()

    with pytest.raises(IRValidationError):
        module()

    torch.testing.assert_close(
        module.parameters_tensor.detach(), poisoned, equal_nan=True
    )
    torch.testing.assert_close(
        optimizer.state[module.parameters_tensor]["exp_avg"].detach(), slot
    )

    # A finite value afterwards resumes from those untouched slots.
    with torch.no_grad():
        module.parameters_tensor.fill_(0.2)
    module().backward()
    optimizer.step()
    assert torch.isfinite(module.parameters_tensor).all()


def test_the_jax_backend_refuses_the_same_way() -> None:
    """The check belongs to the bind boundary, so no backend can skip it."""

    pytest.importorskip("jax")
    module = fq.Module(
        _square_builder,
        1,
        init=torch.tensor([0.2], dtype=torch.float64),
        dtype=torch.float64,
        policy=fq.RuntimePolicy(execution_options=fq.ExecutionOptions(backend="jax")),
    )
    module()
    _fill(module, float("nan"))

    with pytest.raises(IRValidationError):
        module()


@pytest.mark.parametrize(
    ("coefficient", "label"),
    [(0.0, "zero observable"), (1.0, "nonzero observable")],
)
def test_a_hamiltonian_objective_refuses_before_it_reports_an_energy(
    coefficient: float, label: str
) -> None:
    """The zero-observable control matters: an optimizer may drop the circuit work.

    With a zero coefficient a simulator can legitimately answer without
    touching the program, which is exactly when a missing refusal would go
    unnoticed. The bound angle is still invalid and must still be refused.
    """

    from flagquantum.algorithms import Hamiltonian, pauli_term

    module = fq.Module(
        _square_builder,
        1,
        init=torch.tensor([0.2], dtype=torch.float64),
        dtype=torch.float64,
        hamiltonian=Hamiltonian([pauli_term(coefficient, "Z", (0,))]),
        policy=fq.RuntimePolicy(
            execution_options=fq.ExecutionOptions(backend="jax"),
            observable="hamiltonian",
        ),
    )
    module()
    _fill(module, float("nan"))

    with pytest.raises(IRValidationError):
        module()


def test_a_normalized_state_is_a_finite_result_the_refusal_does_not_block() -> None:
    """A guard that refuses valid training would be worse than the bug."""

    module = _module()
    for value in (0.0, 1e-8, -1e-8, 3.0, -3.0, 1e8):
        with torch.no_grad():
            module.parameters_tensor.fill_(value)
        assert torch.isfinite(module()).all(), value


def test_a_batched_angle_with_one_bad_element_is_refused() -> None:
    """A declared batch shares one parameter, so one bad element is enough."""

    def builder(parameters: torch.Tensor) -> fq.Circuit:
        circuit = fq.Circuit(2, bsz=3, dtype=torch.complex128)
        for qubit in range(2):
            circuit.ry(qubit, parameters[0])
        return circuit.cx(0, 1)

    module = fq.Module(builder, 1, init=torch.tensor([0.2]), dtype=torch.float64)
    assert module().shape == (3,)
    _fill(module, float("inf"))

    with pytest.raises(IRValidationError):
        module()


def test_one_bad_feature_in_a_feature_batch_is_refused() -> None:
    def builder(parameters: torch.Tensor, features: torch.Tensor) -> fq.Circuit:
        circuit = fq.Circuit(2, bsz=2, dtype=torch.complex128).rx(0, features)
        return circuit.ry(1, parameters[0]).cx(0, 1)

    module = fq.Module(builder, 1, init=torch.tensor([0.2]), dtype=torch.float64)
    module(torch.tensor([0.1, 0.2], dtype=torch.float64))

    with pytest.raises(IRValidationError):
        module(torch.tensor([0.1, float("inf")], dtype=torch.float64))
