from __future__ import annotations

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler._hybrid import (
    capture_source,
    specialize_and_lower,
    tensor_type,
)
from flagquantum.runtime.executors.statevector.reverse import (
    execute_torch_distributed_statevector_reverse,
)

pytestmark = pytest.mark.integration

WEIGHTS = tensor_type("float64", (None, 4))
DATA = tensor_type("float64", (4,))

SOURCE = """
def cost(weights, data):
    qp.AngleEmbedding(data, wires=range(4))
    for row in weights:
        for wire, parameter in enumerate(row):
            if parameter > 0:
                qp.RX(parameter, wires=wire)
            elif parameter < 0:
                qp.RY(parameter, wires=wire)
        for wire in range(4):
            qp.CNOT(wires=[wire, jnp.mod(wire + 1, 4)])
    return qp.expval(qp.PauliZ(0) + qp.PauliZ(3))
"""

PROGRAM = capture_source(SOURCE, (WEIGHTS, DATA))


def _bound(weights: torch.Tensor, data: torch.Tensor):
    return specialize_and_lower(
        PROGRAM,
        (weights, data),
        circuit_dtype="complex128",
        require_smooth_gradients=True,
    ).bind()


def _hybrid_vjp(weights: torch.Tensor, data: torch.Tensor):
    ir = _bound(weights, data)
    assert all(term.name == "z" and term.coefficient == 1.0 for term in ir.observables)
    return execute_torch_distributed_statevector_reverse(
        ir,
        observable_wires=tuple(term.wires[0] for term in ir.observables),
        device="cpu",
    )


def _reference_value(weights: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    circuit = fq.Circuit(4, dtype=torch.complex128)
    for wire, parameter in enumerate(data):
        circuit.rx(wire, theta=parameter)
    for row in weights:
        for wire, parameter in enumerate(row):
            if bool(parameter > 0):
                circuit.rx(wire, theta=parameter)
            elif bool(parameter < 0):
                circuit.ry(wire, theta=parameter)
        for wire in range(4):
            circuit.cx(wire, (wire + 1) % 4)
    return (circuit.expectation_z(0) + circuit.expectation_z(3)).sum()


def _compiled_forward(weights: torch.Tensor, data: torch.Tensor) -> float:
    result = fq.run(
        _bound(weights, data),
        options=fq.ExecutionOptions(
            mode="statevector",
            device="cpu",
            precision="complex128",
            require_gradients=False,
            allow_approximate=False,
        ),
    )
    return float(result.expectation().detach())


def _central_difference(
    weights: torch.Tensor,
    data: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_gradient = torch.empty_like(weights)
    data_gradient = torch.empty_like(data)
    for index in range(weights.numel()):
        positive = weights.detach().clone()
        negative = weights.detach().clone()
        positive.view(-1)[index] += epsilon
        negative.view(-1)[index] -= epsilon
        weight_gradient.view(-1)[index] = (
            _compiled_forward(positive, data) - _compiled_forward(negative, data)
        ) / (2 * epsilon)
    for index in range(data.numel()):
        positive = data.detach().clone()
        negative = data.detach().clone()
        positive[index] += epsilon
        negative[index] -= epsilon
        data_gradient[index] = (
            _compiled_forward(weights, positive) - _compiled_forward(weights, negative)
        ) / (2 * epsilon)
    return weight_gradient, data_gradient


@pytest.mark.parametrize(
    "values",
    (
        (0.2, 0.3, 0.4, 0.5),
        (-0.2, -0.3, -0.4, -0.5),
        (0.2, -0.3, 0.4, -0.5),
    ),
)
def test_branchwise_adjoint_vjp_matches_dense_autograd_and_finite_difference(
    values: tuple[float, ...],
) -> None:
    weights = torch.tensor([values], dtype=torch.float64, requires_grad=True)
    data = torch.tensor(
        [0.11, -0.22, 0.33, -0.44],
        dtype=torch.float64,
        requires_grad=True,
    )
    reference_weights = weights.detach().clone().requires_grad_(True)
    reference_data = data.detach().clone().requires_grad_(True)
    reference = _reference_value(reference_weights, reference_data)
    expected = torch.autograd.grad(reference, (reference_weights, reference_data))
    finite_difference = _central_difference(weights.detach(), data.detach())

    result = _hybrid_vjp(weights, data)
    result.backward()

    torch.testing.assert_close(result.value, reference.detach(), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(weights.grad, expected[0], atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(data.grad, expected[1], atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(weights.grad, finite_difference[0], atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(data.grad, finite_difference[1], atol=2e-7, rtol=2e-6)
    summary = result.summary()
    assert summary["gradient_method"] == "statevector_adjoint"
    assert summary["observable_wires"] == (0, 3)
    assert summary["backward_distribution_semantics"] == "single_device_fast_path"
    assert summary["backward_uses_full_state_replay"] is False


def test_seeded_adjoint_optimization_decreases_objective() -> None:
    weights = torch.tensor(
        [[0.25, -0.35, 0.45, -0.55]],
        dtype=torch.float64,
        requires_grad=True,
    )
    data = torch.tensor(
        [0.12, -0.24, 0.36, -0.48],
        dtype=torch.float64,
        requires_grad=True,
    )
    losses: list[float] = []

    for _ in range(6):
        result = _hybrid_vjp(weights, data)
        loss = (result.value - 0.25).square()
        losses.append(float(loss.detach()))
        loss.backward()
        with torch.no_grad():
            weights -= 0.01 * weights.grad
            data -= 0.01 * data.grad
            weights.grad = None
            data.grad = None

    assert losses[-1] < losses[0]
    assert all(after < before for before, after in zip(losses, losses[1:]))
