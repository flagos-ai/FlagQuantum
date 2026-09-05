"""Characterization tests for the first replaceable Simulation Engine slice."""

import pytest
import torch

import flagquantum as fq
from flagquantum.runtime.execution import run_native

pytestmark = pytest.mark.integration


def test_local_statevector_ir_execution_preserves_basis_order_and_batch():
    circuit = fq.Circuit(2, bsz=2, dtype=torch.complex128)
    circuit.h(0).cx(0, 1).rz(1, theta=0.37)

    direct = circuit.state()
    from_ir = run_native(
        circuit.to_ir(),
        mode="statevector",
        bsz=2,
        dtype=torch.complex128,
        device="cpu",
        optimize=False,
    )

    assert direct.shape == (2, 4)
    assert direct.dtype is torch.complex128
    assert torch.allclose(from_ir, direct, atol=1e-12, rtol=1e-12)
    expected_row = torch.tensor(
        [
            2**-0.5 * torch.exp(torch.tensor(-0.185j)),
            0.0,
            0.0,
            2**-0.5 * torch.exp(torch.tensor(0.185j)),
        ],
        dtype=torch.complex128,
    )
    assert torch.allclose(direct, expected_row.expand(2, -1), atol=1e-7, rtol=1e-7)
    assert torch.allclose(
        torch.sum(torch.abs(direct) ** 2, dim=-1),
        torch.ones(2, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_local_statevector_gradient_matches_analytic_rotation_gradient():
    theta = torch.tensor(0.413, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(1, dtype=torch.complex128)
    circuit.ry(0, theta=theta)

    loss = circuit.expectation_z(0).sum()
    loss.backward()

    assert theta.grad is not None
    assert torch.allclose(
        theta.grad,
        -torch.sin(theta.detach()),
        atol=1e-12,
        rtol=1e-12,
    )


def test_local_statevector_custom_matrix_keeps_autograd_graph():
    scale = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    cosine = torch.cos(scale)
    sine = torch.sin(scale)
    matrix = (
        torch.stack((cosine, -sine, sine, cosine)).reshape(2, 2).to(torch.complex128)
    )
    circuit = fq.Circuit(1, dtype=torch.complex128).unitary(0, unitary=matrix)

    value = circuit.state()[0, 1].real
    value.backward()

    assert scale.grad is not None
    assert torch.allclose(
        scale.grad,
        torch.cos(scale.detach()),
        atol=1e-12,
        rtol=1e-12,
    )


def test_dense_expectation_public_facade_uses_statevector_numerics():
    state = torch.tensor([2**-0.5, 2**-0.5], dtype=torch.complex128)
    pauli_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)

    value = fq.expectation((pauli_x, (0,)), ket=state)

    assert torch.allclose(value, torch.ones(1, dtype=torch.complex128))
