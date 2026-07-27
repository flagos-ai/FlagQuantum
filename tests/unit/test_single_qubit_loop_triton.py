import pytest
import torch

pytest.importorskip("triton")

import flagquantum as fq
from flagquantum.simulation.triton_kernels import (
    repeated_rx_rz,
    repeated_rx_rz_tangents,
)


def _reference(state, rx_angles, rz_angles):
    output = state
    for layer in range(int(rx_angles.shape[1])):
        rx = rx_angles[:, layer].reshape(-1, 1, 1) / 2
        rz = rz_angles[:, layer].reshape(-1, 1, 1) / 2
        cos_rx, sin_rx = torch.cos(rx), torch.sin(rx)
        zero, one = output[..., :1], output[..., 1:]
        output = torch.cat(
            (cos_rx * zero - 1j * sin_rx * one, -1j * sin_rx * zero + cos_rx * one),
            dim=-1,
        )
        output = output * torch.cat((torch.exp(-1j * rz), torch.exp(1j * rz)), dim=-1)
    return output


def test_repeated_rx_rz_cpu_fallback_matches_reference() -> None:
    torch.manual_seed(5)
    state = torch.randn(2, 7, 2, dtype=torch.complex64)
    rx = torch.randn(2, 5)
    rz = torch.randn(2, 5)

    torch.testing.assert_close(repeated_rx_rz(state, rx, rz), _reference(state, rx, rz))


def test_repeated_rx_rz_cpu_fallback_gradients_match_reference() -> None:
    torch.manual_seed(5)
    state = torch.randn(2, 7, 2, dtype=torch.complex64, requires_grad=True)
    rx = torch.randn(2, 5, requires_grad=True)
    rz = torch.randn(2, 5, requires_grad=True)
    reference_state = state.detach().clone().requires_grad_(True)
    reference_rx = rx.detach().clone().requires_grad_(True)
    reference_rz = rz.detach().clone().requires_grad_(True)
    gradient = torch.randn_like(state)

    actual_gradients = torch.autograd.grad(
        repeated_rx_rz(state, rx, rz), (state, rx, rz), gradient
    )
    reference_gradients = torch.autograd.grad(
        _reference(reference_state, reference_rx, reference_rz),
        (reference_state, reference_rx, reference_rz),
        gradient,
    )

    for actual, reference in zip(actual_gradients, reference_gradients):
        torch.testing.assert_close(actual, reference)


def test_repeated_rx_rz_cpu_tangents_match_jacobian() -> None:
    torch.manual_seed(11)
    state = torch.randn(2, 5, 2, dtype=torch.complex64)
    rx = torch.randn(2, 3)
    rz = torch.randn(2, 3)
    actual = repeated_rx_rz_tangents(state, rx, rz)
    expected = []
    for layer in range(3):
        for family, parameter in (("rx", rx), ("rz", rz)):

            def selected_output(value):
                return _reference(
                    state,
                    value if family == "rx" else rx,
                    value if family == "rz" else rz,
                )

            real_jacobian = torch.autograd.functional.jacobian(
                lambda value: selected_output(value).real,
                parameter,
                vectorize=True,
            )
            imag_jacobian = torch.autograd.functional.jacobian(
                lambda value: selected_output(value).imag,
                parameter,
                vectorize=True,
            )
            jacobian = torch.complex(real_jacobian, imag_jacobian)
            indices = torch.arange(2)
            expected.append(jacobian[indices, :, :, indices, layer])
    torch.testing.assert_close(actual, torch.stack(expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("depth", [1, 4, 16])
def test_repeated_rx_rz_cuda_tangents_match_cpu_reference(depth: int) -> None:
    torch.manual_seed(13)
    state = torch.randn(2, 513, 2, dtype=torch.complex64)
    rx = torch.randn(2, depth)
    rz = torch.randn(2, depth)
    expected = repeated_rx_rz_tangents(state, rx, rz)
    actual = repeated_rx_rz_tangents(state.cuda(), rx.cuda(), rz.cuda()).cpu()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("depth", [1, 4, 16, 64])
def test_repeated_rx_rz_cuda_matches_reference(depth: int) -> None:
    torch.manual_seed(5)
    state = torch.randn(
        3, 513, 2, device="cuda", dtype=torch.complex64, requires_grad=True
    )
    rx = torch.randn(3, depth, device="cuda", requires_grad=True)
    rz = torch.randn(3, depth, device="cuda", requires_grad=True)
    reference_state = state.detach().clone().requires_grad_(True)
    reference_rx = rx.detach().clone().requires_grad_(True)
    reference_rz = rz.detach().clone().requires_grad_(True)

    actual = repeated_rx_rz(state, rx, rz)
    reference = _reference(reference_state, reference_rx, reference_rz)
    gradient = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, (state, rx, rz), gradient)
    reference_gradients = torch.autograd.grad(
        reference,
        (reference_state, reference_rx, reference_rz),
        gradient,
    )

    torch.testing.assert_close(actual, reference, atol=5e-6, rtol=5e-6)
    for actual_gradient, reference_gradient in zip(
        actual_gradients, reference_gradients
    ):
        torch.testing.assert_close(
            actual_gradient, reference_gradient, atol=2e-4, rtol=2e-4
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_circuit_ir_rx_rz_fusion_matches_eager_state_and_vqe_gradients(
    monkeypatch,
) -> None:
    torch.manual_seed(9)
    parameters = torch.randn(3, 4, 2, device="cuda", requires_grad=True)
    reference_parameters = parameters.detach().clone().requires_grad_(True)

    def build(values):
        circuit = fq.Circuit(3, device="cuda")
        circuit.h(0).h(1).h(2)
        for layer in range(4):
            for wire in range(3):
                circuit.rx(wire, theta=values[wire, layer, 0])
                circuit.rz(wire, theta=values[wire, layer, 1])
        circuit.cx(0, 1).cx(1, 2)
        return circuit

    monkeypatch.setenv("FQ_TRITON_SINGLE_QUBIT_LOOP", "1")
    circuit = build(parameters)
    actual = circuit.state(refresh=True)
    actual_loss = fq.zz_chain_hamiltonian(3).expectation(circuit).sum()
    actual_gradient = torch.autograd.grad(actual_loss, parameters)[0]

    monkeypatch.setenv("FQ_TRITON_SINGLE_QUBIT_LOOP", "0")
    reference_circuit = build(reference_parameters)
    reference = reference_circuit.state(refresh=True)
    reference_loss = fq.zz_chain_hamiltonian(3).expectation(reference_circuit).sum()
    reference_gradient = torch.autograd.grad(reference_loss, reference_parameters)[0]

    torch.testing.assert_close(actual, reference, atol=5e-6, rtol=5e-6)
    torch.testing.assert_close(actual_loss, reference_loss, atol=5e-6, rtol=5e-6)
    torch.testing.assert_close(
        actual_gradient, reference_gradient, atol=2e-4, rtol=2e-4
    )
