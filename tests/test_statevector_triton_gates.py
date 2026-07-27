import pytest
import torch

import flagquantum as fq

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


def test_control_one_pack_unpack_matches_index_reference() -> None:
    _require_cuda()
    from flagquantum.runtime.backends.statevector.triton import (
        pack_complex64_control_one,
        unpack_complex64_control_one,
    )

    torch.manual_seed(29)
    state = torch.randn(2, 64, dtype=torch.complex64, device="cuda")
    output = state.clone()
    bit_position = 3
    compressed_start, compressed_end = 5, 25
    compressed = torch.arange(
        compressed_start, compressed_end, dtype=torch.long, device="cuda"
    )
    low = compressed & ((1 << bit_position) - 1)
    indices = ((compressed - low) << 1) | low | (1 << bit_position)

    packed = pack_complex64_control_one(
        state,
        bit_position=bit_position,
        compressed_start=compressed_start,
        compressed_end=compressed_end,
    )
    torch.testing.assert_close(packed, state[:, indices])

    replacement = 2 * packed
    unpack_complex64_control_one(
        replacement,
        output,
        bit_position=bit_position,
        compressed_start=compressed_start,
    )
    expected = state.clone()
    expected[:, indices] = replacement
    torch.testing.assert_close(output, expected)


def test_constant_ry_rz_triton_path_with_cx_matches_cpu() -> None:
    _require_cuda()
    cpu = fq.Circuit(5, device="cpu", dtype=torch.complex64)
    cuda = fq.Circuit(5, device="cuda", dtype=torch.complex64)
    for circuit in (cpu, cuda):
        for wire in range(5):
            circuit.ry(wire, -0.2 + wire * 0.03).rz(wire, 0.1 - wire * 0.02)
        for wire in range(4):
            circuit.cx(wire, wire + 1)

    expected = cpu.state()
    actual = cuda.state().cpu()

    assert torch.allclose(actual, expected, atol=3e-6, rtol=3e-6)
    assert cuda._last_statevector_runtime["triton_ry_rz_pair_executed"] == 5
    assert len(cuda._statevector_constant_parameters) == 10
    cached_pointers = {
        key: value.data_ptr()
        for key, value in cuda._statevector_constant_parameters.items()
    }
    cuda.state(refresh=True)
    assert {
        key: value.data_ptr()
        for key, value in cuda._statevector_constant_parameters.items()
    } == cached_pointers


def test_generic_constant_single_qubit_regions_preserve_input_gradient() -> None:
    _require_cuda()
    torch.manual_seed(7)
    cpu_input = torch.randn(1, 16, dtype=torch.complex64, requires_grad=True)
    cuda_input = cpu_input.detach().cuda().requires_grad_(True)
    cpu = fq.Circuit(4, device="cpu", dtype=torch.complex64, inputs=cpu_input)
    cuda = fq.Circuit(4, device="cuda", dtype=torch.complex64, inputs=cuda_input)
    for circuit in (cpu, cuda):
        circuit.h(0).x(2).rx(0, 0.31).ry(2, -0.27)
        circuit.t(0).s(2).rz(2, 0.14)

    cpu_state = cpu.state()
    cuda_state = cuda.state()
    cpu_loss = cpu_state.real.square().sum()
    cuda_loss = cuda_state.real.square().sum()
    cpu_loss.backward()
    cuda_loss.backward()

    assert torch.allclose(cuda_state.cpu(), cpu_state, atol=3e-6, rtol=3e-6)
    assert torch.allclose(cuda_input.grad.cpu(), cpu_input.grad, atol=3e-6, rtol=3e-6)
    assert cuda._last_statevector_runtime["triton_single_qubit_matrix_regions"] == 2
    assert (
        cuda._last_statevector_runtime["dependency_reordered_single_qubit_regions"] == 2
    )


def test_parameterized_ry_rz_uses_differentiable_fusion(monkeypatch) -> None:
    _require_cuda()
    monkeypatch.setenv("FQ_TRITON_PARAMETERIZED_SINGLE_QUBIT_MATRIX", "1")
    cpu_angle = torch.tensor(0.23, requires_grad=True)
    cuda_angle = cpu_angle.detach().cuda().requires_grad_(True)
    cpu = fq.Circuit(3, device="cpu", dtype=torch.complex64)
    cuda = fq.Circuit(3, device="cuda", dtype=torch.complex64)
    cpu.ry(0, cpu_angle).rz(0, -0.4 * cpu_angle).cx(0, 1).cx(1, 2)
    cuda.ry(0, cuda_angle).rz(0, -0.4 * cuda_angle).cx(0, 1).cx(1, 2)
    cpu_loss = cpu.state().real.square().sum()
    cuda_loss = cuda.state().real.square().sum()

    cpu_loss.backward()
    cuda_loss.backward()

    assert torch.allclose(cuda_loss.cpu(), cpu_loss, atol=3e-6, rtol=3e-6)
    assert torch.allclose(cuda_angle.grad.cpu(), cpu_angle.grad, atol=3e-6, rtol=3e-6)
    assert cuda._last_statevector_runtime["triton_ry_rz_pair_executed"] == 0
    assert cuda._last_statevector_runtime["triton_single_qubit_matrix_regions"] == 1
    assert cuda._last_statevector_runtime["triton_cx_sequence_regions"] == 1
    assert not cuda._statevector_constant_parameters


def test_interleaved_parameterized_regions_match_all_gradients(monkeypatch) -> None:
    _require_cuda()
    monkeypatch.setenv("FQ_TRITON_PARAMETERIZED_SINGLE_QUBIT_MATRIX", "1")
    torch.manual_seed(19)
    cpu_input = torch.randn(1, 16, dtype=torch.complex64, requires_grad=True)
    cuda_input = cpu_input.detach().cuda().requires_grad_(True)
    cpu_parameters = [
        torch.tensor(value, requires_grad=True)
        for value in (0.13, -0.29, 0.41, 0.07, -0.17, 0.23)
    ]
    cuda_parameters = [
        parameter.detach().cuda().requires_grad_(True) for parameter in cpu_parameters
    ]

    def build(inputs, parameters, device):
        circuit = fq.Circuit(4, device=device, dtype=torch.complex64, inputs=inputs)
        circuit.rx(0, parameters[0]).rx(2, parameters[3])
        circuit.ry(0, parameters[1]).ry(2, parameters[4])
        circuit.rz(0, parameters[2]).rz(2, parameters[5])
        return circuit

    cpu = build(cpu_input, cpu_parameters, "cpu")
    cuda = build(cuda_input, cuda_parameters, "cuda")
    weights = torch.randn(1, 16, dtype=torch.complex64)
    cpu_loss = (cpu.state() * weights).real.sum()
    cuda_loss = (cuda.state() * weights.cuda()).real.sum()
    cpu_loss.backward()
    cuda_loss.backward()

    assert torch.allclose(cuda_loss.cpu(), cpu_loss, atol=3e-6, rtol=3e-6)
    assert torch.allclose(cuda_input.grad.cpu(), cpu_input.grad, atol=3e-6, rtol=3e-6)
    for cuda_parameter, cpu_parameter in zip(
        cuda_parameters, cpu_parameters, strict=True
    ):
        assert torch.allclose(
            cuda_parameter.grad.cpu(), cpu_parameter.grad, atol=3e-6, rtol=3e-6
        )
    assert cuda._last_statevector_runtime["triton_single_qubit_matrix_regions"] == 2
    assert cuda._last_statevector_runtime["batched_rx_ry_rz_regions"] == 2
    assert (
        cuda._last_statevector_runtime["dependency_reordered_single_qubit_regions"] == 2
    )
