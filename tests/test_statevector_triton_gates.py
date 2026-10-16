import pytest
import torch

import flagquantum as fq
import flagquantum.simulation.statevector.cx_sequence_dispatch as cx_sequence_dispatch
import flagquantum.simulation.statevector.ry_rz_dispatch as ry_rz_dispatch
import flagquantum.simulation.statevector.single_qubit_matrix_dispatch as single_qubit_matrix_dispatch
import flagquantum.simulation.statevector.two_qubit_matrix_dispatch as two_qubit_matrix_dispatch
from flagquantum.simulation.statevector.operations import _apply_matrix

pytestmark = [pytest.mark.gpu, pytest.mark.integration, pytest.mark.triton]


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


def _reference_local_2q(
    state: torch.Tensor,
    matrix: torch.Tensor,
    first_bit_position: int,
    second_bit_position: int,
) -> torch.Tensor:
    n_qubits = state.shape[1].bit_length() - 1
    wires = (
        n_qubits - 1 - first_bit_position,
        n_qubits - 1 - second_bit_position,
    )
    rest = tuple(qubit for qubit in range(n_qubits) if qubit not in wires)
    permutation = (0, wires[0] + 1, wires[1] + 1) + tuple(qubit + 1 for qubit in rest)
    inverse = [0] * len(permutation)
    for index, axis in enumerate(permutation):
        inverse[axis] = index
    tensor = state.reshape((state.shape[0],) + (2,) * n_qubits).permute(permutation)
    flat = tensor.reshape(state.shape[0], 4, -1)
    result = torch.matmul(matrix, flat)
    return (
        result.reshape((state.shape[0],) + (2,) * n_qubits)
        .permute(tuple(inverse))
        .reshape_as(state)
    )


def test_generic_local_2q_matches_layout_reference_and_exact_alias() -> None:
    _require_cuda()
    from flagquantum.kernels.triton.statevector_gates import apply_complex64_local_2q

    torch.manual_seed(43)
    state = torch.randn(2, 64, dtype=torch.complex64, device="cuda")
    matrix = torch.randn(4, 4, dtype=torch.complex64, device="cuda")
    for first_bit_position, second_bit_position in (
        (0, 1),
        (1, 0),
        (0, 5),
        (5, 0),
        (1, 4),
        (4, 2),
    ):
        expected = _reference_local_2q(
            state,
            matrix,
            first_bit_position,
            second_bit_position,
        )
        actual = apply_complex64_local_2q(
            state,
            matrix,
            first_bit_position=first_bit_position,
            second_bit_position=second_bit_position,
        )
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

        aliased = state.clone()
        returned = apply_complex64_local_2q(
            aliased,
            matrix,
            first_bit_position=first_bit_position,
            second_bit_position=second_bit_position,
            output=aliased,
        )
        assert returned.data_ptr() == aliased.data_ptr()
        torch.testing.assert_close(aliased, expected, atol=2e-6, rtol=2e-6)


def test_generic_local_2q_rejects_unsupported_contracts() -> None:
    _require_cuda()
    from flagquantum.kernels.triton.statevector_gates import apply_complex64_local_2q

    state = torch.randn(2, 64, dtype=torch.complex64, device="cuda")
    matrix = torch.randn(4, 4, dtype=torch.complex64, device="cuda")
    with pytest.raises(ValueError, match="two distinct local bit positions"):
        apply_complex64_local_2q(
            state,
            matrix,
            first_bit_position=2,
            second_bit_position=2,
        )
    with pytest.raises(ValueError, match="4x4 matrix"):
        apply_complex64_local_2q(
            state,
            matrix[:2, :2],
            first_bit_position=1,
            second_bit_position=4,
        )
    noncontiguous_output = torch.empty(
        64, 2, dtype=torch.complex64, device="cuda"
    ).transpose(0, 1)
    with pytest.raises(ValueError, match="contiguous and match"):
        apply_complex64_local_2q(
            state,
            matrix,
            first_bit_position=1,
            second_bit_position=4,
            output=noncontiguous_output,
        )


def test_public_local_2q_runtime_uses_catalog(monkeypatch) -> None:
    _require_cuda()
    monkeypatch.setenv("FQ_TRITON_TWO_QUBIT_MATRIX", "1")
    routes: list[str] = []
    require_cataloged_kernel = (
        two_qubit_matrix_dispatch._require_two_qubit_matrix_kernel
    )

    def capture_catalog_route(*, device_type: str, dtype: str):
        implementation = require_cataloged_kernel(
            device_type=device_type,
            dtype=dtype,
        )
        routes.append(implementation.implementation_id)
        return implementation

    monkeypatch.setattr(
        two_qubit_matrix_dispatch,
        "_require_two_qubit_matrix_kernel",
        capture_catalog_route,
    )
    generator = torch.Generator(device="cuda").manual_seed(47)
    state = torch.randn(
        1,
        1 << 16,
        generator=generator,
        dtype=torch.complex64,
        device="cuda",
    )
    matrix = torch.randn(
        4,
        4,
        generator=generator,
        dtype=torch.complex64,
        device="cuda",
    )
    expected = _reference_local_2q(state, matrix, 15, 0)

    actual = _apply_matrix(state, matrix, (0, 15), 16)

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert routes == ["FQKI-TRITON-SV-009-A"]


def test_public_local_2q_kill_switch_uses_reference(monkeypatch) -> None:
    _require_cuda()
    monkeypatch.setenv("FQ_TRITON_TWO_QUBIT_MATRIX", "0")
    state = torch.randn(1, 1 << 16, dtype=torch.complex64, device="cuda")
    matrix = torch.randn(4, 4, dtype=torch.complex64, device="cuda")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("disabled SV-009 route must not execute")

    monkeypatch.setattr(
        two_qubit_matrix_dispatch,
        "_apply_cataloged_two_qubit_matrix",
        fail_if_called,
    )
    expected = _reference_local_2q(state, matrix, 15, 0)

    actual = _apply_matrix(state, matrix, (0, 15), 16)

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_public_local_2q_gradient_request_preserves_reference(monkeypatch) -> None:
    _require_cuda()
    monkeypatch.setenv("FQ_TRITON_TWO_QUBIT_MATRIX", "1")
    state = torch.randn(
        1,
        1 << 16,
        dtype=torch.complex64,
        device="cuda",
        requires_grad=True,
    )
    matrix = torch.randn(
        4,
        4,
        dtype=torch.complex64,
        device="cuda",
        requires_grad=True,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("gradient requests must stay on the reference path")

    monkeypatch.setattr(
        two_qubit_matrix_dispatch,
        "_apply_cataloged_two_qubit_matrix",
        fail_if_called,
    )
    output = _apply_matrix(state, matrix, (0, 15), 16)
    output.real.sum().backward()

    assert state.grad is not None
    assert matrix.grad is not None


def test_control_one_pack_unpack_matches_index_reference() -> None:
    _require_cuda()
    from flagquantum.kernels.triton.statevector_gates import (
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


def test_local_cx_inplace_matches_index_reference() -> None:
    _require_cuda()
    from flagquantum.kernels.triton.statevector_gates import (
        apply_complex64_local_cx_inplace,
    )

    state = torch.arange(16, device="cuda").reshape(1, 16).to(torch.complex64)
    expected = state.clone()
    indices = torch.arange(16, device="cuda")
    zero = indices[((indices >> 2) & 1).bool() & ~((indices >> 0) & 1).bool()]
    one = zero | 1
    expected[:, zero], expected[:, one] = state[:, one], state[:, zero]

    actual = state.clone()
    apply_complex64_local_cx_inplace(
        actual, control_bit_position=2, target_bit_position=0
    )

    torch.testing.assert_close(actual, expected)


def test_local_cx_segment_matches_reverse_source_permutation() -> None:
    _require_cuda()
    from flagquantum.kernels.triton.statevector_gates import (
        _local_cx_segment_positions,
        apply_complex64_local_cx_segment,
    )

    state = torch.arange(16, device="cuda").reshape(1, 16).to(torch.complex64)
    controls = (3, 2)
    targets = (1, 0)
    source = torch.arange(16, device="cuda")
    for control, target in reversed(tuple(zip(controls, targets, strict=True))):
        source ^= ((source >> control) & 1) << target
    expected = state[:, source]

    _local_cx_segment_positions.cache_clear()
    output = torch.empty_like(state)
    actual = apply_complex64_local_cx_segment(
        state,
        control_bit_positions=controls,
        target_bit_positions=targets,
        output=output,
    )

    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected)
    apply_complex64_local_cx_segment(
        state,
        control_bit_positions=controls,
        target_bit_positions=targets,
        output=output,
    )
    assert _local_cx_segment_positions.cache_info().hits == 1


def test_constant_ry_rz_triton_path_with_cx_matches_cpu(monkeypatch) -> None:
    _require_cuda()
    monkeypatch.setenv("FQ_TRITON_RY_RZ_PAIR", "1")
    catalog_routes = []
    require_cataloged_kernel = ry_rz_dispatch._require_ry_rz_pair_kernel

    def capture_catalog_route(*, device_type, dtype):
        implementation = require_cataloged_kernel(
            device_type=device_type,
            dtype=dtype,
        )
        catalog_routes.append(implementation.implementation_id)
        return implementation

    monkeypatch.setattr(
        ry_rz_dispatch,
        "_require_ry_rz_pair_kernel",
        capture_catalog_route,
    )
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
    assert catalog_routes == ["FQKI-TRITON-SV-004-A"] * 10


def test_generic_constant_single_qubit_regions_preserve_input_gradient(
    monkeypatch,
) -> None:
    _require_cuda()
    catalog_routes = []
    require_cataloged_kernel = (
        single_qubit_matrix_dispatch._require_single_qubit_matrix_kernel
    )

    def capture_catalog_route(*, device_type, dtype):
        implementation = require_cataloged_kernel(
            device_type=device_type,
            dtype=dtype,
        )
        catalog_routes.append(implementation.implementation_id)
        return implementation

    monkeypatch.setattr(
        single_qubit_matrix_dispatch,
        "_require_single_qubit_matrix_kernel",
        capture_catalog_route,
    )
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
    assert catalog_routes == ["FQKI-TRITON-SV-001-B"] * 2


def test_cx_sequence_catalog_path_matches_cpu_forward_and_backward(monkeypatch) -> None:
    _require_cuda()
    catalog_routes = []
    require_cataloged_kernel = cx_sequence_dispatch._require_cx_sequence_kernel

    def capture_catalog_route(*, device_type, dtype):
        implementation = require_cataloged_kernel(
            device_type=device_type,
            dtype=dtype,
        )
        catalog_routes.append(implementation.implementation_id)
        return implementation

    monkeypatch.setattr(
        cx_sequence_dispatch,
        "_require_cx_sequence_kernel",
        capture_catalog_route,
    )
    torch.manual_seed(41)
    cpu_input = torch.randn(2, 16, dtype=torch.complex64, requires_grad=True)
    cuda_input = cpu_input.detach().cuda().requires_grad_(True)
    cpu = fq.Circuit(4, device="cpu", dtype=torch.complex64, inputs=cpu_input)
    cuda = fq.Circuit(4, device="cuda", dtype=torch.complex64, inputs=cuda_input)
    for circuit in (cpu, cuda):
        circuit.cx(0, 1).cx(1, 2).cx(2, 3).cx(3, 0)

    cpu_state = cpu.state()
    cuda_state = cuda.state()
    weights = torch.arange(1, 17, dtype=torch.float32)
    cpu_loss = (cpu_state.real * weights).sum()
    cuda_loss = (cuda_state.real * weights.cuda()).sum()
    cpu_loss.backward()
    cuda_loss.backward()

    torch.testing.assert_close(cuda_state.cpu(), cpu_state)
    torch.testing.assert_close(cuda_input.grad.cpu(), cpu_input.grad)
    assert cuda._last_statevector_runtime["triton_cx_sequence_regions"] == 1
    assert catalog_routes == ["FQKI-TRITON-SV-003-B"]


def test_parameterized_ry_rz_uses_differentiable_fusion(monkeypatch) -> None:
    _require_cuda()
    monkeypatch.setenv("FQ_TRITON_PARAMETERIZED_SINGLE_QUBIT_MATRIX", "1")
    catalog_routes = []
    require_cataloged_kernel = (
        single_qubit_matrix_dispatch._require_single_qubit_matrix_kernel
    )

    def capture_catalog_route(*, device_type, dtype):
        implementation = require_cataloged_kernel(
            device_type=device_type,
            dtype=dtype,
        )
        catalog_routes.append(implementation.implementation_id)
        return implementation

    monkeypatch.setattr(
        single_qubit_matrix_dispatch,
        "_require_single_qubit_matrix_kernel",
        capture_catalog_route,
    )
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
    assert catalog_routes == ["FQKI-TRITON-SV-001-B"]


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
