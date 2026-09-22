"""Ownership tests for Runtime-independent statevector tensor operations."""

import pytest
import torch

import flagquantum.simulation.statevector.operations as statevector_ops
from flagquantum import Circuit
from flagquantum.core import Instruction
from flagquantum.simulation.statevector.local import _rotation_region_angles
from flagquantum.simulation.statevector.operations import (
    _apply_cross_wire_diagonal_cpu,
    _apply_diagonal_gate_eager,
    _apply_diagonal_matrix,
    _apply_local_gate_eager,
    _apply_matrix,
    _apply_matrix_layout,
    _basis_indices_for_wires,
    _basis_offset,
    _combine_gate_basis_blocks_eager,
    _combine_rank_pair_gate_eager,
    _instruction_matrix,
    _wire_mask,
    _zero_basis_local_indices,
)

pytestmark = pytest.mark.unit


def test_single_qubit_fusion_preserves_wire_order_and_reordering_evidence() -> None:
    layout = ((), ())
    program = tuple(
        statevector_ops._StatevectorGateStep(
            Instruction(name, (wire,)),
            layout,
        )
        for name, wire in (("h", 0), ("x", 1), ("s", 0), ("t", 1))
    )

    fused = statevector_ops._fuse_gate_sequences(program)

    assert tuple(step.wires for step in fused) == ((0,), (1,))
    assert tuple(
        tuple(instruction.name for instruction in step.instructions) for step in fused
    ) == (("h", "s"), ("x", "t"))
    assert all(step.dependency_reordered for step in fused)


def test_rotation_region_angles_preserve_values_and_gradients() -> None:
    circuit = Circuit(1)
    theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    step = statevector_ops._StatevectorFusedGateStep(
        instructions=(Instruction("rx", (0,), params={"theta": theta}),),
        wires=(0,),
        layout=((), ()),
    )
    angles = _rotation_region_angles(
        circuit, (step, step), torch.zeros(2, 2, dtype=torch.complex128), None
    )
    assert angles is not None
    torch.testing.assert_close(angles, theta.expand(2, 2, 1))
    torch.testing.assert_close(
        torch.autograd.grad(angles.sum(), theta)[0], theta.new_tensor(4)
    )


def test_rotation_region_angles_decline_unparameterized_gate() -> None:
    step = statevector_ops._StatevectorFusedGateStep(
        instructions=(Instruction("h", (0,)),), wires=(0,), layout=((), ())
    )
    assert (
        _rotation_region_angles(
            Circuit(1), (step,), torch.zeros(1, 2, dtype=torch.complex128), None
        )
        is None
    )


def test_rotation_sequence_rejects_empty_topology() -> None:
    with pytest.raises(ValueError, match="RX/RY/RZ topology"):
        statevector_ops._batched_rotation_sequence_matrices(
            torch.empty(2, 1, 0), names=(), dtype=torch.complex64
        )


def test_rotation_sequence_matches_closed_form_values_and_gradients() -> None:
    angles = torch.linspace(-0.7, 0.9, 18, dtype=torch.float64).reshape(2, 3, 3)
    angles.requires_grad_()
    actual = statevector_ops._batched_rotation_sequence_matrices(
        angles, names=("rx", "ry", "rz"), dtype=torch.complex128
    )
    expected = statevector_ops._batched_rx_ry_rz_matrices(angles)
    actual_gradient = torch.autograd.grad(actual.real.sum(), angles, retain_graph=True)[
        0
    ]
    expected_gradient = torch.autograd.grad(expected.real.sum(), angles)[0]

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_zero_basis_indices_preserve_wire_order():
    indices = _zero_basis_local_indices(
        0,
        4,
        (1, 3),
        n_wires=4,
        rank_bits=0,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(indices, torch.tensor([0, 2, 8, 10]))


def test_basis_bit_extraction_preserves_requested_wire_order():
    basis = _basis_indices_for_wires(torch.arange(8), n_wires=3, wires=(2, 0))

    torch.testing.assert_close(basis, torch.tensor([0, 2, 0, 2, 1, 3, 1, 3]))


def test_gate_basis_offsets_preserve_wire_order():
    assert _wire_mask(4, 1) == 4
    assert tuple(_basis_offset(4, (1, 3), basis) for basis in range(4)) == (
        0,
        1,
        4,
        5,
    )


def test_local_eager_gate_kernel_is_usable_without_runtime_models():
    amplitudes = torch.tensor([[1, 0, 0, 0]], dtype=torch.complex64)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)

    evolved, peak_bytes = _apply_local_gate_eager(
        amplitudes,
        x,
        (1,),
        n_wires=2,
        rank_bits=0,
        chunk_amplitudes=2,
    )

    torch.testing.assert_close(
        evolved, torch.tensor([[0, 1, 0, 0]], dtype=torch.complex64)
    )
    assert peak_bytes >= evolved.numel() * evolved.element_size()


def test_diagonal_eager_gate_kernel_is_usable_without_runtime_models():
    amplitudes = torch.ones((1, 4), dtype=torch.complex64)
    diagonal = torch.tensor([1, 1j], dtype=torch.complex64)

    evolved, scratch_bytes = _apply_diagonal_gate_eager(
        amplitudes,
        diagonal,
        torch.arange(4),
        (1,),
        n_wires=2,
    )

    torch.testing.assert_close(
        evolved, torch.tensor([[1, 1j, 1, 1j]], dtype=torch.complex64)
    )
    assert scratch_bytes == evolved.numel() * evolved.element_size()


@pytest.mark.parametrize("rank_basis", (0, 1))
def test_rank_pair_gate_kernel_is_usable_without_runtime_models(rank_basis):
    local = torch.tensor([[1 + 2j, 3 + 4j]], dtype=torch.complex64)
    remote = torch.tensor([[5 + 6j, 7 + 8j]], dtype=torch.complex64)
    matrix = torch.tensor([[2, 3], [5, 7]], dtype=torch.complex64)
    basis_zero, basis_one = (local, remote) if rank_basis == 0 else (remote, local)

    updated, scratch_bytes = _combine_rank_pair_gate_eager(
        local, remote, matrix, rank_basis=rank_basis
    )

    expected = basis_zero * matrix[rank_basis, 0] + basis_one * matrix[rank_basis, 1]
    torch.testing.assert_close(updated, expected)
    assert scratch_bytes == 3 * updated.numel() * updated.element_size()


def test_gate_basis_block_combination_is_usable_without_runtime_models():
    basis_inputs = (
        torch.tensor([[1, 2]], dtype=torch.complex64),
        torch.tensor([[3, 4]], dtype=torch.complex64),
    )
    matrix = torch.tensor([[2, 3], [5, 7]], dtype=torch.complex64)
    output_basis = torch.tensor([0, 1])

    updated, scratch_bytes = _combine_gate_basis_blocks_eager(
        basis_inputs, matrix, output_basis
    )

    torch.testing.assert_close(updated, torch.tensor([[11, 38]], dtype=torch.complex64))
    assert scratch_bytes == 4 * updated.numel() * updated.element_size()


def test_gate_basis_block_combination_supports_batched_matrices():
    vectors = torch.tensor([[1, 3], [2, 4]], dtype=torch.complex64)
    matrices = torch.tensor(
        [[[2, 3], [5, 7]], [[11, 13], [17, 19]]], dtype=torch.complex64
    )
    basis_inputs = tuple(
        vectors[:, basis : basis + 1].expand(-1, 2) for basis in range(2)
    )

    updated, _ = _combine_gate_basis_blocks_eager(
        basis_inputs, matrices, torch.arange(2)
    )

    expected = torch.bmm(matrices, vectors.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(updated, expected)


def test_constant_gate_matrix_is_constructed_on_cpu_before_device_transfer(
    monkeypatch,
):
    observed = {}

    def recording_gate_matrix(instruction, *, bsz, device, dtype):
        observed["device"] = torch.device(device)
        return torch.eye(2, dtype=dtype, device=device)

    monkeypatch.setattr(statevector_ops, "_gate_matrix", recording_gate_matrix)
    matrix = _instruction_matrix(
        Instruction("rx", (0,), {"theta": 0.2}),
        device=torch.device("meta"),
        dtype=torch.complex128,
    )

    assert observed["device"].type == "cpu"
    assert matrix.device.type == "meta"
    assert matrix.dtype == torch.complex128


def test_tensor_parameter_matrix_preserves_device_autograd_path(monkeypatch):
    observed = {}
    theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)

    def recording_gate_matrix(instruction, *, bsz, device, dtype):
        observed["device"] = torch.device(device)
        return torch.eye(2, dtype=dtype, device=device)

    monkeypatch.setattr(statevector_ops, "_gate_matrix", recording_gate_matrix)
    _instruction_matrix(
        Instruction("rx", (0,), {"theta": theta}),
        device=torch.device("meta"),
        dtype=torch.complex128,
    )

    assert observed["device"].type == "meta"


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
@pytest.mark.parametrize("wire", (0, 1, 3))
def test_cpu_single_qubit_direct_kernel_matches_layout_reference(dtype, wire):
    generator = torch.Generator().manual_seed(1729 + wire)
    state = torch.randn((3, 16), dtype=dtype, generator=generator)
    matrix = torch.randn((3, 2, 2), dtype=dtype, generator=generator)

    actual = _apply_matrix(state, matrix, (wire,), 4)
    expected = _apply_matrix_layout(state, matrix, (wire,), 4)

    torch.testing.assert_close(actual, expected)
    assert actual.is_contiguous()


def test_cpu_single_qubit_direct_kernel_does_not_dispatch_bmm(monkeypatch):
    state = torch.randn((2, 32), dtype=torch.complex128)
    matrix = torch.randn((2, 2), dtype=torch.complex128)

    def reject_bmm(*args, **kwargs):
        raise AssertionError("CPU single-qubit fast path must not call torch.bmm")

    monkeypatch.setattr(torch, "bmm", reject_bmm)

    result = _apply_matrix(state, matrix, (2,), 5)

    assert result.shape == state.shape


def test_cpu_single_qubit_direct_kernel_preserves_state_and_matrix_gradients():
    generator = torch.Generator().manual_seed(1733)
    state = torch.randn(
        (2, 8), dtype=torch.complex128, generator=generator, requires_grad=True
    )
    matrix = torch.randn(
        (2, 2, 2), dtype=torch.complex128, generator=generator, requires_grad=True
    )

    actual = _apply_matrix(state, matrix, (1,), 3)
    expected = _apply_matrix_layout(state, matrix, (1,), 3)
    actual_loss = torch.abs(actual).square().sum()
    expected_loss = torch.abs(expected).square().sum()
    actual_gradients = torch.autograd.grad(
        actual_loss, (state, matrix), retain_graph=True
    )
    expected_gradients = torch.autograd.grad(expected_loss, (state, matrix))

    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
@pytest.mark.parametrize("batched_matrix", (False, True))
def test_cpu_reversed_trailing_two_qubit_kernel_matches_layout_reference(
    dtype, batched_matrix
):
    generator = torch.Generator().manual_seed(1739)
    state = torch.randn((3, 32), dtype=dtype, generator=generator)
    matrix_shape = (3, 4, 4) if batched_matrix else (4, 4)
    matrix = torch.randn(matrix_shape, dtype=dtype, generator=generator)

    actual = _apply_matrix(state, matrix, (4, 3), 5)
    expected = _apply_matrix_layout(state, matrix, (4, 3), 5)

    torch.testing.assert_close(actual, expected)
    assert actual.is_contiguous()


def test_cpu_reversed_trailing_two_qubit_kernel_bypasses_layout(monkeypatch):
    state = torch.randn((2, 32), dtype=torch.complex128)
    matrix = torch.randn((4, 4), dtype=torch.complex128)

    def reject_layout(*args, **kwargs):
        raise AssertionError("reversed trailing two-qubit fast path must bypass layout")

    monkeypatch.setattr(statevector_ops, "_apply_matrix_layout", reject_layout)

    result = _apply_matrix(state, matrix, (4, 3), 5)

    assert result.shape == state.shape


def test_cpu_other_two_qubit_orders_keep_layout_path(monkeypatch):
    state = torch.randn((2, 32), dtype=torch.complex128)
    matrix = torch.randn((4, 4), dtype=torch.complex128)
    observed = []

    def recording_layout(state, matrix, wires, n_wires, layout=None):
        observed.append(tuple(wires))
        return state.clone()

    monkeypatch.setattr(statevector_ops, "_apply_matrix_layout", recording_layout)

    _apply_matrix(state, matrix, (3, 4), 5)
    _apply_matrix(state, matrix, (2, 1), 5)

    assert observed == [(3, 4), (2, 1)]


def test_cpu_reversed_trailing_two_qubit_kernel_preserves_gradients():
    generator = torch.Generator().manual_seed(1741)
    state = torch.randn(
        (2, 16), dtype=torch.complex128, generator=generator, requires_grad=True
    )
    matrix = torch.randn(
        (2, 4, 4), dtype=torch.complex128, generator=generator, requires_grad=True
    )

    actual = _apply_matrix(state, matrix, (3, 2), 4)
    expected = _apply_matrix_layout(state, matrix, (3, 2), 4)
    actual_gradients = torch.autograd.grad(
        torch.abs(actual).square().sum(), (state, matrix), retain_graph=True
    )
    expected_gradients = torch.autograd.grad(
        torch.abs(expected).square().sum(), (state, matrix)
    )

    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("wires", ((0,), (2,), (0, 2), (2, 0)))
def test_cpu_diagonal_kernel_matches_layout_reference(wires):
    generator = torch.Generator().manual_seed(1741 + sum(wires))
    state = torch.randn((2, 8), dtype=torch.complex128, generator=generator)
    dim = 2 ** len(wires)
    diagonal = torch.randn((2, dim), dtype=torch.complex128, generator=generator)
    matrix = torch.diag_embed(diagonal)

    actual = _apply_diagonal_matrix(state, matrix, wires, 3)
    expected = _apply_matrix_layout(state, matrix, wires, 3)

    torch.testing.assert_close(actual, expected)
    assert actual.is_contiguous()


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
@pytest.mark.parametrize("wires", ((0, 2, 4), (4, 1, 3)))
def test_cpu_cross_wire_diagonal_kernel_matches_sequential_application(dtype, wires):
    generator = torch.Generator().manual_seed(1753 + sum(wires))
    state = torch.randn((3, 32), dtype=dtype, generator=generator)
    diagonals = tuple(
        torch.randn((3, 2), dtype=dtype, generator=generator) for _ in wires
    )

    actual = _apply_cross_wire_diagonal_cpu(state, diagonals, wires, 5)
    expected = state
    for wire, diagonal in zip(wires, diagonals, strict=True):
        expected = _apply_diagonal_matrix(
            expected, torch.diag_embed(diagonal), (wire,), 5
        )

    torch.testing.assert_close(actual, expected)
    assert actual.is_contiguous()


def test_cpu_cross_wire_diagonal_kernel_preserves_gradients():
    generator = torch.Generator().manual_seed(1759)
    state = torch.randn(
        (2, 16), dtype=torch.complex128, generator=generator, requires_grad=True
    )
    diagonals = tuple(
        torch.randn(
            (2, 2), dtype=torch.complex128, generator=generator, requires_grad=True
        )
        for _ in range(3)
    )
    wires = (3, 0, 2)

    actual = _apply_cross_wire_diagonal_cpu(state, diagonals, wires, 4)
    expected = state
    for wire, diagonal in zip(wires, diagonals, strict=True):
        expected = _apply_diagonal_matrix(
            expected, torch.diag_embed(diagonal), (wire,), 4
        )
    actual_gradients = torch.autograd.grad(
        torch.abs(actual).square().sum(), (state, *diagonals), retain_graph=True
    )
    expected_gradients = torch.autograd.grad(
        torch.abs(expected).square().sum(), (state, *diagonals)
    )

    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_cpu_cross_wire_diagonal_circuit_preserves_parameter_autograd(monkeypatch):
    generator = torch.Generator().manual_seed(1763)
    inputs = torch.randn(
        (2, 16), dtype=torch.complex128, generator=generator, requires_grad=True
    )
    angles = torch.linspace(-0.4, 0.5, 4, dtype=torch.float64, requires_grad=True)
    circuit = Circuit(4, dtype=torch.complex128, inputs=inputs)
    for wire in range(4):
        circuit.rz(wire, theta=angles[wire]).p(wire, theta=0.07 * (wire + 1))

    monkeypatch.setenv("FQ_CPU_CROSS_WIRE_DIAGONAL_FUSION", "0")
    expected = circuit.state(refresh=True)
    expected_gradients = torch.autograd.grad(
        torch.abs(expected).square().sum(), (inputs, angles), retain_graph=True
    )
    monkeypatch.setenv("FQ_CPU_CROSS_WIRE_DIAGONAL_FUSION", "1")
    actual = circuit.state(refresh=True)
    actual_gradients = torch.autograd.grad(
        torch.abs(actual).square().sum(), (inputs, angles)
    )

    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    assert circuit._last_statevector_runtime["statevector_apply_count"] == 1
