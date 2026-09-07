"""Ownership tests for Runtime-independent statevector tensor operations."""

import pytest
import torch

import flagquantum.simulation.statevector.operations as statevector_ops
from flagquantum.core import Instruction
from flagquantum.simulation.statevector.operations import (
    _apply_diagonal_gate_eager,
    _apply_local_gate_eager,
    _basis_indices_for_wires,
    _basis_offset,
    _combine_gate_basis_blocks_eager,
    _combine_rank_pair_gate_eager,
    _instruction_matrix,
    _wire_mask,
    _zero_basis_local_indices,
)

pytestmark = pytest.mark.unit


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
