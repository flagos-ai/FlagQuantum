"""Ownership tests for Runtime-independent statevector tensor operations."""

import pytest
import torch

import flagquantum.simulation.statevector.operations as statevector_ops
from flagquantum import Circuit
from flagquantum.core import Instruction
from flagquantum.simulation.statevector.local import (
    _batched_kronecker_product,
    _rotation_region_angles,
)
from flagquantum.simulation.statevector.operations import (
    _apply_cross_wire_diagonal_cpu,
    _apply_diagonal_gate_eager,
    _apply_diagonal_matrix,
    _apply_disjoint_diagonal_regions_cpu,
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


def test_two_wire_diagonal_regions_are_partitioned_into_disjoint_matchings() -> None:
    layout = ((), ())
    regions = tuple(
        statevector_ops._StatevectorGateStep(
            Instruction("cz", (left, left + 1)),
            layout,
        )
        for left in range(5)
    )

    optimized = statevector_ops._fuse_cross_wire_diagonal_regions(regions)

    assert len(optimized) == 2
    assert all(
        isinstance(step, statevector_ops._StatevectorCrossWireDiagonalStep)
        for step in optimized
    )
    assert tuple(
        tuple(statevector_ops._region_wires(region) for region in step.regions)
        for step in optimized
        if isinstance(step, statevector_ops._StatevectorCrossWireDiagonalStep)
    ) == (((0, 1), (2, 3), (4, 5)), ((1, 2), (3, 4)))


def test_non_diagonal_region_ends_a_diagonal_matching_segment() -> None:
    layout = ((), ())
    regions = tuple(
        statevector_ops._StatevectorGateStep(Instruction(name, wires), layout)
        for name, wires in (
            ("cz", (0, 1)),
            ("cz", (2, 3)),
            ("cx", (0, 1)),
            ("cz", (0, 1)),
            ("cz", (2, 3)),
        )
    )

    optimized = statevector_ops._fuse_cross_wire_diagonal_regions(regions)

    assert len(optimized) == 3
    assert isinstance(optimized[0], statevector_ops._StatevectorCrossWireDiagonalStep)
    assert isinstance(optimized[1], statevector_ops._StatevectorGateStep)
    assert isinstance(optimized[2], statevector_ops._StatevectorCrossWireDiagonalStep)


def test_dense_single_wire_regions_are_packed_in_bounded_groups() -> None:
    assert statevector_ops._CPU_DISJOINT_DENSE_MAX_WIRES == 4
    layout = ((), ())
    regions = tuple(
        statevector_ops._StatevectorFusedGateStep(
            instructions=(Instruction("ry", (wire,), params={"theta": 0.1}),),
            wires=(wire,),
            layout=layout,
        )
        for wire in range(9)
    )

    optimized = statevector_ops._fuse_disjoint_dense_regions(regions)

    assert tuple(
        (
            len(step.regions)
            if isinstance(step, statevector_ops._StatevectorDisjointDenseStep)
            else 1
        )
        for step in optimized
    ) == (4, 4, 1)


def test_dense_single_wire_grouping_includes_diagonal_regions() -> None:
    layout = ((), ())
    dense = tuple(
        statevector_ops._StatevectorFusedGateStep(
            instructions=(Instruction("ry", (wire,), params={"theta": 0.1}),),
            wires=(wire,),
            layout=layout,
        )
        for wire in range(4)
    )
    diagonal = statevector_ops._StatevectorFusedGateStep(
        instructions=(Instruction("rz", (4,), params={"theta": 0.2}),),
        wires=(4,),
        layout=layout,
        diagonal=True,
    )

    optimized = statevector_ops._fuse_disjoint_dense_regions(
        (*dense[:2], diagonal, *dense[2:])
    )

    assert len(optimized) == 2
    assert isinstance(optimized[0], statevector_ops._StatevectorDisjointDenseStep)
    assert optimized[0].regions == (*dense[:2], diagonal, dense[2])
    assert optimized[1] is dense[3]


def test_unfused_dense_single_wire_layers_include_fixed_gates() -> None:
    layout = ((), ())
    rotations = tuple(
        statevector_ops._StatevectorGateStep(
            Instruction("ry", (wire,), params={"theta": 0.1}), layout
        )
        for wire in range(6)
    )
    fixed = statevector_ops._StatevectorGateStep(Instruction("x", (0,)), layout)

    optimized = statevector_ops._fuse_disjoint_dense_regions(
        (*rotations[:3], fixed, *rotations[3:])
    )

    assert len(optimized) == 2
    assert isinstance(optimized[0], statevector_ops._StatevectorDisjointDenseStep)
    assert optimized[0].regions == rotations[:3]
    assert isinstance(optimized[1], statevector_ops._StatevectorDisjointDenseStep)
    assert optimized[1].regions == (fixed, *rotations[3:])


def test_mixed_width_dense_regions_use_total_wire_bound() -> None:
    layout = ((), ())
    regions = tuple(
        statevector_ops._StatevectorGateStep(instruction, layout)
        for instruction in (
            Instruction("rxx", (0, 1), params={"theta": 0.1}),
            Instruction("ry", (2,), params={"theta": 0.2}),
            Instruction("crx", (3, 4), params={"theta": 0.3}),
            Instruction("ry", (5,), params={"theta": 0.4}),
        )
    )

    optimized = statevector_ops._fuse_disjoint_dense_regions(regions)

    assert len(optimized) == 2
    assert all(
        isinstance(step, statevector_ops._StatevectorDisjointDenseStep)
        for step in optimized
    )
    assert tuple(
        tuple(statevector_ops._region_wires(region) for region in step.regions)
        for step in optimized
        if isinstance(step, statevector_ops._StatevectorDisjointDenseStep)
    ) == (((0, 1), (2,)), ((3, 4), (5,)))


@pytest.mark.parametrize("gate", ("cx", "swap"))
def test_two_wire_permutation_gate_ends_dense_group(gate: str) -> None:
    layout = ((), ())
    regions = tuple(
        statevector_ops._StatevectorGateStep(instruction, layout)
        for instruction in (
            Instruction("ry", (0,), params={"theta": 0.1}),
            Instruction(gate, (1, 2)),
            Instruction("ry", (3,), params={"theta": 0.2}),
        )
    )

    optimized = statevector_ops._fuse_disjoint_dense_regions(regions)

    assert optimized == list(regions)


def test_two_dense_two_wire_regions_are_not_grouped() -> None:
    layout = ((), ())
    regions = tuple(
        statevector_ops._StatevectorGateStep(instruction, layout)
        for instruction in (
            Instruction("rxx", (0, 1), params={"theta": 0.1}),
            Instruction("crx", (2, 3), params={"theta": 0.2}),
        )
    )

    optimized = statevector_ops._fuse_disjoint_dense_regions(regions)

    assert optimized == list(regions)


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


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_cpu_disjoint_diagonal_regions_match_sequential_application(dtype):
    generator = torch.Generator().manual_seed(1761)
    state = torch.randn((2, 32), dtype=dtype, generator=generator)
    wire_groups = ((3, 1), (0,), (4, 2))
    diagonals = tuple(
        torch.randn((2, 2 ** len(wires)), dtype=dtype, generator=generator)
        for wires in wire_groups
    )

    actual = _apply_disjoint_diagonal_regions_cpu(
        state, diagonals, wire_groups, n_wires=5
    )
    expected = state
    for wires, diagonal in zip(wire_groups, diagonals, strict=True):
        expected = _apply_diagonal_matrix(
            expected, torch.diag_embed(diagonal), wires, n_wires=5
        )

    torch.testing.assert_close(actual, expected)
    assert actual.is_contiguous()


def test_cpu_disjoint_diagonal_regions_reject_overlapping_wires():
    state = torch.ones((1, 8), dtype=torch.complex64)
    diagonals = (
        torch.ones((1, 4), dtype=torch.complex64),
        torch.ones((1, 4), dtype=torch.complex64),
    )

    with pytest.raises(ValueError, match="wire-disjoint"):
        _apply_disjoint_diagonal_regions_cpu(
            state, diagonals, ((0, 1), (1, 2)), n_wires=3
        )


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_batched_kronecker_product_matches_torch_kron(dtype):
    generator = torch.Generator().manual_seed(1762)
    matrices = tuple(
        torch.randn((2, dimension, dimension), dtype=dtype, generator=generator)
        for dimension in (2, 4, 2)
    )

    actual = _batched_kronecker_product(matrices, batch_size=2)
    expected = torch.stack(
        tuple(
            torch.kron(
                torch.kron(matrices[0][batch], matrices[1][batch]), matrices[2][batch]
            )
            for batch in range(2)
        )
    )

    torch.testing.assert_close(actual, expected)


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


def test_cpu_two_wire_diagonal_matchings_preserve_parameter_autograd(monkeypatch):
    generator = torch.Generator().manual_seed(1767)
    inputs = torch.randn(
        (2, 64), dtype=torch.complex128, generator=generator, requires_grad=True
    )
    angles = torch.linspace(-0.4, 0.5, 5, dtype=torch.float64, requires_grad=True)
    circuit = Circuit(6, dtype=torch.complex128, inputs=inputs)
    for left in range(5):
        if left % 2:
            circuit.cphase(left, left + 1, theta=angles[left])
        else:
            circuit.rzz(left, left + 1, theta=angles[left])

    monkeypatch.setenv("FQ_CPU_CROSS_WIRE_DIAGONAL_FUSION", "0")
    expected = circuit.state(refresh=True)
    expected_gradients = torch.autograd.grad(
        expected.real.sum(), (inputs, angles), retain_graph=True
    )
    monkeypatch.setenv("FQ_CPU_CROSS_WIRE_DIAGONAL_FUSION", "1")
    actual = circuit.state(refresh=True)
    actual_gradients = torch.autograd.grad(actual.real.sum(), (inputs, angles))

    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    assert circuit._last_statevector_runtime["statevector_apply_count"] == 2


def test_cpu_disjoint_single_wire_fusion_preserves_parameter_autograd(monkeypatch):
    generator = torch.Generator().manual_seed(1771)
    inputs = torch.randn(
        (2, 64), dtype=torch.complex128, generator=generator, requires_grad=True
    )
    angles = torch.linspace(-0.4, 0.5, 12, dtype=torch.float64, requires_grad=True)
    circuit = Circuit(6, dtype=torch.complex128, inputs=inputs)
    for wire in range(6):
        circuit.rx(wire, theta=angles[wire]).ry(wire, theta=angles[wire + 6])

    monkeypatch.setenv("FQ_CPU_DISJOINT_SINGLE_WIRE_FUSION", "0")
    expected = circuit.state(refresh=True)
    expected_gradients = torch.autograd.grad(
        expected.real.sum(), (inputs, angles), retain_graph=True
    )
    monkeypatch.setenv("FQ_CPU_DISJOINT_SINGLE_WIRE_FUSION", "1")
    actual = circuit.state(refresh=True)
    actual_gradients = torch.autograd.grad(actual.real.sum(), (inputs, angles))

    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    assert circuit._last_statevector_runtime["statevector_apply_count"] == 2


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_cpu_unfused_single_wire_layer_matches_sequential_execution(monkeypatch, dtype):
    generator = torch.Generator().manual_seed(1773)
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    inputs = torch.randn((2, 64), dtype=dtype, generator=generator)
    angles = torch.linspace(-0.4, 0.5, 6, dtype=real_dtype)
    circuit = Circuit(6, dtype=dtype, inputs=inputs)
    for wire in range(6):
        circuit.ry(wire, theta=angles[wire])

    monkeypatch.setenv("FQ_CPU_DISJOINT_SINGLE_WIRE_FUSION", "0")
    expected = circuit.state(refresh=True)
    monkeypatch.setenv("FQ_CPU_DISJOINT_SINGLE_WIRE_FUSION", "1")
    actual = circuit.state(refresh=True)

    torch.testing.assert_close(actual, expected)
    assert circuit._last_statevector_runtime["statevector_apply_count"] == 2


def test_cpu_unfused_single_wire_layer_preserves_parameter_autograd(monkeypatch):
    generator = torch.Generator().manual_seed(1775)
    inputs = torch.randn(
        (2, 64), dtype=torch.complex128, generator=generator, requires_grad=True
    )
    angles = torch.linspace(-0.4, 0.5, 6, dtype=torch.float64, requires_grad=True)
    circuit = Circuit(6, dtype=torch.complex128, inputs=inputs)
    for wire in range(6):
        circuit.ry(wire, theta=angles[wire])

    monkeypatch.setenv("FQ_CPU_DISJOINT_SINGLE_WIRE_FUSION", "0")
    expected = circuit.state(refresh=True)
    expected_gradients = torch.autograd.grad(
        expected.real.sum(), (inputs, angles), retain_graph=True
    )
    monkeypatch.setenv("FQ_CPU_DISJOINT_SINGLE_WIRE_FUSION", "1")
    actual = circuit.state(refresh=True)
    actual_gradients = torch.autograd.grad(actual.real.sum(), (inputs, angles))

    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_cpu_mixed_single_wire_layer_preserves_parameter_autograd(
    monkeypatch, dtype: torch.dtype
) -> None:
    generator = torch.Generator().manual_seed(1777)
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    inputs = torch.randn((2, 64), dtype=dtype, generator=generator, requires_grad=True)
    angles = torch.linspace(-0.4, 0.5, 3, dtype=real_dtype, requires_grad=True)
    circuit = Circuit(6, dtype=dtype, inputs=inputs)
    circuit.x(0).y(1).rz(2, theta=angles[0]).ry(3, theta=angles[1])
    circuit.x(4).p(5, theta=angles[2])

    monkeypatch.setenv("FQ_CPU_DISJOINT_SINGLE_WIRE_FUSION", "0")
    expected = circuit.state(refresh=True)
    expected_gradients = torch.autograd.grad(
        expected.real.sum(), (inputs, angles), retain_graph=True
    )
    baseline_statistics = dict(circuit._last_statevector_runtime)

    monkeypatch.setenv("FQ_CPU_DISJOINT_SINGLE_WIRE_FUSION", "1")
    actual = circuit.state(refresh=True)
    actual_gradients = torch.autograd.grad(actual.real.sum(), (inputs, angles))
    fused_statistics = dict(circuit._last_statevector_runtime)

    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    assert baseline_statistics["statevector_apply_count"] == 6
    assert baseline_statistics["permutation_gates"] == 2
    assert baseline_statistics["fixed_single_qubit_specialized_gates"] == 1
    assert baseline_statistics["diagonal_elementwise_gates"] == 2
    assert fused_statistics["statevector_apply_count"] == 2
    assert fused_statistics["permutation_gates"] == 0
    assert fused_statistics["fixed_single_qubit_specialized_gates"] == 0
    assert fused_statistics["diagonal_elementwise_gates"] == 0


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_cpu_mixed_width_layers_preserve_parameter_autograd(
    monkeypatch, dtype: torch.dtype
) -> None:
    generator = torch.Generator().manual_seed(1781)
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    inputs = torch.randn((2, 64), dtype=dtype, generator=generator, requires_grad=True)
    angles = torch.linspace(-0.4, 0.5, 4, dtype=real_dtype, requires_grad=True)
    circuit = Circuit(6, dtype=dtype, inputs=inputs)
    circuit.rxx(0, 1, theta=angles[0]).ry(2, theta=angles[1])
    circuit.crx(3, 4, theta=angles[2]).rz(5, theta=angles[3])

    monkeypatch.setenv("FQ_CPU_DISJOINT_SINGLE_WIRE_FUSION", "0")
    expected = circuit.state(refresh=True)
    expected_gradients = torch.autograd.grad(
        expected.real.sum(), (inputs, angles), retain_graph=True
    )
    baseline_statistics = dict(circuit._last_statevector_runtime)

    monkeypatch.setenv("FQ_CPU_DISJOINT_SINGLE_WIRE_FUSION", "1")
    actual = circuit.state(refresh=True)
    actual_gradients = torch.autograd.grad(actual.real.sum(), (inputs, angles))
    fused_statistics = dict(circuit._last_statevector_runtime)

    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    assert baseline_statistics["statevector_apply_count"] == 4
    assert fused_statistics["statevector_apply_count"] == 2
    assert fused_statistics["diagonal_elementwise_gates"] == 0


@pytest.mark.parametrize(
    ("gate", "counter"),
    (("x", "permutation_gates"), ("y", "fixed_single_qubit_specialized_gates")),
)
def test_single_fixed_gate_keeps_its_specialized_cpu_path(
    gate: str, counter: str
) -> None:
    circuit = Circuit(1, dtype=torch.complex128)
    getattr(circuit, gate)(0)

    circuit.state(refresh=True)

    assert circuit._last_statevector_runtime["statevector_apply_count"] == 1
    assert circuit._last_statevector_runtime[counter] == 1


def _two_disjoint_dense_two_wire_circuit(
    batch: int,
    dtype: torch.dtype,
    *,
    requires_grad: bool = False,
) -> tuple[Circuit, torch.Tensor, torch.Tensor]:
    """Build a batched circuit whose first layer holds two disjoint 2-wire gates."""

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    generator = torch.Generator().manual_seed(1783)
    inputs = torch.randn(
        (batch, 16),
        dtype=dtype,
        generator=generator,
        requires_grad=requires_grad,
    )
    angles = torch.linspace(-0.4, 0.5, 4, dtype=real_dtype, requires_grad=requires_grad)
    circuit = Circuit(4, dtype=dtype, inputs=inputs)
    circuit.rxx(0, 1, theta=angles[0])
    circuit.ryy(2, 3, theta=angles[1])
    circuit.ry(0, theta=angles[2])
    circuit.rz(3, theta=angles[3])
    return circuit, inputs, angles


def _disjoint_dense_region_shapes(circuit: Circuit) -> tuple[tuple[int, ...], ...]:
    """Return the region widths of every cached disjoint dense step."""

    program = list(circuit._backend_programs.values())[-1]
    return tuple(
        tuple(len(statevector_ops._region_wires(region)) for region in step.regions)
        for step in program
        if isinstance(step, statevector_ops._StatevectorDisjointDenseStep)
    )


def test_batched_plan_groups_two_disjoint_dense_two_wire_regions() -> None:
    instructions = (
        Instruction("rxx", (0, 1), params={"theta": 0.1}),
        Instruction("ryy", (2, 3), params={"theta": 0.2}),
    )

    program = statevector_ops._compile_statevector_program(
        instructions,
        4,
        enable_triton_loop=False,
        enable_cpu_disjoint_single_wire=True,
        max_two_wire_regions=2,
    )

    assert len(program) == 1
    step = program[0]
    assert isinstance(step, statevector_ops._StatevectorDisjointDenseStep)
    assert tuple(
        tuple(statevector_ops._region_wires(region) for region in step.regions)
    ) == ((0, 1), (2, 3))


def test_single_state_plan_keeps_two_dense_two_wire_regions_separate() -> None:
    instructions = (
        Instruction("rxx", (0, 1), params={"theta": 0.1}),
        Instruction("ryy", (2, 3), params={"theta": 0.2}),
    )

    program = statevector_ops._compile_statevector_program(
        instructions,
        4,
        enable_triton_loop=False,
        enable_cpu_disjoint_single_wire=True,
        max_two_wire_regions=1,
    )

    assert [(type(step).__name__, step.instruction.name) for step in program] == [
        ("_StatevectorGateStep", "rxx"),
        ("_StatevectorGateStep", "ryy"),
    ]


@pytest.mark.parametrize("gate", ("cx", "swap"))
def test_batched_plan_keeps_standalone_permutation_gates_as_barriers(gate: str) -> None:
    instructions = (
        Instruction("rxx", (0, 1), params={"theta": 0.1}),
        Instruction(gate, (2, 3)),
        Instruction("ryy", (0, 1), params={"theta": 0.2}),
    )

    program = statevector_ops._compile_statevector_program(
        instructions,
        4,
        enable_triton_loop=False,
        enable_cpu_disjoint_single_wire=True,
        max_two_wire_regions=2,
    )

    assert not any(
        isinstance(step, statevector_ops._StatevectorDisjointDenseStep)
        for step in program
    )
    assert [step.instruction.name for step in program] == ["rxx", gate, "ryy"]


def test_batched_plan_keeps_cx_sequence_compilation() -> None:
    instructions = (
        Instruction("rxx", (0, 2), params={"theta": 0.1}),
        Instruction("ryy", (1, 3), params={"theta": 0.2}),
        *(Instruction("cx", (wire, wire + 4)) for wire in range(4)),
    )

    program = statevector_ops._compile_statevector_program(
        instructions,
        8,
        enable_triton_loop=False,
        enable_cpu_disjoint_single_wire=True,
        max_two_wire_regions=2,
    )

    assert isinstance(program[0], statevector_ops._StatevectorDisjointDenseStep)
    assert program[1] == statevector_ops._StatevectorCXSequenceStep(
        controls=(0, 1, 2, 3),
        targets=(4, 5, 6, 7),
    )


def test_batch_one_execution_plan_keeps_the_single_state_grouping(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FQ_CPU_DISJOINT_SINGLE_WIRE_FUSION", "1")
    circuit, _, _ = _two_disjoint_dense_two_wire_circuit(1, torch.complex128)
    circuit.state(refresh=True)

    key = list(circuit._backend_programs)[-1]
    compiled = statevector_ops._compile_statevector_program(
        circuit._instructions,
        circuit.n_wires,
        enable_triton_loop=False,
        enable_cpu_cross_wire_diagonal=True,
        enable_cpu_disjoint_single_wire=True,
        max_two_wire_regions=1,
    )

    assert key[-2:] == ("cpu_disjoint_dense_max_two_wire_regions", 1)
    assert circuit._backend_programs[key] == compiled
    assert _disjoint_dense_region_shapes(circuit) == ((2, 1),)

    monkeypatch.setenv("FQ_CPU_DISJOINT_SINGLE_WIRE_FUSION", "0")
    rolled_back_circuit, _, _ = _two_disjoint_dense_two_wire_circuit(
        1, torch.complex128
    )
    rolled_back = rolled_back_circuit.state(refresh=True)
    monkeypatch.setenv("FQ_CPU_DISJOINT_SINGLE_WIRE_FUSION", "1")
    fused_circuit, _, _ = _two_disjoint_dense_two_wire_circuit(1, torch.complex128)

    torch.testing.assert_close(fused_circuit.state(refresh=True), rolled_back)


def test_batched_grouping_reduces_applies_only_at_batch_two_or_more(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FQ_CPU_DISJOINT_SINGLE_WIRE_FUSION", "1")
    plans = {}
    for batch in (1, 2, 4):
        circuit, _, _ = _two_disjoint_dense_two_wire_circuit(batch, torch.complex64)
        circuit.state(refresh=True)
        plans[batch] = (
            circuit._last_statevector_runtime["statevector_apply_count"],
            _disjoint_dense_region_shapes(circuit),
        )

    assert plans[1] == (3, ((2, 1),))
    assert plans[2] == (2, ((2, 2), (1, 1)))
    assert plans[4] == (2, ((2, 2), (1, 1)))


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_batched_grouped_two_wire_regions_match_sequential_state(
    monkeypatch, dtype: torch.dtype
) -> None:
    monkeypatch.setenv("FQ_CPU_DISJOINT_SINGLE_WIRE_FUSION", "0")
    reference_circuit, _, _ = _two_disjoint_dense_two_wire_circuit(2, dtype)
    expected = reference_circuit.state(refresh=True)

    monkeypatch.setenv("FQ_CPU_DISJOINT_SINGLE_WIRE_FUSION", "1")
    fused_circuit, _, _ = _two_disjoint_dense_two_wire_circuit(2, dtype)
    actual = fused_circuit.state(refresh=True)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_batched_grouped_two_wire_regions_preserve_gradients(
    monkeypatch, dtype: torch.dtype
) -> None:
    monkeypatch.setenv("FQ_CPU_DISJOINT_SINGLE_WIRE_FUSION", "0")
    reference_circuit, reference_inputs, reference_angles = (
        _two_disjoint_dense_two_wire_circuit(2, dtype, requires_grad=True)
    )
    expected = reference_circuit.state(refresh=True)
    expected_gradients = torch.autograd.grad(
        expected.real.sum(),
        (reference_inputs, reference_angles),
        retain_graph=True,
    )

    monkeypatch.setenv("FQ_CPU_DISJOINT_SINGLE_WIRE_FUSION", "1")
    fused_circuit, fused_inputs, fused_angles = _two_disjoint_dense_two_wire_circuit(
        2, dtype, requires_grad=True
    )
    actual = fused_circuit.state(refresh=True)
    actual_gradients = torch.autograd.grad(
        actual.real.sum(), (fused_inputs, fused_angles)
    )

    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_batched_program_cache_key_distinguishes_batch_dependent_plans(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FQ_CPU_DISJOINT_SINGLE_WIRE_FUSION", "1")
    circuit, batch_inputs, _ = _two_disjoint_dense_two_wire_circuit(4, torch.complex128)
    circuit.state(refresh=True)
    batched_plan = _disjoint_dense_region_shapes(circuit)

    # Reusing one circuit at a different batch size must not reuse its plan.
    circuit._inputs = batch_inputs[:1].clone()
    circuit.state(refresh=True)
    single_state_plan = _disjoint_dense_region_shapes(circuit)

    assert batched_plan == ((2, 2), (1, 1))
    assert single_state_plan == ((2, 1),)
    assert len(circuit._backend_programs) == 2
    batched_key, single_state_key = circuit._backend_programs
    assert batched_key[-2:] == ("cpu_disjoint_dense_max_two_wire_regions", 2)
    assert single_state_key[-2:] == ("cpu_disjoint_dense_max_two_wire_regions", 1)


def test_batched_two_wire_grouping_rolls_back_with_environment_switch(
    monkeypatch,
) -> None:
    for batch in (2, 4):
        monkeypatch.setenv("FQ_CPU_DISJOINT_SINGLE_WIRE_FUSION", "1")
        fused_circuit, _, _ = _two_disjoint_dense_two_wire_circuit(
            batch, torch.complex64
        )
        fused = fused_circuit.state(refresh=True)
        fused_applies = fused_circuit._last_statevector_runtime[
            "statevector_apply_count"
        ]

        monkeypatch.setenv("FQ_CPU_DISJOINT_SINGLE_WIRE_FUSION", "0")
        rolled_back_circuit, _, _ = _two_disjoint_dense_two_wire_circuit(
            batch, torch.complex64
        )
        rolled_back = rolled_back_circuit.state(refresh=True)
        rolled_back_applies = rolled_back_circuit._last_statevector_runtime[
            "statevector_apply_count"
        ]

        assert fused_applies < rolled_back_applies
        assert rolled_back_applies == 4
        assert _disjoint_dense_region_shapes(rolled_back_circuit) == ()
        torch.testing.assert_close(fused, rolled_back)
