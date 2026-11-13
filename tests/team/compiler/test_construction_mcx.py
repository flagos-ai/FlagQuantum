"""Exact basis permutations and coherent borrowed-work restoration for MCX."""

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler._construction.mcx import _dirty_mcx, one_clean_mcx
from flagquantum.core.ir import Instruction

pytestmark = pytest.mark.unit


def apply_permutation(
    width: int, operations: tuple[Instruction, ...], indices: torch.Tensor
) -> torch.Tensor:
    values = indices.clone()
    for op in operations:
        enabled = torch.ones_like(values, dtype=torch.bool)
        for control in op.wires[:-1]:
            enabled &= (values & (1 << (width - 1 - control))) != 0
        values ^= enabled.to(torch.int64) << (width - 1 - op.wires[-1])
    return values


@pytest.mark.parametrize("count", range(3, 16))
def test_one_clean_all_logical_columns_including_nonzero_target(count: int) -> None:
    width = count + 2
    inputs = torch.arange(1 << (width - 1), dtype=torch.int64) << 1
    mask = ((1 << count) - 1) << 2
    expected = inputs ^ (((inputs & mask) == mask).to(torch.int64) << 1)
    instructions = one_clean_mcx(tuple(range(count)), count, count + 1)
    assert torch.equal(apply_permutation(width, instructions, inputs), expected)
    assert all(op.name in {"cx", "ccx"} for op in instructions)


@pytest.mark.parametrize("count", range(3, 8))
def test_dirty_ladder_restores_every_dirty_input_and_target_column(count: int) -> None:
    width = 2 * count - 1
    controls = tuple(range(count))
    target = count
    borrowed = tuple(range(count + 1, width))
    inputs = torch.arange(1 << width, dtype=torch.int64)
    mask = ((1 << count) - 1) << (width - count)
    expected = inputs ^ (
        ((inputs & mask) == mask).to(torch.int64) << (width - count - 1)
    )
    assert torch.equal(
        apply_permutation(width, _dirty_mcx(controls, target, borrowed), inputs),
        expected,
    )


@pytest.mark.parametrize("count", [3, 4, 7])
def test_actual_circuit_preserves_random_complex_phase(count: int) -> None:
    width = count + 2
    generator = torch.Generator().manual_seed(1900 + count)
    amplitudes = torch.randn(
        1 << (width - 1), dtype=torch.complex128, generator=generator
    )
    amplitudes /= torch.linalg.vector_norm(amplitudes)
    state = torch.zeros(1 << width, dtype=torch.complex128)
    state[::2] = amplitudes
    circuit = fq.Circuit(width, dtype=torch.complex128, inputs=state.reshape(1, -1))
    for instruction in one_clean_mcx(tuple(range(count)), count, count + 1):
        circuit.gate(instruction.name, instruction.wires)
    indices = torch.arange(1 << width, dtype=torch.int64)
    mask = ((1 << count) - 1) << 2
    expected = torch.zeros_like(state)
    expected[indices ^ (((indices & mask) == mask).to(torch.int64) << 1)] = state
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "controls,target,flag",
    [((0, 1), 2, 3), ((0, 1, 1), 2, 3), ((0, 1, 2), 2, 3), ((0, 1, 2), 3, 0)],
)
def test_private_lowering_rejects_invalid_borrow_layout(
    controls: tuple[int, ...], target: int, flag: int
) -> None:
    with pytest.raises(ValueError):
        one_clean_mcx(controls, target, flag)
