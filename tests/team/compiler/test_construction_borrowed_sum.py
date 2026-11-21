"""Work-free XOR sums preserve arbitrary targets and full complex phases."""

from dataclasses import replace

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler._construction.codegen.arithmetic import (
    ArithmeticCandidate,
    Schedule,
    adjoint_value_candidate,
    schedule_arithmetic,
)
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.compiler.registers._fourier_kernels import borrowed_sum_xor
from flagquantum.core.ir import CircuitIR, Instruction
from tests.team.compiler.test_construction_borrowed_input_bits import candidate

pytestmark = pytest.mark.unit


def _permutation(
    width: int, borrowed_width: int, shift: int, signed: bool
) -> torch.Tensor:
    output_width = borrowed_width + 1
    indices = torch.arange(1 << (width + borrowed_width + output_width))
    x = indices >> (borrowed_width + output_width)
    y = (indices >> output_width) & ((1 << borrowed_width) - 1)
    if signed:
        y = torch.where(y >= 1 << (borrowed_width - 1), y - (1 << borrowed_width), y)
    mask = (1 << output_width) - 1
    return (indices & ~mask) | ((indices & mask) ^ (((x << shift) + y) & mask))


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize(
    "width,borrowed_width,shift",
    [(1, 1, 0), (1, 2, 0), (1, 2, 1), (2, 2, 0), (2, 3, 1)],
)
def test_small_complete_unitaries(
    width: int, borrowed_width: int, shift: int, signed: bool
) -> None:
    size = width + 2 * borrowed_width + 1
    wires = tuple(range(size))
    ir = CircuitIR(
        size,
        tuple(
            borrowed_sum_xor(
                wires[:width],
                wires[width : width + borrowed_width],
                wires[width + borrowed_width :],
                shift=shift,
                signed=signed,
            )
        ),
    )
    count = 1 << size
    actual = (
        fq.Circuit.from_ir(
            ir,
            inputs=torch.eye(count, dtype=torch.complex128),
            bsz=count,
            dtype=torch.complex128,
        )
        .state()
        .reshape(count, count)
    )
    expected = torch.zeros_like(actual)
    expected[
        torch.arange(count), _permutation(width, borrowed_width, shift, signed)
    ] = 1
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


def _tight_candidate(
    width: int,
    signed_width: int,
    strategy: Schedule,
    *,
    expression: str = "x+y",
    max_operations: int = 1000,
) -> ArithmeticCandidate:
    graph = candidate(
        width, signed_width, strategy, optimized=True, expression=expression
    )
    program = graph.lifetimes.program
    block = program.body.blocks[0]
    total = width + signed_width
    output_width = block.operations[-1].operands[0].type.parameters[0]
    return schedule_arithmetic(
        (program,),
        {
            argument.id: (
                tuple(range(width))
                if argument.type.parameters[0] == width
                else tuple(range(width, total))
            )
            for argument in block.arguments
        },
        tuple(range(total, total + output_width)),
        entry=program.name,
        n_qubits=total + output_width,
        workspace_state="fresh_zero",
        scheduling=strategy,
        max_operations=max_operations,
    )


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
@pytest.mark.parametrize("expression", ["x+y", "y+x"])
def test_original_signed_graph_and_its_inverse_on_complex_inputs(
    strategy: Schedule, expression: str
) -> None:
    compiled = _tight_candidate(3, 5, strategy, expression=expression)
    assert compiled.peak_workspace == 0
    assert compiled.circuit.n_wires == 14
    assert len(compiled.circuit.instructions) == 85
    assert compiled.conservative_errors
    assert {rule for _, rule in compiled.implementation_rules} == {
        "terminal_borrowed_fourier_sum"
    }
    inverse = adjoint_value_candidate(compiled)
    for seed in (37, 91):
        initial = torch.randn(
            1 << 14,
            dtype=torch.complex128,
            generator=torch.Generator().manual_seed(seed),
        )
        initial /= initial.norm()
        actual = (
            fq.Circuit.from_ir(compiled.circuit, inputs=initial, dtype=torch.complex128)
            .state()
            .flatten()
        )
        expected = torch.empty_like(initial)
        expected[_permutation(3, 5, 1, True)] = initial
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
        restored = (
            fq.Circuit.from_ir(inverse.circuit, inputs=actual, dtype=torch.complex128)
            .state()
            .flatten()
        )
        torch.testing.assert_close(restored, initial, atol=1e-12, rtol=0)


@pytest.mark.parametrize("x", range(8))
def test_original_all_input_pairs_and_arbitrary_target_columns(x: int) -> None:
    compiled = _tight_candidate(3, 5, "retain")
    gates: list[Instruction] = []
    for gate in compiled.circuit.instructions:
        if any(wire < 3 for wire in gate.wires):
            # The unsigned source is read-only and controls diagonal gates only.
            # Its eight basis blocks are invariant; batching the eleven other
            # qubits verifies every column without a dense fourteen-bit matrix.
            assert gate.name == "cphase" and gate.wires[0] < 3 <= gate.wires[1]
            if x & (1 << (2 - gate.wires[0])):
                gates.append(Instruction("phase", (gate.wires[1] - 3,), gate.params))
        else:
            gates.append(replace(gate, wires=tuple(wire - 3 for wire in gate.wires)))
    ir = CircuitIR(11, tuple(gates))
    indices = torch.arange(2048)
    y = indices >> 6
    signed_y = torch.where(y >= 16, y - 32, y)
    destinations = (indices & ~63) | ((indices & 63) ^ ((2 * x + signed_y) & 63))
    for first in range(0, 2048, 64):
        initial = torch.zeros((64, 2048), dtype=torch.complex128)
        initial[torch.arange(64), torch.arange(first, first + 64)] = 1
        actual = (
            fq.Circuit.from_ir(ir, inputs=initial, bsz=64, dtype=torch.complex128)
            .state()
            .reshape(64, 2048)
        )
        expected = torch.zeros_like(actual)
        expected[torch.arange(64), destinations[first : first + 64]] = 1
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("expression", ["x-y", "x*x+y", "(x+y)+y"])
def test_unproved_graphs_keep_workspace_refusal(expression: str) -> None:
    with pytest.raises(HybridVerificationError, match="lowering.workspace"):
        _tight_candidate(3, 5, "retain", expression=expression)


def test_aliases_keep_the_existing_proved_double_rule() -> None:
    compiled = _tight_candidate(3, 5, "retain", expression="x+x")
    assert compiled.peak_workspace == 0
    assert "terminal_borrowed_fourier_sum" not in {
        rule for _, rule in compiled.implementation_rules
    }


@pytest.mark.parametrize("width,signed_width", [(1, 2), (1, 3)])
def test_other_signed_formats_complete_unitaries(width: int, signed_width: int) -> None:
    compiled = _tight_candidate(width, signed_width, "retain")
    count = 1 << compiled.circuit.n_wires
    actual = (
        fq.Circuit.from_ir(
            compiled.circuit,
            inputs=torch.eye(count, dtype=torch.complex128),
            bsz=count,
            dtype=torch.complex128,
        )
        .state()
        .reshape(count, count)
    )
    expected = torch.zeros_like(actual)
    expected[torch.arange(count), _permutation(width, signed_width, 1, True)] = 1
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


def test_borrowed_sum_operation_budget_is_not_relaxed() -> None:
    assert (
        len(_tight_candidate(3, 5, "retain", max_operations=85).circuit.instructions)
        == 85
    )
    with pytest.raises(HybridVerificationError, match="lowering.operations"):
        _tight_candidate(3, 5, "retain", max_operations=84)


@pytest.mark.parametrize(
    "gate",
    [
        Instruction("rz", (0,), {"theta": 0.3}),
        Instruction("cphase", (0, 1), {"theta": 0.3}, metadata={"unverified": True}),
        Instruction("h", (0,), metadata={"unverified": True}),
    ],
)
def test_inverse_rejects_damaged_fourier_payload(gate: Instruction) -> None:
    original = _tight_candidate(3, 5, "retain")
    with pytest.raises(HybridVerificationError, match="register.adjoint"):
        adjoint_value_candidate(
            replace(original, circuit=replace(original.circuit, instructions=(gate,)))
        )


@pytest.mark.parametrize(
    "source,borrowed,target,shift",
    [
        ((0,), (0,), (1, 2), 0),
        ((0,), (1,), (2,), 0),
        ((0, 1), (2,), (3, 4), 0),
        ((0,), (1,), (2, 3), 1),
        ((-1,), (1,), (2, 3), 0),
    ],
)
def test_invalid_kernel_bindings_reject(
    source: tuple[int, ...],
    borrowed: tuple[int, ...],
    target: tuple[int, ...],
    shift: int,
) -> None:
    with pytest.raises(ValueError):
        tuple(borrowed_sum_xor(source, borrowed, target, shift=shift))
