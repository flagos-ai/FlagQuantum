"""Independent predicate and complex-state checks for one-carry comparisons."""

from collections.abc import Sequence
from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler._arithmetic_kernels import borrowed_less_xor
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.lowering import schedule_arithmetic
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.core.ir import Instruction

pytestmark = pytest.mark.unit


def execute(gates: Sequence[Instruction], ones: set[int]) -> set[int]:
    state = set(ones)
    for gate in gates:
        assert gate.name in {"x", "cx", "ccx"}
        if all(wire in state for wire in gate.wires[:-1]):
            state.symmetric_difference_update((gate.wires[-1],))
    return state


@pytest.mark.parametrize("width", range(1, 6))
def test_all_words_targets_and_inverse(width: int) -> None:
    a = tuple(range(width))
    b = tuple(range(width, 2 * width))
    target, carry = 2 * width, 2 * width + 1
    gates = borrowed_less_xor(a, b, target, carry)
    for left in range(1 << width):
        for right in range(1 << width):
            inputs = {
                wire
                for value, ports in ((left, a), (right, b))
                for index, wire in enumerate(ports)
                if value >> (width - 1 - index) & 1
            }
            for marker in (0, 1):
                initial = inputs | ({target} if marker else set())
                expected = inputs | ({target} if marker ^ (left < right) else set())
                assert execute(gates, initial) == expected
                assert execute(tuple(reversed(gates)), expected) == initial


def test_complex_coherent_target_and_spectator_preserve_absolute_phase() -> None:
    generator = torch.Generator().manual_seed(541)
    logical = torch.randn(256, generator=generator, dtype=torch.complex128)
    logical /= logical.norm()
    initial = torch.zeros((128, 2, 2), dtype=torch.complex128)
    initial[:, 0, :] = logical.reshape(128, 2)
    initial = initial.flatten()
    gates = borrowed_less_xor((0, 1, 2), (3, 4, 5), 6, 7)
    circuit = fq.Circuit(9, dtype=torch.complex128, inputs=initial)
    for gate in gates:
        getattr(circuit, gate.name)(*gate.wires)
    expected = torch.zeros_like(initial)
    for left in range(8):
        for right in range(8):
            for marker in (0, 1):
                for spectator in (0, 1):
                    source = 4 * (2 * (8 * left + right) + marker) + spectator
                    dest = (
                        4 * (2 * (8 * left + right) + (marker ^ (left < right)))
                        + spectator
                    )
                    expected[dest] = initial[source]
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=1e-12, rtol=0)
    for gate in reversed(gates):
        getattr(circuit, gate.name)(*gate.wires)
    torch.testing.assert_close(circuit.state().flatten(), initial, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "a,b,target,carry,error",
    [
        ((), (1,), 2, 3, ValueError),
        ((0,), (1, 2), 3, 4, ValueError),
        ((0, 0), (1, 2), 3, 4, ValueError),
        ((0, 1), (1, 2), 3, 4, ValueError),
        ((0,), (1,), 0, 2, ValueError),
        ((0,), (1,), 2, 0, ValueError),
        ((0,), (1,), 2, 2, ValueError),
        ((-1,), (1,), 2, 3, ValueError),
        ((False,), (1,), 2, 3, TypeError),
        ((0,), (1,), True, 3, TypeError),
        ((0,), (1,), 2, 3.0, TypeError),
    ],
)
def test_invalid_borrowed_bindings(
    a: tuple[int, ...],
    b: tuple[int, ...],
    target: int,
    carry: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        borrowed_less_xor(a, b, target, carry)


def less(x: int, y: int) -> bool:
    return x < y


@pytest.mark.parametrize("scheduling", ["retain", "recompute"])
@pytest.mark.parametrize("mode", ["value", "phase"])
def test_scheduler_uses_one_clean_carry_and_preserves_coherent_outputs(
    scheduling: Literal["retain", "recompute"], mode: Literal["value", "phase"]
) -> None:
    graph = capture(less, {"x": 3, "y": 3})
    arguments = graph.body.blocks[0].arguments
    inputs = {arguments[0].id: (0, 1, 2), arguments[1].id: (3, 4, 5)}
    output = (6,) if mode == "value" else ()
    work_start = 6 + len(output)
    workspace = (work_start, work_start + 1)
    candidate = schedule_arithmetic(
        (graph,),
        inputs,
        output,
        entry=graph.name,
        n_qubits=work_start + 2,
        workspace=workspace,
        workspace_state="fresh_zero",
        scheduling=scheduling,
        output_mode=mode,
    )
    assert candidate.peak_workspace == 2
    assert max(len(event.scratch) for event in candidate.events) == 1
    generator = torch.Generator().manual_seed(739)
    logical = torch.randn(1 << work_start, generator=generator, dtype=torch.complex128)
    logical /= logical.norm()
    initial = torch.zeros(1 << (work_start + 2), dtype=torch.complex128)
    initial[::4] = logical
    circuit = fq.Circuit(work_start + 2, dtype=torch.complex128, inputs=initial)
    for gate in candidate.circuit.instructions:
        getattr(circuit, gate.name)(*gate.wires)
    expected = torch.zeros_like(initial)
    for x in range(8):
        for y in range(8):
            if mode == "value":
                for target in (0, 1):
                    source = 4 * (2 * (8 * x + y) + target)
                    destination = 4 * (2 * (8 * x + y) + (target ^ (x < y)))
                    expected[destination] = initial[source]
            else:
                index = 4 * (8 * x + y)
                expected[index] = (-1 if x < y else 1) * initial[index]
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=1e-12, rtol=0)
    with pytest.raises(HybridVerificationError, match="workspace"):
        schedule_arithmetic(
            (graph,),
            inputs,
            output,
            entry=graph.name,
            n_qubits=work_start + 1,
            workspace=workspace[:1],
            workspace_state="fresh_zero",
            scheduling=scheduling,
            output_mode=mode,
        )
    with pytest.raises(HybridVerificationError, match="operations"):
        schedule_arithmetic(
            (graph,),
            inputs,
            output,
            entry=graph.name,
            n_qubits=work_start + 2,
            workspace=workspace,
            workspace_state="fresh_zero",
            scheduling=scheduling,
            output_mode=mode,
            max_operations=len(candidate.circuit.instructions) - 1,
        )
