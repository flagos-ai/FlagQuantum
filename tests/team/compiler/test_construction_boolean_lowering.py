"""Exact XOR, clean workspace and resource checks for the AO-03 candidate."""

from itertools import product

import pytest
import torch

import flagquantum as fq
from benchmarks.synthesis.verification import classical_apply, encode
from flagquantum.compiler._construction.boolean_lowering import (
    BooleanCandidateError,
    lower_boolean_value,
)
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.lowering import schedule_arithmetic
from flagquantum.compiler.resource_estimation import estimate_resources
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def energy(x: int, y: int) -> int:
    return x * x + y


def square(x: int) -> int:
    return x**2


def linear(x: int) -> int:
    return 3 * x + 1


def product_plus(x: int, y: int) -> int:
    return x * y + 1


@pytest.mark.parametrize(
    ("function", "widths", "oracle"),
    [
        (energy, {"x": 2, "y": 2}, lambda x, y: x**2 + y),
        (square, {"x": 2}, lambda x: x**2),
        (square, {"x": 3}, lambda x: x**2),
        (linear, {"x": 2}, lambda x: 3 * x + 1),
        (linear, {"x": 1}, lambda x: 3 * x + 1),
        (product_plus, {"x": 2, "y": 1}, lambda x, y: x * y + 1),
        (product_plus, {"x": 1, "y": 2}, lambda x, y: x * y + 1),
    ],
)
def test_complete_domain_and_arbitrary_xor(function, widths, oracle) -> None:
    graph = capture(function, widths)
    block = graph.body.blocks[0]
    width = block.operations[-1].operands[0].type.parameters[0]
    bindings = {}
    start = 0
    for arg, size in zip(block.arguments, widths.values(), strict=True):
        bindings[arg.id] = tuple(range(start, start + size))
        start += size
    output = tuple(range(start, start + width))
    scratch = (start + width,)
    candidate = lower_boolean_value(
        (graph,),
        bindings,
        output,
        entry=graph.name,
        n_qubits=scratch[0] + 1,
        workspace=scratch,
        workspace_state="fresh_zero",
    )
    old = schedule_arithmetic(
        (graph,),
        bindings,
        output,
        entry=graph.name,
        n_qubits=scratch[0] + 48,
        workspace=tuple(range(scratch[0], scratch[0] + 48)),
        workspace_state="fresh_zero",
    )
    assert len(candidate.circuit.instructions) < len(old.circuit.instructions)
    registers = (*bindings.values(), output)
    for values in product(*(range(1 << w) for w in widths.values())):
        for target in range(1 << width):
            initial = encode((*values, target), registers)
            expected = encode((*values, target ^ oracle(*values)), registers)
            assert classical_apply(candidate.circuit, initial) == expected

    if function is energy:
        prepared = CircuitIR(
            9,
            (
                *(Instruction("h", (wire,)) for wire in range(4)),
                *candidate.circuit.instructions,
            ),
        )
        resources = estimate_resources(prepared)
        assert len(candidate.workspace) == 1
        assert resources.n_operations <= 24 and resources.depth <= 20
        assert prepared.n_wires == 9
        state = torch.zeros(512, dtype=torch.complex128)
        expected = torch.zeros_like(state)
        for x, y, target in product(range(4), range(4), range(16)):
            initial = (x << 7) | (y << 5) | (target << 1)
            final = (x << 7) | (y << 5) | ((target ^ oracle(x, y)) << 1)
            state[initial] = complex(1, (initial % 7) - 3)
            expected[final] = state[initial]
        norm = state.norm()
        result = fq.Circuit.from_ir(
            candidate.circuit, inputs=state / norm, dtype=torch.complex128
        )
        torch.testing.assert_close(
            result.state().reshape(-1), expected / norm, atol=1e-12, rtol=0
        )


def test_budget_and_unavailable_clean_workspace_are_distinct() -> None:
    graph = capture(energy, {"x": 2, "y": 2})
    x, y = graph.body.blocks[0].arguments
    bindings = {x.id: (0, 1), y.id: (2, 3)}
    with pytest.raises(BooleanCandidateError, match="mapping:"):
        lower_boolean_value(
            (graph,), bindings, (4, 5, 6, 7), entry=graph.name, n_qubits=8
        )
    with pytest.raises(BooleanCandidateError, match="budget:"):
        lower_boolean_value(
            (graph,),
            bindings,
            (4, 5, 6, 7),
            entry=graph.name,
            n_qubits=9,
            workspace=(8,),
            workspace_state="fresh_zero",
            max_operations=1,
        )
