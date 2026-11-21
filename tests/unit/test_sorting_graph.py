"""Sorting wiring reuses supplied comparison definitions and owned histories."""

from collections.abc import Callable

import pytest

from flagquantum.compiler.registers.comparison.ordering_declaration import (
    ComparatorDeclaration,
)
from flagquantum.compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.sorting.network_graph import (
    bitonic_merge_graph,
    bitonic_sort_graph,
    parallel_comparison_graph,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register

pytestmark = pytest.mark.unit


def _history_comparator(flip: bool) -> ProgramRecord:
    signature = ComparatorDeclaration(2).signature
    bit = quantum_bit_type()
    x = bit_gate_function(
        "history_x", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
    )
    builder = ComponentProgramBuilder(signature, (x,))
    history: BindingCollection = builder.allocate(dtype=bit)
    if flip:
        history = builder.call_dict(x.name, {"q": history})["q"]
    return builder.finalize({**builder.inputs, "out": history})


@pytest.mark.parametrize(
    "factory,words,histories",
    (
        (lambda comparator: parallel_comparison_graph(5, 2, comparator), 5, 2),
        (lambda comparator: bitonic_merge_graph(2, comparator), 4, 4),
        (lambda comparator: bitonic_sort_graph(4, comparator), 4, 6),
    ),
)
def test_comparator_replacement_changes_every_history_without_changing_consumers(
    factory: Callable[[ProgramRecord], ProgramRecord], words: int, histories: int
) -> None:
    for flip in (False, True):
        record = factory(_history_comparator(flip))
        bound = bind_register_program(
            record,
            tuple((2 * i, 2 * i + 1) for i in range(words)),
            clean=tuple(range(2 * words, 2 * words + histories)),
            allow_workspace_outputs=True,
        )
        assert len(bound.outputs) == words + histories
        assert tuple(g.name for g in bound.circuit.instructions) == ("x",) * (
            histories if flip else 0
        )
        assert set(g.wires[0] for g in bound.circuit.instructions) == (
            set(range(2 * words, 2 * words + histories)) if flip else set()
        )
