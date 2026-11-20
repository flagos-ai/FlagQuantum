"""Stored triangular indexing composes square/add/XOR with exact cleanup."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.sorting.contiguous_index import (
    contiguous_index_ir,
    contiguous_index_requirements,
)
from flagquantum.compiler.registers.sorting.contiguous_index_component import (
    contiguous_index_program,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_weighted_columns_component import normalized_feedback

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "width,result_width", ((1, 1), (1, 3), (2, 1), (2, 4), (3, 4), (4, 8), (5, 12))
)
@pytest.mark.parametrize("expanded", (False, True))
def test_contiguous_calls_and_primitive_dependencies(
    width: int, result_width: int, expanded: bool
) -> None:
    from qualtran.bloqs.arithmetic.conversions.contiguous_index import ToContiguousIndex
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    graph = contiguous_index_program(width, result_width)
    source = ToContiguousIndex(width, result_width)
    signature = graph.functions[0].registers
    assert signature is not None
    assert [(r.name, r.dtype.quantum_bits, r.side.name) for r in signature] == [
        (r.name, r.dtype.num_qubits, r.side.name) for r in source.signature
    ]
    functions = {function.name: function for function in graph.functions}
    calls = [
        op.callee
        for op in graph.functions[0].blocks[0].operations
        if op.opcode == "call"
        and op.callee is not None
        and not op.callee.startswith("__fq_rebinding_")
    ]
    assert len(calls) == 7
    assert calls[1] == calls[2] and calls[4] == calls[5] and calls[1] != calls[4]
    assert [
        [(r.name, r.side.name) for r in functions[name].registers or ()]
        for name in calls
    ] == [
        [("a", "THRU"), ("result", "RIGHT")],
        [("a", "THRU"), ("b", "THRU")],
        [("a", "THRU"), ("b", "THRU")],
        [("x", "THRU"), ("y", "THRU")],
        [("a", "THRU"), ("b", "THRU")],
        [("a", "THRU"), ("b", "THRU")],
        [("a", "THRU"), ("result", "LEFT")],
    ]
    if expanded:
        graph, _ = expand_component_calls(graph)
    mu, nu, output = (
        tuple(range(width)),
        tuple(range(width, 2 * width)),
        tuple(range(2 * width, 2 * width + result_width)),
    )
    count, slots = contiguous_index_requirements(width)
    work = tuple(range(2 * width + result_width, 2 * width + result_width + count))
    bound = bind_register_program(graph, (mu, nu, output), clean=work)
    reference = contiguous_index_ir(mu, nu, output, work, tuple(range(slots)))
    assert_source(
        normalized_feedback(bound.circuit.instructions), normalized_feedback(reference)
    )
    assert bound.outputs == (mu, nu, output)
    counts = Counter(g.name for g in bound.circuit.instructions)
    events = 2 * (width * (width - 1) + 2 * (2 * width - 1))
    assert counts["t"] + counts["tdg"] == 4 * events
    assert counts["measure"] == counts["reset"] == events
    assert get_cost_value(source, QECGatesCost()).toffoli == width * width + width - 1


@pytest.mark.parametrize("width,result_width", ((1, 2), (2, 3), (3, 6)))
def test_contiguous_full_complex_occupied_results_and_clean_work(
    width: int, result_width: int
) -> None:
    count, _ = contiguous_index_requirements(width)
    start = 2 * width + result_width
    work = tuple(range(start, start + count))
    bound = bind_register_program(
        contiguous_index_program(width, result_width),
        (
            tuple(range(width)),
            tuple(range(width, 2 * width)),
            tuple(range(2 * width, start)),
        ),
        clean=work,
    )
    n = start + count + 1
    before: dict[int, complex] = {}
    after: dict[int, complex] = {}
    for mu, nu, spectator in product(range(1 << width), range(1 << width), (0, 1)):
        occupied = (mu + nu) % (1 << result_width)
        mask = (mu + nu * (nu + 1) // 2) % (1 << result_width)
        prefix = ((mu << width) | nu) << (result_width + count + 1)
        i = prefix | occupied << (count + 1) | spectator
        j = prefix | (occupied ^ mask) << (count + 1) | spectator
        amplitude = complex(mu + 1, nu - 2 * spectator) / 2 ** (width + 1)
        before[i] = amplitude
        after[j] = amplitude
    norm = sum(abs(amplitude) ** 2 for amplitude in before.values()) ** 0.5
    before = {i: amplitude / norm for i, amplitude in before.items()}
    after = {i: amplitude / norm for i, amplitude in after.items()}
    for initial, expected in ((before, after), (after, before)):
        actual = execute_sparse_register(
            CircuitIR(n, bound.circuit.instructions), initial, seed=19
        )
        assert all(
            abs(actual.get(i, 0) - expected.get(i, 0)) < 4e-12
            for i in actual.keys() | expected.keys()
        )
    if width == 1:
        initial_state = torch.zeros(1 << n, dtype=torch.complex128)
        expected_state = torch.zeros_like(initial_state)
        for i, amplitude in before.items():
            initial_state[i] = torch.tensor(amplitude, dtype=torch.complex128)
        for i, amplitude in after.items():
            expected_state[i] = torch.tensor(amplitude, dtype=torch.complex128)
        events = sum(g.name == "measure" for g in bound.circuit.instructions)
        for initial_vector, expected_vector in (
            (initial_state, expected_state),
            (expected_state, initial_state),
        ):
            for history in product((0, 1), repeat=events):
                state = forced_branch(
                    initial_vector, bound.circuit.instructions, history
                )
                torch.testing.assert_close(
                    state, expected_vector / 2 ** (events / 2), atol=3e-12, rtol=0
                )


def test_normal_append_replaces_flat_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    from flagquantum.compiler.registers.arithmetic import integer_square
    from flagquantum.compiler.registers.sorting import contiguous_index

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Stored graph must use shared calls")

    monkeypatch.setattr(integer_square, "integer_square_ir", refuse)
    monkeypatch.setattr(contiguous_index, "contiguous_index_ir", refuse)
    bound = bind_register_program(
        contiguous_index_program(1, 2), ((0,), (1,), (2, 3)), clean=(4, 5, 6)
    )
    initial = torch.zeros(256, dtype=torch.complex128)
    for mu, nu, occupied, spectator in product((0, 1), (0, 1), range(4), (0, 1)):
        initial[(mu << 7) | (nu << 6) | (occupied << 4) | spectator] = torch.tensor(
            complex(mu + 1, nu + spectator), dtype=torch.complex128
        )
    initial /= initial.norm()
    circuit = DynamicCircuit(8, inputs=initial, dtype=torch.complex128)
    operation = FieldOperation(8, bound.circuit.instructions)
    circuit.append(operation)
    circuit.append(operation)
    result = run_dynamic(circuit, shots=4, seed=43, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(4, -1), initial.expand(4, -1), atol=3e-12, rtol=0
    )
