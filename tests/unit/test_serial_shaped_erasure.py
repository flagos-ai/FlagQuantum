"""Sequential shaped calls retain separate measurement events on reused wires."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler._component_trace.program_call_graph import (
    program_component_call_graph,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_workspace import LifetimeEvent, source_workspace_gates
from tests.unit.test_component_control_selection import assert_gate_parameters
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_structured_control import normalize_trace
from tests.unit.test_controlled_workspace_lifetimes import PATTERNS
from tests.unit.test_shaped_controlled_erasure import (
    initial_and_expected,
    shaped_fixture,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ORDERS = ((0, 1), (1, 0), (0, 0))


def serial_fixture(
    values: tuple[int, ...],
    dirty: bool,
    order: tuple[int, ...],
    inverse: bool = False,
    free_dirty: bool | None = None,
) -> tuple[ProgramRecord, Any]:
    first, source_first, _ = shaped_fixture(
        values, dirty, (1, 2), order[0], inverse, free_dirty
    )
    second, source_second, _ = shaped_fixture(
        values, dirty, (1, 2), order[1], inverse, free_dirty
    )
    assert first.functions[1:] == second.functions[1:]
    root = first.functions[0]
    assert root.registers is not None
    signature = Signature(root.registers)
    definitions = (
        replace(root, name="first"),
        replace(second.functions[0], name="second"),
        *first.functions[1:],
    )
    builder = ComponentProgramBuilder(signature, definitions, name="serial")
    observer = _source(signature)
    names = ("first", "first") if order[0] == order[1] else ("first", "second")
    if order[0] == order[1]:
        source_second = source_first
    data = dict(builder.inputs)
    src_data = dict(observer.inputs)
    for name, source in zip(names, (source_first, source_second), strict=True):
        data = builder.call_dict(name, data)
        src_data = observer.builder.add_d(source, **src_data)
    return builder.finalize(data), observer.builder.finalize(**src_data)


def bind_serial(record: ProgramRecord, n: int, dirty: bool) -> BoundRegisterProgram:
    life = 2 * n + 3
    return bind_register_program(
        record,
        (*((i,) for i in range(n)), (n, n + 1), (n + 2, n + 3)),
        clean=tuple(w for w in range(n + 4, 2 * n + 5) if not dirty or w != life),
        borrowed=(life,) if dirty else (),
        entry="serial",
    )


def reference_serial(
    source: Any, n: int, dirty: bool
) -> tuple[tuple[Instruction, ...], tuple[LifetimeEvent, ...]]:
    life = 2 * n + 3
    return source_workspace_gates(
        source,
        {"ctrl": tuple(range(n)), "data": tuple(range(n, n + 4))},
        tuple(w for w in range(n + 4, 2 * n + 5) if not dirty or w != life),
        (life,) if dirty else (),
    )


def expected_sequence(
    state: torch.Tensor, values: tuple[int, ...], order: tuple[int, ...]
) -> torch.Tensor:
    n = len(values)
    width = state.numel().bit_length() - 1
    indexes = torch.arange(state.numel())
    pattern = sum(value << (n - 1 - i) for i, value in enumerate(values))
    active = indexes >> (width - n) == pattern
    for selected in order:
        mask = 1 << (width - 1 - n - 2 * selected)
        bit = (indexes & mask) != 0
        hadamard = (
            state[indexes & ~mask] + torch.where(bit, -1, 1) * state[indexes | mask]
        ) / 2**0.5
        phase = torch.exp(1j * np.pi / 4 * bit.to(torch.float64))
        state = torch.where(active, hadamard * phase, state)
    return state


@pytest.mark.parametrize("values", PATTERNS)
@pytest.mark.parametrize("dirty", (False, True))
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("expanded", (False, True))
def test_serial_calls_reuse_work_but_preserve_event_slots_and_all_branches(
    values: tuple[int, ...],
    dirty: bool,
    order: tuple[int, ...],
    expanded: bool,
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    record, source = serial_fixture(values, dirty, order)
    assert normalize_trace(_native_trace(record, "serial")) == normalize_trace(
        _source_trace(_source(Signature([])), source)
    )
    terminals = {
        op.callee: instance.bloq
        for op, (instance, _, _) in zip(
            record.functions[0].blocks[0].operations,
            source.iter_bloqsoqs(),
            strict=True,
        )
        if op.callee is not None
    }
    graph = program_component_call_graph(record, "serial", keep=terminals.__contains__)
    weighted: Counter[Any] = Counter()
    for name, count in graph.leaf_counts.items():
        weighted[terminals[name]] += count
    assert weighted == source.bloq_counts()
    if expanded:
        record, _ = expand_component_calls(record, "serial")
    n = len(values)
    bound = bind_serial(record, n, dirty)
    original, events = reference_serial(source, n, dirty)
    assert_gate_parameters(bound.circuit.instructions, original)
    assert tuple((e.kind, e.wires, e.dirty) for e in bound.events if e.wires) == events
    measured = [g for g in bound.circuit.instructions if g.name == "measure"]
    assert [g.metadata["classical_bit"] for g in measured] == list(range(2 * n))
    assert [g.wires for g in measured[:n]] == [g.wires for g in measured[n:]]
    assert bound.outputs == (*((i,) for i in range(n)), (n, n + 1), (n + 2, n + 3))
    initial, _ = initial_and_expected(n, dirty, values, 0)
    expected = expected_sequence(initial, values, order)
    for history in product((0, 1), repeat=2 * n):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        upstream = forced_branch(initial, original, history)
        torch.testing.assert_close(actual, upstream, atol=4e-12, rtol=0)
        torch.testing.assert_close(actual, expected / 2**n, atol=4e-12, rtol=0)
    circuit = DynamicCircuit(2 * n + 5, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(2 * n + 5, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=8, seed=67, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), expected.expand(8, -1), atol=4e-12, rtol=0
    )
    model = get_cost_value(source, QECGatesCost())
    assert model.total_t_count() == 2 * (35 + 4 * (n - 1))
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts["ccx"] == 4 and counts["t"] + counts["tdg"] == 8 * n
    assert counts["measure"] == counts["reset"] == 2 * n
    assert get_cost_value(source, QubitCount()) == bound.circuit.n_wires == 2 * n + 5
    for category in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert getattr(model, category) == sum(
            count * getattr(get_cost_value(terminals[name], QECGatesCost()), category)
            for name, count in graph.leaf_counts.items()
        )


@pytest.mark.parametrize("value", (0, 1))
@pytest.mark.parametrize("dirty", (False, True))
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("expanded", (False, True))
def test_serial_exact_inverse_restores_every_history_pair(
    value: int,
    dirty: bool,
    order: tuple[int, ...],
    expanded: bool,
) -> None:
    forward, _ = serial_fixture((value,), dirty, order)
    inverse, source = serial_fixture(
        (value,), dirty, tuple(reversed(order)), inverse=True
    )
    if expanded:
        inverse, _ = expand_component_calls(inverse, "serial")
    computed = bind_serial(forward, 1, dirty)
    reversed_bound = bind_serial(inverse, 1, dirty)
    original, events = reference_serial(source, 1, dirty)
    assert_gate_parameters(reversed_bound.circuit.instructions, original)
    assert (
        tuple((e.kind, e.wires, e.dirty) for e in reversed_bound.events if e.wires)
        == events
    )
    initial, _ = initial_and_expected(1, dirty, (value,), 0)
    for first in product((0, 1), repeat=2):
        intermediate = forced_branch(initial, computed.circuit.instructions, first)
        for second in product((0, 1), repeat=2):
            actual = forced_branch(
                intermediate, reversed_bound.circuit.instructions, second
            )
            upstream = forced_branch(intermediate, original, second)
            torch.testing.assert_close(actual, upstream, atol=4e-12, rtol=0)
            torch.testing.assert_close(actual, initial / 4, atol=4e-12, rtol=0)


@pytest.mark.parametrize("values", tuple(product((0, 1), repeat=2)))
@pytest.mark.parametrize("dirty", (False, True))
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("expanded", (False, True))
def test_two_control_serial_exact_inverse_all_history_pairs(
    values: tuple[int, ...],
    dirty: bool,
    order: tuple[int, ...],
    expanded: bool,
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    forward, source_forward = serial_fixture(values, dirty, order)
    inverse, source_inverse = serial_fixture(
        values, dirty, tuple(reversed(order)), inverse=True
    )
    observer = _source(Signature([]))
    assert normalize_trace(_native_trace(inverse, "serial")) == normalize_trace(
        _source_trace(observer, source_inverse)
    )
    terminals = {
        op.callee: instance.bloq
        for op, (instance, _, _) in zip(
            inverse.functions[0].blocks[0].operations,
            source_inverse.iter_bloqsoqs(),
            strict=True,
        )
        if op.callee is not None
    }
    graph = program_component_call_graph(inverse, "serial", keep=terminals.__contains__)
    weighted: Counter[Any] = Counter()
    for name, count in graph.leaf_counts.items():
        weighted[terminals[name]] += count
    assert weighted == source_inverse.bloq_counts()
    if expanded:
        forward, _ = expand_component_calls(forward, "serial")
        inverse, _ = expand_component_calls(inverse, "serial")
    computed = bind_serial(forward, 2, dirty)
    backwards = bind_serial(inverse, 2, dirty)
    original_forward, forward_events = reference_serial(source_forward, 2, dirty)
    original_inverse, inverse_events = reference_serial(source_inverse, 2, dirty)
    assert_gate_parameters(computed.circuit.instructions, original_forward)
    assert_gate_parameters(backwards.circuit.instructions, original_inverse)
    assert (
        tuple((e.kind, e.wires, e.dirty) for e in computed.events if e.wires)
        == forward_events
    )
    assert (
        tuple((e.kind, e.wires, e.dirty) for e in backwards.events if e.wires)
        == inverse_events
    )
    assert computed.outputs == backwards.outputs == ((0,), (1,), (2, 3), (4, 5))
    for gates in (computed.circuit.instructions, backwards.circuit.instructions):
        measured = [g for g in gates if g.name == "measure"]
        assert [g.metadata["classical_bit"] for g in measured] == [0, 1, 2, 3]
        assert [g.wires for g in measured] == [(8,), (6,), (8,), (6,)]
    initial, _ = initial_and_expected(2, dirty, values, 0)
    expected = expected_sequence(initial, values, order)
    for first in product((0, 1), repeat=4):
        intermediate = forced_branch(initial, computed.circuit.instructions, first)
        upstream_forward = forced_branch(initial, original_forward, first)
        torch.testing.assert_close(intermediate, upstream_forward, atol=4e-12, rtol=0)
        torch.testing.assert_close(intermediate, expected / 4, atol=4e-12, rtol=0)
        for second in product((0, 1), repeat=4):
            actual = forced_branch(intermediate, backwards.circuit.instructions, second)
            upstream = forced_branch(upstream_forward, original_inverse, second)
            torch.testing.assert_close(actual, upstream, atol=4e-12, rtol=0)
            torch.testing.assert_close(actual, initial / 16, atol=4e-12, rtol=0)
            for wire in ((6, 8) if dirty else (6, 7, 8)):
                assert (
                    torch.count_nonzero(actual.reshape((2,) * 9).select(wire, 1)) == 0
                )
    # Compose both operations into one execution so all eight event slots must
    # be rebound by the existing user-facing append path.
    circuit = DynamicCircuit(9, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(9, computed.circuit.instructions))
    circuit.append(FieldOperation(9, backwards.circuit.instructions))
    result = run_dynamic(circuit, shots=8, seed=71, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), initial.expand(8, -1), atol=4e-12, rtol=0
    )
    model = get_cost_value(source_inverse, QECGatesCost())
    assert (
        model.total_t_count() == 78
        and get_cost_value(source_inverse, QubitCount()) == 9
    )
    counts = Counter(g.name for g in backwards.circuit.instructions)
    assert counts["ccx"] == 4 and counts["t"] + counts["tdg"] == 16
    assert counts["measure"] == counts["reset"] == 4
    for category in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert getattr(model, category) == sum(
            count * getattr(get_cost_value(terminals[name], QECGatesCost()), category)
            for name, count in graph.leaf_counts.items()
        )
