"""Serial clean and borrowed calls preserve separate pool ownership."""

from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler._component_trace.program_call_graph import (
    program_component_call_graph,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.component_graph import _namespace_helpers
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_workspace import source_workspace_gates
from tests.unit.test_component_control_selection import assert_gate_parameters
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_structured_control import normalize_trace
from tests.unit.test_controlled_workspace_lifetimes import PATTERNS
from tests.unit.test_serial_shaped_erasure import expected_sequence
from tests.unit.test_shaped_controlled_erasure import shaped_fixture

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def cross_pool_fixture(
    values: tuple[int, ...], first_dirty: bool, inverse: bool = False
) -> tuple[ProgramRecord, Any]:
    calls: tuple[tuple[int, bool], ...] = ((0, first_dirty), (1, not first_dirty))
    if inverse:
        calls = tuple(reversed(calls))
    definitions: list[FunctionRecord] = []
    sources = []
    for position, (selected, dirty) in enumerate(calls):
        record, source, _ = shaped_fixture(values, dirty, (1, 2), selected, inverse)
        namespaced = _namespace_helpers(record, f"call{position}")
        definitions.extend(namespaced.functions)
        sources.append(source)
    registers = definitions[0].registers
    assert registers is not None
    signature = Signature(registers)
    builder = ComponentProgramBuilder(signature, definitions, name="cross_pool")
    observer = _source(signature)
    data = dict(builder.inputs)
    src_data = dict(observer.inputs)
    for position, source in enumerate(sources):
        data = builder.call_dict(f"call{position}_shaped", data)
        src_data = observer.builder.add_d(source, **src_data)
    return builder.finalize(data), observer.builder.finalize(**src_data)


def bind_cross_pool(record: ProgramRecord, n: int) -> BoundRegisterProgram:
    return bind_register_program(
        record,
        (*((i,) for i in range(n)), (n, n + 1), (n + 2, n + 3)),
        clean=tuple(range(n + 4, 2 * n + 5)),
        borrowed=(2 * n + 5,),
        entry="cross_pool",
    )


@pytest.mark.parametrize("values", PATTERNS)
@pytest.mark.parametrize("first_dirty", (False, True))
@pytest.mark.parametrize("expanded", (False, True))
def test_calls_with_different_allocation_flags_keep_pools_and_exact_inverse(
    values: tuple[int, ...],
    first_dirty: bool,
    expanded: bool,
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    n = len(values)
    width = 2 * n + 6
    records = [
        cross_pool_fixture(values, first_dirty, inverse) for inverse in (False, True)
    ]
    bounds = []
    source_gates = []
    for record, source in records:
        assert normalize_trace(_native_trace(record, "cross_pool")) == normalize_trace(
            _source_trace(_source(Signature([])), source)
        )
        terminals = {
            op.callee: instance.bloq
            for op, (instance, _, _) in zip(
                record.functions[0].blocks[0].operations,
                source.iter_bloqsoqs(),
                strict=True,
            )
        }
        graph = program_component_call_graph(
            record, "cross_pool", keep=terminals.__contains__
        )
        assert (
            Counter(
                {terminals[name]: count for name, count in graph.leaf_counts.items()}
            )
            == source.bloq_counts()
        )
        if expanded:
            record, _ = expand_component_calls(record, "cross_pool")
        bound = bind_cross_pool(record, n)
        gates, events = source_workspace_gates(
            source,
            {"ctrl": tuple(range(n)), "data": tuple(range(n, n + 4))},
            tuple(range(n + 4, 2 * n + 5)),
            (width - 1,),
        )
        assert_gate_parameters(bound.circuit.instructions, gates)
        assert (
            tuple((e.kind, e.wires, e.dirty) for e in bound.events if e.wires) == events
        )
        assert [(kind, wires) for kind, wires, dirty in events if dirty] == [
            ("allocate", (width - 1,)),
            ("free", (width - 1,)),
        ]
        assert all(width - 1 not in wires for _, wires, dirty in events if not dirty)
        assert bound.outputs == (*((i,) for i in range(n)), (n, n + 1), (n + 2, n + 3))
        assert [
            g.metadata["classical_bit"] for g in gates if g.name == "measure"
        ] == list(range(2 * n))
        model = get_cost_value(source, QECGatesCost())
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
                count
                * getattr(get_cost_value(terminals[name], QECGatesCost()), category)
                for name, count in graph.leaf_counts.items()
            )
        counts = Counter(g.name for g in gates)
        assert counts["t"] + counts["tdg"] == 8 * n and counts["ccx"] == 4
        assert counts["measure"] == counts["reset"] == 2 * n
        assert model.total_t_count() == 2 * (35 + 4 * (n - 1))
        # One extra caller pool wire is necessary to offer both clean and
        # borrowed choices. The logical source peak does not count that pool.
        assert get_cost_value(source, QubitCount()) == width - 1
        assert bound.circuit.n_wires == width
        bounds.append(bound)
        source_gates.append(gates)
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    for index in range(initial.numel()):
        if any(index & (1 << (width - 1 - wire)) for wire in range(n + 4, width - 1)):
            continue
        initial[index] = torch.tensor(
            complex(index + 1, 3 - index), dtype=torch.complex128
        )
    initial /= initial.norm()
    expected = expected_sequence(initial, values, (0, 1))
    histories = tuple(product((0, 1), repeat=2 * n))
    for first in histories:
        intermediate = forced_branch(initial, bounds[0].circuit.instructions, first)
        upstream_forward = forced_branch(initial, source_gates[0], first)
        torch.testing.assert_close(intermediate, upstream_forward, atol=4e-12, rtol=0)
        torch.testing.assert_close(intermediate, expected / 2**n, atol=4e-12, rtol=0)
        for second in histories:
            actual = forced_branch(intermediate, bounds[1].circuit.instructions, second)
            upstream = forced_branch(upstream_forward, source_gates[1], second)
            torch.testing.assert_close(actual, upstream, atol=4e-12, rtol=0)
            torch.testing.assert_close(
                actual, initial / 2 ** (2 * n), atol=4e-12, rtol=0
            )
    circuit = DynamicCircuit(width, inputs=initial, dtype=torch.complex128)
    for bound in bounds:
        circuit.append(FieldOperation(width, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=8, seed=83, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), initial.expand(8, -1), atol=4e-12, rtol=0
    )
