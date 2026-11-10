"""Shaped views compose with controlled computation and measured cleanup."""

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
from flagquantum.compiler.registers.component_adjoint import adjoint_component_hierarchy
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.layout.component_cast import cast_word_function
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_component_control_selection import assert_gate_parameters
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_hierarchy_adjoint import source_adjoint
from tests.unit.test_component_structured_control import normalize_trace
from tests.unit.test_controlled_workspace_lifetimes import (
    PATTERNS,
    controlled_fixture,
    lifetime_source_gates,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
LAYOUTS = (((2,), 0), ((1, 2), 1))


def shaped_fixture(
    values: tuple[int, ...],
    dirty: bool,
    shape: tuple[int, ...],
    selected: int,
    inverse: bool = False,
) -> tuple[ProgramRecord, Any, Any]:
    """Store real calls around one existing controlled lifetime computation."""
    from qualtran import QAny, QUInt
    from qualtran.bloqs.bookkeeping import Cast

    child, source_child = controlled_fixture(values, dirty, dirty)
    if inverse:
        adjoints = {
            "ch": "ch",
            "cp": "cp_inverse",
            "ccnot": "ccnot",
            "allocate": "allocate_adjoint",
            "free": "free_adjoint",
        }
        adjoints.update(
            {f.name: f.name for f in child.functions if f.name.endswith("_x")}
        )
        child = adjoint_component_hierarchy(child, adjoints)
        source_child = source_adjoint(source_child.decompose_bloq())
    n = len(values)
    opaque, uint = RegisterType("opaque", 2), RegisterType("uint", 2)
    signature = Signature(
        (
            Register("ctrl", RegisterType("bit", 1), (n,) if n > 1 else ()),
            Register("data", opaque, shape),
        )
    )
    definitions = (
        *child.functions,
        cast_word_function("view", opaque, uint, shape),
        cast_word_function("unview", uint, opaque, shape),
    )
    builder = ComponentProgramBuilder(signature, definitions, name="shaped")
    observer = _source(signature)
    data = np.asarray(
        builder.call_dict("view", {"reg": builder.inputs["data"]})["reg"], dtype=object
    ).copy()
    src_data = np.asarray(
        observer.builder.add(
            Cast(QAny(2), QUInt(2), shape), reg=observer.inputs["data"]
        ),
        dtype=object,
    ).copy()
    bits = builder.split(data.flat[selected])
    src_bits = observer.builder.split(src_data.flat[selected])
    outputs = builder.call_dict(
        "main", {"ctrl": builder.inputs["ctrl"], "q": bits[0], "spectator": bits[1]}
    )
    src_outputs = observer.builder.add_d(
        source_child, ctrl=observer.inputs["ctrl"], q=src_bits[0], spectator=src_bits[1]
    )
    data.flat[selected] = builder.join([outputs["q"], outputs["spectator"]], uint)
    src_data.flat[selected] = observer.builder.join(
        [src_outputs["q"], src_outputs["spectator"]], dtype=QUInt(2)
    )
    returned = builder.call_dict("unview", {"reg": data})["reg"]
    src_returned = observer.builder.add(Cast(QUInt(2), QAny(2), shape), reg=src_data)
    return (
        builder.finalize({"ctrl": outputs["ctrl"], "data": returned}),
        observer.builder.finalize(ctrl=src_outputs["ctrl"], data=src_returned),
        source_child,
    )


def bind_shaped(record: ProgramRecord, n: int, dirty: bool) -> BoundRegisterProgram:
    life, phase = 2 * n + 3, 2 * n + 4
    return bind_register_program(
        record,
        (*((i,) for i in range(n)), (n, n + 1), (n + 2, n + 3)),
        clean=tuple(w for w in range(n + 4, phase + 1) if not dirty or w != life),
        borrowed=(life,) if dirty else (),
        entry="shaped",
    )


def original_shaped_gates(
    source: Any,
    child: Any,
    n: int,
    native_gates: tuple[Instruction, ...],
) -> tuple[tuple[Instruction, ...], tuple[tuple[str, bool], ...]]:
    """Read original outer Cast/Split/Join and the unmodified controlled child."""
    ports = {"ctrl": tuple(range(n)), "data": tuple(range(n, n + 4))}
    measurements = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in native_gates
        if g.name == "measure"
    }
    events: list[tuple[str, bool]] = []

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if bloq is not child:
            return None
        mapping = (
            *inputs["ctrl"],
            *inputs["q"],
            *inputs["spectator"],
            *range(n + 4, 2 * n + 5),
        )
        bits = {i: measurements[w] for i, w in enumerate(mapping) if w in measurements}
        gates, lifetime = lifetime_source_gates(child, n, bits)
        events.extend(lifetime)
        return (
            tuple(replace(g, wires=tuple(mapping[w] for w in g.wires)) for g in gates),
            inputs,
        )

    gates, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    return gates, tuple(events)


def initial_and_expected(
    n: int, dirty: bool, values: tuple[int, ...], selected: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply independent controlled H then T to the selected MSB only."""
    width = 2 * n + 5
    life = 2 * n + 3
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    for index in range(initial.numel()):
        if any(
            index & (1 << (width - 1 - wire))
            for wire in range(n + 4, width)
            if not dirty or wire != life
        ):
            continue
        initial[index] = torch.tensor(
            complex(index + 1, 3 - index), dtype=torch.complex128
        )
    initial /= initial.norm()
    expected = torch.zeros_like(initial)
    mask = 1 << (width - 1 - n - 2 * selected)
    for index in range(initial.numel()):
        controls = tuple((index >> (width - 1 - i)) & 1 for i in range(n))
        if controls != values:
            expected[index] += initial[index]
            continue
        for output in (0, 1):
            target = index & ~mask | (mask if output else 0)
            expected[target] += (
                initial[index]
                * (-1 if index & mask and output else 1)
                / 2**0.5
                * np.exp(1j * np.pi * output / 4)
            )
    return initial, expected


@pytest.mark.parametrize("values", PATTERNS)
@pytest.mark.parametrize("dirty", (False, True))
@pytest.mark.parametrize("shape,selected", LAYOUTS)
@pytest.mark.parametrize("expanded", (False, True))
def test_shaped_controlled_compute_erasure_all_histories(
    values: tuple[int, ...],
    dirty: bool,
    shape: tuple[int, ...],
    selected: int,
    expanded: bool,
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    module, source, child = shaped_fixture(values, dirty, shape, selected)
    assert normalize_trace(_native_trace(module, "shaped")) == normalize_trace(
        _source_trace(_source(Signature([])), source)
    )
    root = module.functions[0]
    terminals = {
        op.callee: instance.bloq
        for op, (instance, _, _) in zip(
            root.blocks[0].operations, source.iter_bloqsoqs(), strict=True
        )
        if op.callee is not None
    }
    graph = program_component_call_graph(
        module, entry="shaped", keep=terminals.__contains__
    )
    assert (
        Counter({terminals[name]: count for name, count in graph.leaf_counts.items()})
        == source.bloq_counts()
    )
    if expanded:
        module, _ = expand_component_calls(module, "shaped")
    n = len(values)
    bound = bind_shaped(module, n, dirty)
    assert bound.outputs == (*((i,) for i in range(n)), (n, n + 1), (n + 2, n + 3))
    original, events = original_shaped_gates(
        source, child, n, bound.circuit.instructions
    )
    assert_gate_parameters(bound.circuit.instructions, original)
    assert (
        tuple((e.kind, e.dirty) for e in bound.events if e.wires == (2 * n + 3,))
        == events
        == (("allocate", dirty), ("free", dirty))
    )
    initial, expected = initial_and_expected(n, dirty, values, selected)
    for history in product((0, 1), repeat=n):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        upstream = forced_branch(initial, original, history)
        torch.testing.assert_close(actual, upstream, atol=3e-12, rtol=0)
        torch.testing.assert_close(actual, expected / 2 ** (n / 2), atol=3e-12, rtol=0)
        for wire in range(n + 4, 2 * n + 5):
            if dirty and wire == 2 * n + 3:
                continue
            assert (
                torch.count_nonzero(actual.reshape((2,) * (2 * n + 5)).select(wire, 1))
                == 0
            )
    circuit = DynamicCircuit(2 * n + 5, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(2 * n + 5, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=8, seed=53, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), expected.expand(8, -1), atol=3e-12, rtol=0
    )
    counts = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["ccx"] == 2 and counts["t"] + counts["tdg"] == 4 * n
    assert counts["measure"] == counts["reset"] == n
    assert model.total_t_count() == 35 + 4 * (n - 1)
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
@pytest.mark.parametrize("shape,selected", LAYOUTS)
@pytest.mark.parametrize("expanded", (False, True))
def test_explicit_shaped_inverse_restores_all_measurement_history_pairs(
    value: int,
    dirty: bool,
    shape: tuple[int, ...],
    selected: int,
    expanded: bool,
) -> None:
    record, _, _ = shaped_fixture((value,), dirty, shape, selected)
    inverse, source, child = shaped_fixture(
        (value,), dirty, shape, selected, inverse=True
    )
    assert normalize_trace(_native_trace(inverse, "shaped")) == normalize_trace(
        _source_trace(_source(Signature([])), source)
    )
    if expanded:
        inverse, _ = expand_component_calls(inverse, "shaped")
    forward = bind_shaped(record, 1, dirty)
    backward = bind_shaped(inverse, 1, dirty)
    original, _ = original_shaped_gates(source, child, 1, backward.circuit.instructions)
    assert_gate_parameters(backward.circuit.instructions, original)
    initial, _ = initial_and_expected(1, dirty, (value,), selected)
    for first, second in product((0, 1), repeat=2):
        computed = forced_branch(initial, forward.circuit.instructions, (first,))
        actual = forced_branch(computed, backward.circuit.instructions, (second,))
        upstream = forced_branch(computed, original, (second,))
        torch.testing.assert_close(actual, upstream, atol=3e-12, rtol=0)
        torch.testing.assert_close(actual, initial / 2, atol=3e-12, rtol=0)
