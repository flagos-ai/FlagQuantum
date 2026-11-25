"""Frozen modular multiply calls retain exact work, feedback and output aliases."""

from collections import Counter
from typing import Any

import pytest

from flagquantum.compiler.registers.modular.modular_doubling_component import (
    modular_doubling_program,
)
from flagquantum.compiler.registers.modular.modular_multiplication_component import (
    modular_multiplication_program,
)
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.register_types import (
    montgomery_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_stream(
    source: Any, bound: BoundRegisterProgram, ports: SourcePorts
) -> tuple[Instruction, ...]:
    from qualtran import Side
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.arithmetic.addition import OutOfPlaceAdder
    from qualtran.bloqs.bookkeeping import Allocate, Cast, Free, Join, Split
    from qualtran.bloqs.mcmt import And
    from qualtran.bloqs.mod_arithmetic.mod_addition import CtrlScaleModAdd

    cursor = 0
    root_allocations: set[int] = set()
    slots = MeasurementSlots(bound.circuit.instructions)

    def leaf(bloq: Any, old: SourcePorts, desired: SourcePorts) -> SourceResult | None:
        nonlocal cursor
        if (
            bloq is source
            or isinstance(
                bloq,
                (
                    Join,
                    Split,
                    Cast,
                    CtrlScaleModAdd,
                    OutOfPlaceAdder,
                ),
            )
            or isinstance(bloq, Adjoint)
            and isinstance(
                bloq.subbloq,
                (CtrlScaleModAdd, OutOfPlaceAdder),
            )
        ):
            return None
        if isinstance(bloq, Free):
            return (), {}
        if isinstance(bloq, Allocate):
            event = next(
                e
                for e in bound.events
                if e.kind == "allocate"
                and e.instruction_offset == cursor
                and e.position not in root_allocations
            )
            root_allocations.add(event.position)
            return (), {"reg": event.wires}
        import numpy as np
        from qualtran.bloqs.basic_gates import (
            IntEffect,
            IntState,
            ZeroEffect,
            ZeroState,
        )

        base = bloq.subbloq if isinstance(bloq, Adjoint) else bloq
        if isinstance(base, (IntState, IntEffect, ZeroState, ZeroEffect)):
            # Basis effects have no complete Cirq export. Verify the original
            # unnormalized tensor, then bind its existing X loader and lifetime.
            register = next(iter(bloq.signature))
            bits = (
                tuple(base.dtype.to_bits(base.val))
                if isinstance(base, (IntState, IntEffect))
                else (0,)
            )
            size = len(bits)
            vector = np.zeros(1 << size, dtype=np.complex128)
            vector[sum(bit << (size - 1 - i) for i, bit in enumerate(bits))] = 1
            np.testing.assert_array_equal(bloq.tensor_contract().reshape(-1), vector)
            state = register.side == Side.RIGHT
            if state:
                event = next(
                    e
                    for e in bound.events
                    if e.kind == "allocate"
                    and e.instruction_offset == cursor
                    and e.position not in root_allocations
                )
                root_allocations.add(event.position)
                wires = event.wires
                returned = {register.name: wires}
            else:
                wires = old[register.name]
                returned = {}
            expected = tuple(
                Instruction("x", (wire,))
                for wire, bit in zip(wires, bits, strict=True)
                if bit
            )
            assert all(returned[k] == v for k, v in desired.items())
            cursor += len(expected)
            return expected, returned
        atomic = bloq.as_composite_bloq()
        start = max(w for wires in old.values() for w in wires) + 100
        dummy = tuple(range(start, start + 1000))
        returned = {
            r.name: old[r.name] for r in bloq.signature.rights() if r.name in old
        }
        dry_outputs = dict(returned)
        if isinstance(bloq, And) and not bloq.uncompute:
            dry_outputs["target"] = dummy[:1]
        dry_slots = {
            w: w for w in (*dummy, *(w for wires in old.values() for w in wires))
        }
        dry = reference_arithmetic_gates(
            atomic,
            old,
            dummy,
            dry_slots,
            expected_outputs=dry_outputs,
            reuse_allocations=False,
            preserve_components=True,
        )
        plan = tuple(
            w
            for e in bound.events
            if e.kind == "allocate"
            and e.position not in root_allocations
            and e.instruction_offset is not None
            and cursor <= e.instruction_offset < cursor + len(dry)
            for w in e.wires
        )
        returned = {
            r.name: old[r.name] for r in bloq.signature.rights() if r.name in old
        }
        if isinstance(bloq, And) and not bloq.uncompute:
            returned["target"] = plan[:1]
        expected = reference_arithmetic_gates(
            atomic,
            old,
            plan,
            slots,
            expected_outputs=returned,
            reuse_allocations=False,
            preserve_components=True,
        )
        assert len(expected) == len(dry)
        assert all(returned[k] == v for k, v in desired.items())
        cursor += len(expected)
        return expected, returned

    gates, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports and cursor == len(bound.circuit.instructions)
    return gates


@pytest.mark.parametrize("n,p,k", [(2, 3, 2), (3, 5, 2), (5, 13, 3), (8, 221, 123)])
@pytest.mark.parametrize("montgomery", [False, True])
def test_controlled_multiplier_full_source_graph(
    n: int, p: int, k: int, montgomery: bool
) -> None:
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import CModMulK
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    dtype = montgomery_register_type(n, p) if montgomery else unsigned_register_type(n)
    original = QMontgomeryUInt(n, p) if montgomery else QUInt(n)
    graph = modular_multiplication_program(dtype, p, k)
    source = CModMulK(original, k, p)
    assert _native_trace(graph) == _source_trace(
        _source(Signature([])), source.decompose_bloq()
    )
    ports = {"ctrl": (0,), "x": tuple(range(1, n + 1))}
    bound = bind_register_program(
        graph, tuple(ports.values()), clean=tuple(range(n + 1, 6 * n + 10))
    )
    assert_source(bound.circuit.instructions, source_stream(source, bound, ports))
    counts = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count(ts_per_cswap=7)
    assert counts["measure"] == model.measurement
    active: set[int] = set()
    peak = 0
    for event in bound.events:
        if event.kind == "allocate":
            active.update(event.wires)
        else:
            active.difference_update(event.wires)
        peak = max(peak, len(active))
    assert not active and n + 1 + peak == get_cost_value(source, QubitCount())


@pytest.mark.parametrize("n,p", [(1, 1), (2, 3), (3, 5), (5, 13), (8, 251)])
@pytest.mark.parametrize("montgomery", [False, True])
def test_doubling_source_stream_and_output_ownership(
    n: int, p: int, montgomery: bool
) -> None:
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import ModDbl
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    dtype = montgomery_register_type(n, p) if montgomery else unsigned_register_type(n)
    original = QMontgomeryUInt(n, p) if montgomery else QUInt(n)
    graph = modular_doubling_program(dtype, p)
    source = ModDbl(original, p)
    compare_layers(graph, source, {f.name for f in graph.functions if f.name != "main"})
    inputs = (tuple(range(n)),)
    bound = bind_register_program(
        graph,
        inputs,
        clean=tuple(range(n, 6 * n + 10)),
        input_ownership=((0, False),),
        allow_workspace_outputs=True,
    )
    alloc = tuple(w for e in bound.events if e.kind == "allocate" for w in e.wires)
    expected = reference_arithmetic_gates(
        source,
        {"x": inputs[0]},
        alloc,
        MeasurementSlots(bound.circuit.instructions),
        reuse_allocations=False,
        preserve_components=True,
        expected_outputs={"x": bound.outputs[0]},
    )
    assert_source(bound.circuit.instructions, expected)
    assert bound.outputs == (tuple(range(1, n)) + (n + 1,),)
    counts = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count()
    assert counts["measure"] == model.measurement
    live = set(inputs[0])
    peak = len(live)
    for event in bound.events:
        if event.kind == "allocate":
            live.update(event.wires)
        else:
            live.difference_update(event.wires)
        peak = max(peak, len(live))
    assert live == set(bound.outputs[0]) and peak == get_cost_value(
        source, QubitCount()
    )
