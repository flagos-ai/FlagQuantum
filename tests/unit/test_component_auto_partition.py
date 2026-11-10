"""Typed AutoPartition roots retain original partition calls and quantum channels."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler._component_trace.program_call_graph import (
    program_component_call_graph,
)
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.layout.auto_partition import AutoPartition
from flagquantum.compiler.registers.layout.component_auto_partition import (
    auto_partition_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    source_partition_leaf,
    walk_source_graph,
)
from tests.unit.test_component_control_selection import assert_gate_parameters
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_structured_control import normalize_trace, original_gates
from tests.unit.test_specialized_component_control import positive_fixture

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

LAYOUTS = (
    (("packed", (("ctrl", 1), (None, 1), ("q", 1), ("spectator", 1))),),
    (("a", (("ctrl", 1), ("q", 1))), ("b", (("spectator", 1), (None, 1)))),
    (("a", ((None, 1), ("ctrl", 1))), ("b", (("q", 1), ("spectator", 1)))),
)


def source_partition(child: Any, layout: AutoPartition) -> Any:
    from qualtran import QAny, Register
    from qualtran.bloqs.bookkeeping.auto_partition import AutoPartition as Original
    from qualtran.bloqs.bookkeeping.auto_partition import Unused

    return Original(
        child,
        [
            (
                Register(outer, QAny(sum(width for _, width in group))),
                [Unused(width) if port is None else port for port, width in group],
            )
            for outer, group in layout.groups
        ],
        left_only=layout.left_only,
    )


def partition_gates(
    source: Any,
    child: Any,
    inputs: SourcePorts,
    outputs: SourcePorts,
    work: tuple[int, ...],
    bits: dict[int, int],
) -> tuple[Instruction, ...]:
    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        partition = source_partition_leaf(bloq, inputs, desired)
        if partition is not None:
            return partition
        if bloq == child:
            return original_gates(bloq, inputs, work, bits), inputs
        return None

    gates, returned = walk_source_graph(source, inputs, outputs, leaf)
    assert returned == outputs
    return gates


@pytest.mark.parametrize("groups", LAYOUTS)
@pytest.mark.parametrize("expanded", (False, True))
def test_packed_call_graph_all_histories_amplitudes_unused_and_resources(
    groups: tuple[tuple[str, tuple[tuple[str | None, int], ...]], ...], expanded: bool
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    record, child = positive_fixture("spectator")
    layout = AutoPartition(groups)
    module = auto_partition_component_program(record, layout)
    source = source_partition(child, layout)
    observer = _source(Signature([]))
    root = next(f for f in module.functions if f.name == "packed")
    assert tuple(observer.signature(Signature(root.registers or ()))) == tuple(
        source.signature
    )
    assert normalize_trace(_native_trace(module, "packed")) == normalize_trace(
        _source_trace(observer, source.decompose_bloq())
    )
    terminals = {
        op.callee: instance.bloq
        for op, (instance, _, _) in zip(
            root.blocks[0].operations,
            source.decompose_bloq().iter_bloqsoqs(),
            strict=True,
        )
        if op.callee is not None
    }
    counts = program_component_call_graph(
        module, entry="packed", keep=terminals.__contains__
    ).leaf_counts
    assert (
        Counter({terminals[name]: count for name, count in counts.items()})
        == source.bloq_counts()
    )
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
            count * getattr(get_cost_value(terminals[name], QECGatesCost()), category)
            for name, count in counts.items()
        )
    cursor = 0
    ports = {}
    for outer, parts in groups:
        width = sum(width for _, width in parts)
        ports[outer] = tuple(range(cursor, cursor + width))
        cursor += width
    if expanded:
        module, _ = expand_component_calls(module, "packed")
    bound = bind_register_program(
        module, tuple(ports.values()), clean=(4,), entry="packed"
    )
    assert bound.outputs == tuple(ports.values())
    assert bound.circuit.n_wires == get_cost_value(source, QubitCount()) == 5
    gates = bound.circuit.instructions
    bits = {
        g.wires[0]: g.metadata["classical_bit"] for g in gates if g.name == "measure"
    }
    assert_gate_parameters(
        gates, partition_gates(source, child, ports, ports, (4,), bits)
    )
    inputs, _ = layout.bind(ports)
    ctrl = inputs["ctrl"][0]
    q = inputs["q"][0]
    initial = torch.zeros(32, dtype=torch.complex128)
    initial[::2] = torch.tensor(
        [complex(i + 1, 7 - i) for i in range(16)], dtype=torch.complex128
    )
    initial /= initial.norm()
    expected = torch.zeros_like(initial)
    for index in range(16):
        if (index >> (3 - ctrl)) & 1:
            old = (index >> (3 - q)) & 1
            for out in (0, 1):
                target = (index & ~(1 << (3 - q))) | (out << (3 - q))
                expected[target << 1] += (
                    initial[index << 1]
                    * (-1 if old and out else 1)
                    / 2**0.5
                    * np.exp(1j * np.pi * out / 4)
                )
        else:
            expected[index << 1] += initial[index << 1]
    for bit in (0, 1):
        actual = forced_branch(initial, gates, (bit,))
        torch.testing.assert_close(actual, expected / 2**0.5, atol=2e-12, rtol=0)
        assert torch.count_nonzero(actual[1::2]) == 0
    assert verify_program_record(module) == ()


@pytest.mark.parametrize("shape", ((2,), (1, 2)))
@pytest.mark.parametrize("left_only", (False, True))
def test_array_partitions_preserve_source_output_permutation(
    shape: tuple[int, ...], left_only: bool
) -> None:
    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.core.ir.program.registers import Register, RegisterType

    registers = (
        Register("bits", RegisterType("bit", 1), shape),
        Register("spare", RegisterType("uint", 2)),
    )
    signature = Signature(registers)
    builder = ComponentProgramBuilder(signature)
    observer = _source(signature)
    returned = np.asarray(builder.inputs["bits"]).reshape(-1)[::-1].reshape(shape)
    source_returned = (
        np.asarray(observer.inputs["bits"]).reshape(-1)[::-1].reshape(shape)
    )
    record = builder.finalize({"bits": returned, "spare": builder.inputs["spare"]})
    child = observer.builder.finalize(
        bits=source_returned, spare=observer.inputs["spare"]
    )
    layout = AutoPartition(
        (("packed", (("bits", 2), ("spare", 2))),), left_only=left_only
    )
    module = auto_partition_component_program(record, layout)
    source = source_partition(child, layout)
    bound = bind_register_program(module, ((0, 1, 2, 3),), entry="packed")
    root = next(f for f in module.functions if f.name == "packed")
    assert tuple(
        _source(Signature([])).signature(Signature(root.registers or ()))
    ) == tuple(source.signature)
    assert bound.circuit.instructions == ()
    assert tuple(wire for port in bound.outputs for wire in port) == (1, 0, 2, 3)
    initial = torch.tensor(
        [complex(i + 1, 5 - i) for i in range(16)], dtype=torch.complex128
    )
    initial /= initial.norm()
    source_state = torch.as_tensor(
        np.einsum("ij,j->i", source.tensor_contract(), initial.numpy())
    )
    expected = torch.empty_like(initial)
    for index in range(16):
        logical = ((index >> 2) & 1) << 3 | ((index >> 3) & 1) << 2 | (index & 3)
        expected[logical] = initial[index]
    torch.testing.assert_close(source_state, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "failure", ("entry", "name", "width", "outputs", "unused_left")
)
def test_invalid_partition_rejects_without_changing_child(failure: str) -> None:
    record, _ = positive_fixture("spectator")
    before = repr(record)
    groups = (("a", (("ctrl", 1), ("q", 1), ("spectator", 1))),)
    if failure == "width":
        groups = (("a", (("ctrl", 2), ("q", 1), ("spectator", 1))),)
    with pytest.raises(ValueError):
        layout = AutoPartition(
            groups,
            child_outputs=("q",) if failure == "outputs" else None,
            left_only=failure == "outputs",
        )
        if failure == "unused_left":
            layout = AutoPartition(
                (("a", (("ctrl", 1), ("q", 1), ("spectator", 1), (None, 1))),),
                left_only=True,
            )
        auto_partition_component_program(
            record,
            layout,
            entry="absent" if failure == "entry" else "main",
            name="main" if failure == "name" else "packed",
        )
    assert repr(record) == before


def test_left_only_consumes_original_left_port_and_retains_output_owner() -> None:
    from qualtran.bloqs.basic_gates import Hadamard

    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.core.ir import CircuitIR, Instruction
    from flagquantum.core.ir.program.register_bindings import RegisterBinding
    from flagquantum.core.ir.program.registers import (
        Register,
        RegisterSide,
        RegisterType,
    )
    from tests.unit.test_component_control import body

    bit = RegisterType("bit", 1)
    signature = Signature(
        (Register("drop", bit, side=RegisterSide.LEFT), Register("q", bit))
    )
    builder = ComponentProgramBuilder(
        signature,
        body("h", (Register("q", bit),), CircuitIR(1, (Instruction("h", (0,)),))),
    )
    dropped = builder.inputs["drop"]
    assert isinstance(dropped, RegisterBinding)
    builder.free(dropped)
    record = builder.finalize(builder.call_dict("h", {"q": builder.inputs["q"]}))
    observer = _source(signature)
    observer.builder.free(observer.inputs["drop"])
    q = observer.builder.add(Hadamard(), q=observer.inputs["q"])
    child = observer.builder.finalize(q=q)
    layout = AutoPartition(
        (("data", (("drop", 1), ("q", 1))),), child_outputs=("q",), left_only=True
    )
    module = auto_partition_component_program(record, layout)
    source = source_partition(child, layout)
    root = next(f for f in module.functions if f.name == "packed")
    assert tuple(
        _source(Signature([])).signature(Signature(root.registers or ()))
    ) == tuple(source.signature)
    bound = bind_register_program(
        module, ((0, 1),), entry="packed", input_ownership=((0, False),)
    )
    assert bound.outputs == ((1,),)
    assert bound.circuit.instructions == (Instruction("h", (1,)),)
    initial = torch.tensor([1 + 2j, 3 - 1j, 0, 0], dtype=torch.complex128) / 15**0.5
    actual = forced_branch(initial, bound.circuit.instructions, ())
    expected = torch.as_tensor(
        np.einsum("ij,j->i", source.tensor_contract(), initial.numpy())
    )
    torch.testing.assert_close(actual[:2], expected, atol=1e-12, rtol=0)
    assert torch.count_nonzero(actual[2:]) == 0


def test_packed_operation_keeps_existing_append_execution() -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    record, _ = positive_fixture("spectator")
    module = auto_partition_component_program(record, AutoPartition(LAYOUTS[0]))
    bound = bind_register_program(module, ((0, 1, 2, 3),), clean=(4,), entry="packed")
    initial = torch.zeros(32, dtype=torch.complex128)
    initial[::2] = torch.tensor(
        [complex(i + 1, 5 - i) for i in range(16)], dtype=torch.complex128
    )
    initial /= initial.norm()
    expected = forced_branch(initial, bound.circuit.instructions, (0,)) * 2**0.5
    circuit = DynamicCircuit(5, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(5, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=8, seed=29, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), expected.expand(8, -1), atol=2e-12, rtol=0
    )


@pytest.mark.parametrize("direction", ("left", "right"))
def test_repacked_directional_children_reject_at_admission(direction: str) -> None:
    from qualtran import BloqBuilder, QAny, QBit, Side
    from qualtran import Register as OriginalRegister
    from qualtran import Signature as OriginalSignature
    from qualtran.bloqs.bookkeeping.auto_partition import AutoPartition as Original

    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.core.ir.program.register_bindings import RegisterBinding
    from flagquantum.core.ir.program.registers import (
        Register,
        RegisterSide,
        RegisterType,
    )

    bit = RegisterType("bit", 1)
    side = RegisterSide.LEFT if direction == "left" else RegisterSide.RIGHT
    builder = ComponentProgramBuilder(Signature((Register("q", bit, side=side),)))
    if direction == "left":
        q = builder.inputs["q"]
        assert isinstance(q, RegisterBinding)
        builder.free(q)
        record = builder.finalize({})
    else:
        record = builder.finalize({"q": builder.allocate(dtype=bit)})
    with pytest.raises(ValueError):
        auto_partition_component_program(record, AutoPartition((("x", (("q", 1),)),)))
    original_builder, soqs = BloqBuilder.from_signature(
        OriginalSignature(
            (
                OriginalRegister(
                    "q", QBit(), side=Side.LEFT if direction == "left" else Side.RIGHT
                ),
            )
        )
    )
    if direction == "left":
        original_builder.free(soqs["q"])
        child = original_builder.finalize()
        # Source construction accepts this, then repacking requests a missing right port.
        with pytest.raises(KeyError):
            Original(child, [(OriginalRegister("x", QAny(1)), ["q"])]).decompose_bloq()
    else:
        child = original_builder.finalize(q=original_builder.allocate(dtype=QBit()))
        with pytest.raises(ValueError):
            Original(child, [(OriginalRegister("x", QAny(1)), ["q"])])


def test_partition_helper_names_avoid_existing_definitions() -> None:
    from dataclasses import replace

    record, _ = positive_fixture("spectator")
    record = replace(
        record,
        functions=(
            *record.functions,
            replace(record.functions[-1], name="__fq_partition_0_reserved"),
        ),
    )
    module = auto_partition_component_program(record, AutoPartition(LAYOUTS[0]))
    assert verify_program_record(module) == ()
    assert any(f.name == "__fq_partition_1_split0" for f in module.functions)
