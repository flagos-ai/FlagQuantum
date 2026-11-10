"""Packed existing controls retain source ordering, phases and exact cleanup."""

from __future__ import annotations

import re
from collections import Counter
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
from flagquantum.compiler.registers.control.specialized_component_control import (
    specialized_component_control_system,
)
from flagquantum.compiler.registers.layout.auto_partition import AutoPartition
from flagquantum.compiler.registers.layout.component_auto_partition import (
    auto_partition_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    source_partition_leaf,
    walk_source_graph,
)
from tests.unit.test_component_auto_partition import source_partition
from tests.unit.test_component_control_selection import assert_gate_parameters, fixtures
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_hierarchy_adjoint import source_adjoint
from tests.unit.test_component_structured_control import normalize_trace, original_gates
from tests.unit.test_register_constant_addition import reference_arithmetic_gates

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def packed_fixture(
    layers: int, leading: bool, inverse: bool, activation: int = 1
) -> tuple[ProgramRecord, Any, Any]:
    """Pack the requested source control specialization or its exact adjoint."""
    _, _, native, originals = fixtures("spectator", True)
    module, _ = native[activation]
    child = originals[activation]
    if inverse:
        module = adjoint_component_hierarchy(module, {"ch": "ch", "cp": "cp_inverse"})
        child = source_adjoint(child.decompose_bloq())
    groups: tuple[tuple[str, tuple[tuple[str | None, int], ...]], ...] = (
        ("control", (("ctrl", 1),)),
        ("data", (("q", 1), (None, 1), ("spectator", 1))),
    )
    if not leading:
        groups = groups[::-1]
    layout = AutoPartition(groups)
    wrapper = source_partition(child, layout)
    module = auto_partition_component_program(module, layout, name="packed0")
    if layers == 2:
        layout = AutoPartition(
            tuple(
                (name, ((name, width),))
                for name, width in (("control", 1), ("data", 3))
            )[:: 1 if leading else -1]
        )
        wrapper = source_partition(wrapper, layout)
        module = auto_partition_component_program(
            module, layout, entry="packed0", name="packed1"
        )
    return module, wrapper, child


def source_gates(
    source: Any,
    child: Any,
    inputs: SourcePorts,
    work: tuple[int, ...],
    bits: dict[int, int],
) -> tuple[Instruction, ...]:
    from qualtran.bloqs.mcmt.and_bloq import And

    def leaf(
        bloq: Any, ports: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        partition = source_partition_leaf(bloq, ports, desired)
        if partition is not None:
            return partition
        if bloq == child:
            return original_gates(bloq, ports, work[-1:], bits), ports
        if isinstance(bloq, And):
            outputs = {"ctrl": ports["ctrl"]}
            if not bloq.uncompute:
                outputs["target"] = work[:1]
            gates = reference_arithmetic_gates(
                bloq,
                {**ports, **outputs},
                (),
                bits,
                expected_outputs=outputs,
                cirq_style=True,
            )
            return gates, outputs
        return None

    gates, outputs = walk_source_graph(source, inputs, inputs, leaf)
    assert outputs == inputs
    return gates


@pytest.mark.parametrize("layers", (1, 2))
@pytest.mark.parametrize("leading", (False, True))
@pytest.mark.parametrize("old,new", tuple(product((0, 1), repeat=2)))
@pytest.mark.parametrize("expanded", (False, True))
def test_packed_existing_controls_and_exact_inverse(
    layers: int, leading: bool, old: int, new: int, expanded: bool
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.mcmt.specialized_ctrl import _get_ctrl_system_1bit_cv
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    circuits = []
    for inverse in (False, True):
        module, wrapper, child = packed_fixture(layers, leading, inverse)
        entry = f"packed{layers-1}"
        source, callback = _get_ctrl_system_1bit_cv(
            wrapper,
            CtrlSpec(cvs=new),
            current_ctrl_bit=old,
            get_ctrl_bloq_and_ctrl_reg_name=lambda value, wrapper=wrapper: (
                wrapper,
                "control",
            ),
        )
        selected, adder = specialized_component_control_system(
            module, "control", old, new, entry
        )
        assert verify_program_record(selected) == ()
        observer = _source(Signature([]))
        root = selected.functions[0]
        assert tuple(observer.signature(Signature(root.registers or ()))) == tuple(
            source.signature
        )
        assert normalize_trace(_native_trace(selected, entry)) == normalize_trace(
            _source_trace(observer, source.decompose_bloq())
        )
        names = {
            op.callee: inst.bloq
            for op, (inst, _, _) in zip(
                root.blocks[0].operations,
                source.decompose_bloq().iter_bloqsoqs(),
                strict=True,
            )
            if op.callee is not None
        }
        counts = program_component_call_graph(
            selected, entry=entry, keep=names.__contains__
        ).leaf_counts
        assert (
            Counter({names[name]: count for name, count in counts.items()})
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
                count * getattr(get_cost_value(names[name], QECGatesCost()), category)
                for name, count in counts.items()
            )
        signature = Signature(
            (
                Register("added", quantum_bit_type()),
                Register("control", quantum_bit_type()),
                Register(
                    "data",
                    next(r.dtype for r in root.registers or () if r.name == "data"),
                ),
            )
        )
        builder = ComponentProgramBuilder(signature, selected.functions, name="caller")
        observer = _source(signature)
        values = {name: builder.inputs[name] for name in ("control", "data")}
        source_values = {name: observer.inputs[name] for name in ("control", "data")}
        added, data = adder(builder, builder.inputs["added"], values)
        src_added, src_data = callback(
            observer.builder, (observer.inputs["added"],), source_values
        )
        caller = builder.finalize(
            dict(zip((r.name for r in signature), (added, *data), strict=True))
        )
        composite = observer.builder.finalize(
            **dict(
                zip((r.name for r in signature), (*src_added, *src_data), strict=True)
            )
        )
        assert normalize_trace(_native_trace(caller, "caller")) == normalize_trace(
            _source_trace(observer, composite)
        )
        assert tuple(values) == tuple(source_values)
        if expanded:
            caller, _ = expand_component_calls(caller, entry="caller")
        bound = bind_register_program(
            caller, ((0,), (1,), (2, 3, 4)), clean=(5, 6), entry="caller"
        )
        assert bound.outputs == ((0,), (1,), (2, 3, 4))
        gates = bound.circuit.instructions
        bits = {
            g.wires[0]: g.metadata["classical_bit"]
            for g in gates
            if g.name == "measure"
        }
        reference = source_gates(
            source, child, {"control": (0, 1), "data": (2, 3, 4)}, (5, 6), bits
        )
        assert_gate_parameters(gates, reference)
        emitted = Counter(g.name for g in gates)
        assert emitted["t"] + emitted["tdg"] == 8
        assert emitted["measure"] == emitted["reset"] == model.measurement == 2
        assert get_cost_value(source, QubitCount()) == bound.circuit.n_wires == 7
        circuits.append(gates)
    initial = torch.zeros(128, dtype=torch.complex128)
    initial[::4] = torch.tensor(
        [complex(i + 1, 7 - 2 * i) for i in range(32)], dtype=torch.complex128
    )
    initial /= initial.norm()
    expected = torch.zeros_like(initial)
    for ctrl0, ctrl1, q, unused, spare in product((0, 1), repeat=5):
        index = (ctrl0 * 16 + ctrl1 * 8 + q * 4 + unused * 2 + spare) * 4
        if (ctrl0, ctrl1) != (new, old):
            expected[index] += initial[index]
        else:
            for out in (0, 1):
                target = (ctrl0 * 16 + ctrl1 * 8 + out * 4 + unused * 2 + spare) * 4
                expected[target] += (
                    initial[index]
                    * (-1 if q and out else 1)
                    / 2**0.5
                    * np.exp(1j * np.pi * out / 4)
                )
    for first in product((0, 1), repeat=2):
        state = forced_branch(initial, circuits[0], first)
        torch.testing.assert_close(state, expected / 2, atol=3e-12, rtol=0)
        for second in product((0, 1), repeat=2):
            restored = forced_branch(state, circuits[1], second)
            torch.testing.assert_close(restored, initial / 4, atol=3e-12, rtol=0)
            assert torch.count_nonzero(restored.reshape(-1, 4)[:, 1:]) == 0


def test_packed_helper_runs_through_append() -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    module, _, _ = packed_fixture(2, False, False)
    selected, _ = specialized_component_control_system(
        module, "control", 0, 1, "packed1"
    )
    bound = bind_register_program(
        selected, ((0,), (1,), (2, 3, 4)), clean=(5, 6), entry="packed1"
    )
    initial = torch.zeros(128, dtype=torch.complex128)
    initial[::4] = torch.tensor(
        [complex(i + 1, 2 - i) for i in range(32)], dtype=torch.complex128
    )
    initial /= initial.norm()
    expected = forced_branch(initial, bound.circuit.instructions, (0, 0)) * 2
    circuit = DynamicCircuit(7, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(7, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=8, seed=29, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), expected.expand(8, -1), atol=3e-12, rtol=0
    )


@pytest.mark.parametrize("layers", (1, 2))
@pytest.mark.parametrize("leading", (False, True))
def test_specialized_adjoint_reverses_wrapped_callback(
    layers: int, leading: bool
) -> None:
    from dataclasses import replace

    from qualtran import Bloq, CtrlSpec
    from qualtran.bloqs.mcmt.specialized_ctrl import (
        AdjointWithSpecializedCtrl,
        SpecializeOnCtrlBit,
        _get_ctrl_system_1bit_cv,
    )

    from flagquantum.compiler.registers.component_expansion import (
        expand_component_calls_once,
    )
    from flagquantum.compiler.registers.control.component_adjoint_control import (
        adjoint_component_control_system,
    )
    from flagquantum.compiler.registers.control.component_control_selection import (
        select_component_control_system,
    )
    from flagquantum.compiler.registers.control.component_extension import (
        StructuredControlAdder,
    )
    from flagquantum.compiler.registers.control.control_predicate import ControlRegister
    from flagquantum.core.ir.program.registers import RegisterType
    from tests.unit.test_component_control import add_h, add_phase
    from tests.unit.test_component_control_selection import native_fallback

    forward, data_source, _, _ = fixtures("spectator", True)
    layout = AutoPartition((("data", (("q", 1), (None, 1), ("spectator", 1))),))
    data_source = source_partition(data_source, layout)
    forward = auto_partition_component_program(forward, layout)
    renames = {"main": "unpacked", "packed": "main"}
    forward = replace(
        forward,
        functions=tuple(
            replace(
                f,
                name=renames.get(f.name, f.name),
                blocks=tuple(
                    replace(
                        block,
                        operations=tuple(
                            replace(
                                op,
                                callee=(
                                    renames.get(op.callee, op.callee)
                                    if op.callee is not None
                                    else None
                                ),
                            )
                            for op in block.operations
                        ),
                    )
                    for block in f.blocks
                ),
            )
            for f in forward.functions
        ),
    )
    module, wrapper, child = packed_fixture(layers, leading, False)
    entry = f"packed{layers-1}"
    controls = (Register("added", quantum_bit_type()),)
    patterns = (ControlRegister(1, (1,)),)

    def original_system(component: Any, spec: Any) -> Any:
        return _get_ctrl_system_1bit_cv(
            component,
            spec,
            current_ctrl_bit=None,
            get_ctrl_bloq_and_ctrl_reg_name=lambda value: (wrapper, "control"),
        )

    stored = type(
        "StoredPackedBody",
        (Bloq,),
        {
            "signature": data_source.signature,
            "decompose_bloq": lambda component: data_source.decompose_bloq(),
            "get_ctrl_system": original_system,
        },
    )()

    def controlled(
        regs: tuple[Register, ...], values: tuple[ControlRegister, ...]
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        return select_component_control_system(
            regs,
            values,
            lambda value: (module, "control"),
            native_fallback(forward, {"h": add_h, "p": add_phase}),
            entry=entry,
        )

    adjoints = {"ch": "ch", "cp": "cp_inverse"}
    for f in module.functions:
        if re.fullmatch(r"__fq_partition_[0-9]+_(split|join)[0-9]+", f.name):
            other = (
                f.name.replace("_split", "_join")
                if "_split" in f.name
                else f.name.replace("_join", "_split")
            )
            adjoints[f.name] = other
    selected, adder = adjoint_component_control_system(
        forward, controls, patterns, frozenset((1,)), controlled, controlled, adjoints
    )
    original, callback = AdjointWithSpecializedCtrl(
        stored, SpecializeOnCtrlBit.ONE
    ).get_ctrl_system(CtrlSpec())
    observer = _source(Signature([]))
    assert tuple(
        observer.signature(Signature(selected.functions[0].registers or ()))
    ) == tuple(original.signature)
    signature = Signature((*controls, Register("data", RegisterType("opaque", 3))))
    builder = ComponentProgramBuilder(signature, selected.functions, name="caller")
    observer = _source(signature)
    values = {"data": builder.inputs["data"]}
    upstream_values = {"data": observer.inputs["data"]}
    added, data = adder(builder, (builder.inputs["added"],), values)
    src_added, src_data = callback(
        observer.builder, (observer.inputs["added"],), upstream_values
    )
    caller = builder.finalize({"added": added[0], "data": data[0]})
    composite = observer.builder.finalize(
        added=tuple(src_added)[0], data=tuple(src_data)[0]
    )
    caller, _ = expand_component_calls_once(
        caller,
        "caller",
        predicate=lambda _, op: op.callee is not None
        and op.callee.startswith("__fq_adjoint_control_"),
    )
    assert normalize_trace(_native_trace(caller, "caller")) == normalize_trace(
        _source_trace(observer, composite)
    )
    assert tuple(values) == tuple(upstream_values)
    bound = bind_register_program(caller, ((0,), (1, 2, 3)), clean=(4,), entry="caller")
    gates = bound.circuit.instructions
    bits = {
        g.wires[0]: g.metadata["classical_bit"] for g in gates if g.name == "measure"
    }

    def leaf(
        bloq: Any, ports: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        partition = source_partition_leaf(bloq, ports, desired)
        if partition is not None:
            return partition
        if bloq == child.adjoint():
            return original_gates(bloq, ports, (4,), bits), ports
        return None

    reference, outputs = walk_source_graph(
        composite,
        {"added": (0,), "data": (1, 2, 3)},
        {"added": (0,), "data": (1, 2, 3)},
        leaf,
    )
    assert outputs == {"added": (0,), "data": (1, 2, 3)}
    assert_gate_parameters(gates, reference)
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    model = get_cost_value(composite, QECGatesCost())
    emitted = Counter(g.name for g in gates)
    assert emitted["t"] + emitted["tdg"] == 4
    assert emitted["measure"] == emitted["reset"] == model.measurement == 1
    assert model.total_t_count() == 27
    assert get_cost_value(composite, QubitCount()) == bound.circuit.n_wires == 5
    forward_gates = bind_register_program(
        module,
        ((0,), (1, 2, 3)) if leading else ((1, 2, 3), (0,)),
        clean=(4,),
        entry=entry,
    ).circuit.instructions
    initial = torch.zeros(32, dtype=torch.complex128)
    initial[::2] = torch.tensor(
        [complex(i + 1, 4 - i) for i in range(16)], dtype=torch.complex128
    )
    initial /= initial.norm()
    for first, second in product((0, 1), repeat=2):
        state = forced_branch(initial, forward_gates, (first,))
        restored = forced_branch(state, gates, (second,))
        torch.testing.assert_close(restored, initial / 2, atol=3e-12, rtol=0)
        assert torch.count_nonzero(restored[1::2]) == 0
