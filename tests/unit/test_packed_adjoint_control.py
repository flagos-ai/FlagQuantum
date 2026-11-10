"""Whole existing-control adders follow the pinned specialized-adjoint chain."""

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
from flagquantum.compiler.registers.component_bindings import RegisterBindingError
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_expansion import (
    expand_component_calls,
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
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterType
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
from tests.unit.test_packed_specialized_control import packed_fixture
from tests.unit.test_register_constant_addition import reference_arithmetic_gates

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_existing_control_system(
    current: Any, wrapper: Any, old: int, new: int
) -> Any:
    """Invoke the unchanged existing-control specialization and adjoint callback."""
    from qualtran import Bloq, CtrlSpec
    from qualtran.bloqs.mcmt.specialized_ctrl import (
        AdjointWithSpecializedCtrl,
        SpecializeOnCtrlBit,
        _get_ctrl_system_1bit_cv,
    )

    def system(component: Any, spec: Any) -> Any:
        return _get_ctrl_system_1bit_cv(
            component,
            spec,
            current_ctrl_bit=old,
            get_ctrl_bloq_and_ctrl_reg_name=lambda value: (wrapper, "control"),
        )

    stored = type(
        "StoredPackedExistingControl",
        (Bloq,),
        {
            "signature": current.signature,
            "decompose_bloq": lambda component: current.decompose_bloq(),
            "get_ctrl_system": system,
        },
    )()
    return AdjointWithSpecializedCtrl(stored, SpecializeOnCtrlBit.BOTH).get_ctrl_system(
        CtrlSpec(cvs=new)
    )


def existing_adjoint_fixture(
    layers: int, leading: bool, old: int, new: int
) -> tuple[ProgramRecord, Any, Any, Any, StructuredControlAdder, str, ProgramRecord]:
    """Call the real source factory and reverse the native factory's actual adder."""
    forward, current, _ = packed_fixture(layers, leading, False, old)
    positive, wrapper, child = packed_fixture(layers, leading, False)
    entry = f"packed{layers-1}"
    adjoints = {"ch": "ch", "cp": "cp_inverse"}
    selected_forward = []

    def controlled(
        controls: tuple[Register, ...], patterns: tuple[ControlRegister, ...]
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        selected, adder = select_component_control_system(
            controls,
            patterns,
            lambda value: (positive, "control"),
            unexpected_fallback,
            current_value=old,
            entry=entry,
        )
        selected_forward.append(selected)
        for f in selected.functions:
            if re.fullmatch(r"__fq_partition_[0-9]+_(split|join)[0-9]+", f.name):
                adjoints[f.name] = (
                    f.name.replace("_split", "_join")
                    if "_split" in f.name
                    else f.name.replace("_join", "_split")
                )
            elif re.fullmatch(r"__fq_specialized_[0-9]+_(and|erase)", f.name):
                adjoints[f.name] = (
                    f.name.removesuffix("and") + "erase"
                    if f.name.endswith("and")
                    else f.name.removesuffix("erase") + "and"
                )
        return selected, adder

    def unexpected_fallback(
        controls: tuple[Register, ...], patterns: tuple[ControlRegister, ...]
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        raise AssertionError(
            "The selected single-bit specialization must not use fallback"
        )

    module, adder = adjoint_component_control_system(
        forward,
        (Register("added", quantum_bit_type()),),
        (ControlRegister(1, (new,)),),
        frozenset((0, 1)),
        controlled,
        unexpected_fallback,
        adjoints,
        entry,
    )
    source, callback = source_existing_control_system(current, wrapper, old, new)
    return module, source, callback, child, adder, entry, selected_forward[0]


def inverse_source_gates(
    source: Any,
    child: Any,
    ports: SourcePorts,
    work: tuple[int, ...],
    bits: dict[int, int],
) -> tuple[Instruction, ...]:
    from qualtran.bloqs.mcmt.and_bloq import And

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        partition = source_partition_leaf(bloq, inputs, desired)
        if partition is not None:
            return partition
        if bloq == child.adjoint():
            return original_gates(bloq, inputs, work[-1:], bits), inputs
        if isinstance(bloq, And):
            outputs = {"ctrl": inputs["ctrl"]}
            if not bloq.uncompute:
                outputs["target"] = work[:1]
            gates = reference_arithmetic_gates(
                bloq,
                {**inputs, **outputs},
                (),
                bits,
                expected_outputs=outputs,
                cirq_style=True,
            )
            return gates, outputs
        return None

    gates, returned = walk_source_graph(source, ports, ports, leaf)
    assert returned == ports
    return gates


@pytest.mark.parametrize("layers", (1, 2))
@pytest.mark.parametrize("old,new", tuple(product((0, 1), repeat=2)))
@pytest.mark.parametrize("expanded", (False, True))
def test_whole_existing_control_adjoint(
    layers: int, old: int, new: int, expanded: bool
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    module, source, callback, child, adder, entry, forward = existing_adjoint_fixture(
        layers, True, old, new
    )
    assert verify_program_record(module) == ()
    observer = _source(Signature([]))
    assert tuple(
        observer.signature(Signature(module.functions[0].registers or ()))
    ) == tuple(source.signature)
    assert normalize_trace(_native_trace(module, entry)) == normalize_trace(
        _source_trace(observer, source.decompose_bloq())
    )
    signature = Signature(
        (
            Register("added", quantum_bit_type()),
            Register("control", RegisterType("opaque", 1)),
            Register("data", RegisterType("opaque", 3)),
        )
    )
    builder = ComponentProgramBuilder(signature, module.functions, name="caller")
    observer = _source(signature)
    values = {name: builder.inputs[name] for name in ("control", "data")}
    source_values = {name: observer.inputs[name] for name in ("control", "data")}
    added, data = adder(builder, (builder.inputs["added"],), values)
    src_added, src_data = callback(
        observer.builder, (observer.inputs["added"],), source_values
    )
    caller = builder.finalize(
        dict(zip((r.name for r in signature), (*added, *data), strict=True))
    )
    composite = observer.builder.finalize(
        **dict(zip((r.name for r in signature), (*src_added, *src_data), strict=True))
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
    assert tuple(values) == tuple(source_values)
    root = next(f for f in caller.functions if f.name == "caller")
    terminals = {
        op.callee: instance.bloq
        for op, (instance, _, _) in zip(
            root.blocks[0].operations, composite.iter_bloqsoqs(), strict=True
        )
        if op.callee is not None
    }
    counts = program_component_call_graph(
        caller, entry="caller", keep=terminals.__contains__
    ).leaf_counts
    assert (
        Counter({terminals[name]: count for name, count in counts.items()})
        == composite.bloq_counts()
    )
    model = get_cost_value(composite, QECGatesCost())
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
    if expanded:
        caller, _ = expand_component_calls(caller, "caller")
    bound = bind_register_program(
        caller, ((0,), (1,), (2, 3, 4)), clean=(5, 6), entry="caller"
    )
    assert bound.outputs == ((0,), (1,), (2, 3, 4))
    gates = bound.circuit.instructions
    bits = {
        g.wires[0]: g.metadata["classical_bit"] for g in gates if g.name == "measure"
    }
    assert_gate_parameters(
        gates,
        inverse_source_gates(
            composite,
            child,
            {"added": (0,), "control": (1,), "data": (2, 3, 4)},
            (5, 6),
            bits,
        ),
    )
    direct = bind_register_program(
        module, ((0,), (1,), (2, 3, 4)), clean=(5, 6), entry=entry
    )
    assert direct.circuit == bound.circuit
    emitted = Counter(g.name for g in gates)
    assert emitted["t"] + emitted["tdg"] == 8
    assert emitted["measure"] == emitted["reset"] == model.measurement == 2
    assert get_cost_value(composite, QubitCount()) == bound.circuit.n_wires == 7
    initial = torch.zeros(128, dtype=torch.complex128)
    initial[::4] = torch.tensor(
        [complex(i + 1, 5 - 2 * i) for i in range(32)], dtype=torch.complex128
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
                    * np.exp(-1j * np.pi * q / 4)
                    * (-1 if q and out else 1)
                    / 2**0.5
                )
    forward_gates = bind_register_program(
        forward, ((0,), (1,), (2, 3, 4)), clean=(5, 6), entry=entry
    ).circuit.instructions
    for second in product((0, 1), repeat=2):
        state = forced_branch(initial, gates, second)
        torch.testing.assert_close(state, expected / 2, atol=3e-12, rtol=0)
        assert torch.count_nonzero(state.reshape(-1, 4)[:, 1:]) == 0
        for first in product((0, 1), repeat=2):
            computed = forced_branch(initial, forward_gates, first)
            restored = forced_branch(computed, gates, second)
            torch.testing.assert_close(restored, initial / 4, atol=3e-12, rtol=0)


@pytest.mark.parametrize("layers", (1, 2))
@pytest.mark.parametrize("old,new", tuple(product((0, 1), repeat=2)))
def test_original_nonleading_existing_adjoint_callback_is_invalid(
    layers: int, old: int, new: int
) -> None:
    from qualtran import BloqError

    _, current, _ = packed_fixture(layers, False, False, old)
    _, wrapper, _ = packed_fixture(layers, False, False)
    with pytest.raises(
        BloqError, match="Finalizing register dtypes are not consistent"
    ):
        source_existing_control_system(current, wrapper, old, new)
    with pytest.raises(
        RegisterBindingError, match="Finalizing register dtypes are not consistent"
    ):
        existing_adjoint_fixture(layers, False, old, new)


@pytest.mark.parametrize("old,new", tuple(product((0, 1), repeat=2)))
def test_whole_packed_inverse_executes_through_append(old: int, new: int) -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    module, _, _, _, _, entry, _ = existing_adjoint_fixture(2, True, old, new)
    bound = bind_register_program(
        module, ((0,), (1,), (2, 3, 4)), clean=(5, 6), entry=entry
    )
    initial = torch.zeros(128, dtype=torch.complex128)
    initial[::4] = torch.tensor(
        [complex(i + 1, 4 - i) for i in range(32)], dtype=torch.complex128
    )
    initial /= initial.norm()
    expected = forced_branch(initial, bound.circuit.instructions, (0, 0)) * 2
    circuit = DynamicCircuit(7, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(7, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=8, seed=29, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), expected.expand(8, -1), atol=3e-12, rtol=0
    )
