"""Original single-control dispatch selects the same stored bodies and fallbacks."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from flagquantum.compiler.registers.component_control import (
    ControlAdder,
    _control_names,
    control_component_hierarchy,
    control_component_via_predicate,
)
from flagquantum.compiler.registers.control.component_control_selection import (
    ComponentControlSystem,
    select_component_control_system,
)
from flagquantum.compiler.registers.control.component_extension import (
    StructuredControlAdder,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.primitives.controlled_rotation import (
    controlled_z_power_ir,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_control import add_h, add_phase, body, fixture_program
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_structured_control import normalize_trace, original_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def native_fallback(
    record: ProgramRecord, mapping: dict[str, ControlAdder]
) -> ComponentControlSystem:
    def system(
        controls: tuple[Register, ...], patterns: tuple[ControlRegister, ...]
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        registers = record.functions[0].registers
        assert registers is not None
        names = _control_names(registers, len(controls))
        actual = tuple(
            Register(name, r.dtype, r.shape)
            for name, r in zip(names, controls, strict=True)
        )
        if len(actual) == 1 and not actual[0].shape and patterns[0].bits() == (1,):
            module = control_component_hierarchy(record, mapping)
        else:
            module = control_component_via_predicate(record, mapping, actual, patterns)

        def add(
            builder: ComponentProgramBuilder,
            added: tuple[BindingCollection, ...],
            values: dict[str, BindingCollection],
        ) -> tuple[tuple[BindingCollection, ...], tuple[BindingCollection, ...]]:
            values |= dict(zip(names, added, strict=False))
            outputs = builder.call_tuple("main", values)
            return outputs[: len(names)], outputs[len(names) :]

        return module, add

    return system


def fixtures(
    spectator: str, both: bool
) -> tuple[ProgramRecord, Any, dict[int, tuple[ProgramRecord, str]], dict[int, Any]]:
    from qualtran import Bloq, CtrlSpec
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
    from qualtran.bloqs.mcmt.specialized_ctrl import _get_ctrl_system_1bit_cv

    record, controlled = fixture_program(1, spectator)
    bit = quantum_bit_type()
    record = replace(
        record,
        functions=(
            *record.functions,
            *body(
                "p_inverse",
                (Register("q", bit),),
                CircuitIR(1, (Instruction("phase", (0,), {"theta": -np.pi / 4}),)),
            ),
            *body(
                "cp_inverse",
                (Register("q", bit, (2,)),),
                CircuitIR(3, controlled_z_power_ir(0, 1, -0.25, 2, 0)),
            ),
        ),
    )
    positive = control_component_hierarchy(record, {"h": add_h, "p": add_phase})
    native = {1: (positive, controlled.ctrl_reg_names[0])}
    originals = {1: controlled}
    if both:
        native[0] = (
            control_component_via_predicate(
                record,
                {"h": add_h, "p": add_phase},
                (Register(controlled.ctrl_reg_names[0], bit),),
                (ControlRegister(1, (0,)),),
            ),
            controlled.ctrl_reg_names[0],
        )
        originals[0] = ControlledViaAnd(controlled.subbloq, CtrlSpec(cvs=0))

    def decompose(component: object) -> Any:
        return controlled.subbloq

    def system(component: Any, ctrl_spec: Any) -> Any:
        return _get_ctrl_system_1bit_cv(
            component,
            ctrl_spec,
            current_ctrl_bit=None,
            get_ctrl_bloq_and_ctrl_reg_name=lambda cv: (
                (originals[cv], controlled.ctrl_reg_names[0])
                if cv in originals
                else None
            ),
        )

    source = type(
        "StoredSpecializedBody",
        (Bloq,),
        {
            "signature": controlled.subbloq.signature,
            "decompose_bloq": decompose,
            "get_ctrl_system": system,
        },
    )()
    return record, source, native, originals


def assert_gate_parameters(
    actual: tuple[Instruction, ...], expected: tuple[Instruction, ...]
) -> None:
    assert_source(actual, expected)
    assert [(g.name, g.wires, dict(g.params)) for g in actual if g.params] == [
        (g.name, g.wires, dict(g.params)) for g in expected if g.params
    ]


@pytest.mark.parametrize(
    "both,value,spectator", tuple(product((False, True), (0, 1), ("spectator", "ctrl")))
)
def test_direct_selection_and_negative_fallback_match_original(
    both: bool, value: int, spectator: str
) -> None:
    from qualtran import CtrlSpec

    record, source, native, originals = fixtures(spectator, both)
    controls = (Register("added", quantum_bit_type()),)
    patterns = (ControlRegister(1, (value,)),)
    queries = []
    fallback_calls = []
    fallback = native_fallback(record, {"h": add_h, "p": add_phase})

    def get(value: int) -> tuple[ProgramRecord, str] | None:
        queries.append(value)
        return native.get(value)

    def default(
        controls: tuple[Register, ...], patterns: tuple[ControlRegister, ...]
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        fallback_calls.append(True)
        return fallback(controls, patterns)

    module, adder = select_component_control_system(controls, patterns, get, default)
    original, callback = source.get_ctrl_system(CtrlSpec(cvs=value))
    assert queries == [value]
    assert bool(fallback_calls) == (value == 0 and not both)
    observer = _source(Signature([]))
    assert tuple(
        observer.signature(Signature(module.functions[0].registers or ()))
    ) == tuple(original.signature)
    signature = Signature((*controls, *(record.functions[0].registers or ())))
    builder = ComponentProgramBuilder(signature, module.functions, name="caller")
    upstream = _source(signature)
    values = {
        r.name: builder.inputs[r.name] for r in (record.functions[0].registers or ())
    }
    source_values = {
        r.name: upstream.inputs[r.name] for r in (record.functions[0].registers or ())
    }
    added, data = adder(builder, (builder.inputs["added"],), values)
    src_added, src_data = callback(
        upstream.builder, (upstream.inputs["added"],), source_values
    )
    caller = builder.finalize(
        dict(zip((r.name for r in signature), (*added, *data), strict=True))
    )
    composite = upstream.builder.finalize(
        **dict(zip((r.name for r in signature), (*src_added, *src_data), strict=True))
    )
    assert normalize_trace(_native_trace(caller, "caller")) == normalize_trace(
        _source_trace(upstream, composite)
    )
    assert tuple(values) == tuple(source_values)
    bound = bind_register_program(
        caller, ((0,), (1,), (2,)), clean=(3,), entry="caller"
    )
    gates = bound.circuit.instructions
    bits = {
        g.wires[0]: g.metadata["classical_bit"] for g in gates if g.name == "measure"
    }
    assert_gate_parameters(
        gates,
        original_gates(
            original,
            {next(iter(original.signature)).name: (0,), "q": (1,), spectator: (2,)},
            (3,),
            bits,
        ),
    )
    initial = torch.zeros(16, dtype=torch.complex128)
    initial[::2] = torch.tensor(
        [complex(i + 1, 2 - i) for i in range(8)], dtype=torch.complex128
    )
    initial /= initial.norm()
    expected = torch.zeros_like(initial)
    for ctrl, q, spare in product((0, 1), repeat=3):
        amp = initial[((ctrl << 2) | (q << 1) | spare) << 1]
        for out in (0, 1):
            if ctrl == value:
                expected[((ctrl << 2) | (out << 1) | spare) << 1] += (
                    amp
                    * ((-1 if q and out else 1) / 2**0.5)
                    * np.exp(1j * np.pi * out / 4)
                )
            elif out == q:
                expected[((ctrl << 2) | (q << 1) | spare) << 1] += amp
    for history in product((0, 1), repeat=1):
        torch.testing.assert_close(
            forced_branch(initial, gates, history),
            expected / 2**0.5,
            atol=2e-12,
            rtol=0,
        )


@pytest.mark.parametrize("count,value", tuple(product((2, 3, 5), (0, 1))))
def test_multi_bit_dispatch_never_queries_specialization(
    count: int, value: int
) -> None:
    record, source, _, _ = fixtures("spectator", True)
    controls = (Register("added", quantum_bit_type(), (count,)),)
    patterns = (ControlRegister(1, (value,) * count),)

    def unexpected(value: int) -> tuple[ProgramRecord, str] | None:
        pytest.fail("multi-bit dispatch queried single-bit specialization")

    module, _ = select_component_control_system(
        controls,
        patterns,
        unexpected,
        native_fallback(record, {"h": add_h, "p": add_phase}),
    )
    from qualtran import CtrlSpec

    original, _ = source.get_ctrl_system(CtrlSpec(cvs=[value] * count))
    observer = _source(Signature([]))
    assert tuple(
        observer.signature(Signature(module.functions[0].registers or ()))
    ) == tuple(original.signature)
    assert normalize_trace(_native_trace(module)) == normalize_trace(
        _source_trace(observer, original.decompose_bloq())
    )


@pytest.mark.parametrize("value,current", tuple(product((0, 1), (0, 1))))
def test_existing_control_selects_only_positive(value: int, current: int) -> None:
    record, source, native, _ = fixtures("spectator", True)
    controls = (Register("added", quantum_bit_type()),)
    patterns = (ControlRegister(1, (value,)),)
    queried = []

    def get(value: int) -> tuple[ProgramRecord, str] | None:
        queried.append(value)
        return native[value]

    module, adder = select_component_control_system(
        controls,
        patterns,
        get,
        native_fallback(record, {"h": add_h, "p": add_phase}),
        current,
    )
    assert queried == [1]
    registers = module.functions[0].registers
    assert registers is not None and registers[0].shape == (2,)
    positive, control_name = native[1]
    signature = Signature((controls[0], *(positive.functions[0].registers or ())))
    builder = ComponentProgramBuilder(signature, module.functions, name="caller")
    values = {
        r.name: builder.inputs[r.name] for r in (positive.functions[0].registers or ())
    }
    added, data = adder(builder, (builder.inputs["added"],), values)
    caller = builder.finalize(
        dict(zip((r.name for r in signature), (*added, *data), strict=True))
    )
    direct = bind_register_program(module, tuple((i,) for i in range(4)), clean=(4, 5))
    actual = bind_register_program(
        caller, tuple((i,) for i in range(4)), clean=(4, 5), entry="caller"
    )
    assert actual.circuit == direct.circuit
    assert np.asarray(values[control_name]).shape == (2,)


@pytest.mark.parametrize("current", (None, 0, 1))
def test_missing_positive_specialization_rejects(current: int | None) -> None:
    controls = (Register("added", quantum_bit_type()),)
    patterns = (ControlRegister(1, (1,)),)

    def absent(value: int) -> tuple[ProgramRecord, str] | None:
        return None

    def unexpected(
        controls: tuple[Register, ...], patterns: tuple[ControlRegister, ...]
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        pytest.fail("missing positive specialization entered fallback")

    with pytest.raises(ValueError, match="positive"):
        select_component_control_system(controls, patterns, absent, unexpected, current)


@pytest.mark.parametrize("current", (-1, 2, True))
def test_invalid_existing_activation_rejects(current: int) -> None:
    record, _, native, _ = fixtures("spectator", True)
    with pytest.raises(ValueError, match="activation"):
        select_component_control_system(
            (Register("added", quantum_bit_type()),),
            (ControlRegister(1, (1,)),),
            native.get,
            native_fallback(record, {"h": add_h, "p": add_phase}),
            current,
        )


@pytest.mark.parametrize("count", (0, 2))
def test_direct_callback_preserves_original_wrong_arity_failure(count: int) -> None:
    from qualtran import CtrlSpec

    record, source, native, _ = fixtures("spectator", True)
    module, adder = select_component_control_system(
        (Register("added", quantum_bit_type()),),
        (ControlRegister(1, (1,)),),
        native.get,
        native_fallback(record, {"h": add_h, "p": add_phase}),
    )
    _, original = source.get_ctrl_system(CtrlSpec())
    signature = Signature(
        (Register("added", quantum_bit_type()), *(record.functions[0].registers or ()))
    )
    builder = ComponentProgramBuilder(signature, module.functions, name="caller")
    observer = _source(signature)
    values = {
        r.name: builder.inputs[r.name] for r in (record.functions[0].registers or ())
    }
    original_values = {
        r.name: observer.inputs[r.name] for r in (record.functions[0].registers or ())
    }
    with pytest.raises(ValueError):
        adder(builder, (builder.inputs["added"],) * count, values)
    with pytest.raises(ValueError):
        original(observer.builder, (observer.inputs["added"],) * count, original_values)
    assert tuple(values) == tuple(original_values) == ("q", "spectator")


def test_mixed_control_fallback_closes_graph_resources_and_all_branches() -> None:
    from collections import Counter

    from qualtran import CtrlSpec, QBit, QUInt
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler._component_trace.program_call_graph import (
        program_component_call_graph,
    )
    from flagquantum.core.ir.program.registers import RegisterType

    record, source, native, _ = fixtures("spectator", True)
    controls = (
        Register("a", quantum_bit_type()),
        Register("b", RegisterType("uint", 2)),
    )
    patterns = (ControlRegister(1, (0,)), ControlRegister(2, (2,)))
    module, _ = select_component_control_system(
        controls,
        patterns,
        native.get,
        native_fallback(record, {"h": add_h, "p": add_phase}),
    )
    original, _ = source.get_ctrl_system(
        CtrlSpec(qdtypes=(QBit(), QUInt(2)), cvs=(np.array(0), np.array(2)))
    )
    observer = _source(Signature([]))
    assert tuple(
        observer.signature(Signature(module.functions[0].registers or ()))
    ) == tuple(original.signature)
    assert normalize_trace(_native_trace(module)) == normalize_trace(
        _source_trace(observer, original.decompose_bloq())
    )
    terminals = {
        op.callee: instance.bloq
        for op, (instance, _, _) in zip(
            module.functions[0].blocks[0].operations,
            original.decompose_bloq().iter_bloqsoqs(),
            strict=True,
        )
        if op.callee is not None
    }
    counts = program_component_call_graph(
        module, keep=terminals.__contains__
    ).leaf_counts
    assert (
        Counter({terminals[name]: count for name, count in counts.items()})
        == original.bloq_counts()
    )
    model = get_cost_value(original, QECGatesCost())
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
    bound = bind_register_program(module, ((0,), (1, 2), (3,), (4,)), clean=(5, 6, 7))
    assert bound.circuit.n_wires == get_cost_value(original, QubitCount()) == 8
    gates = bound.circuit.instructions
    bits = {
        g.wires[0]: g.metadata["classical_bit"] for g in gates if g.name == "measure"
    }
    ports = {
        original.ctrl_reg_names[0]: (0,),
        original.ctrl_reg_names[1]: (1, 2),
        "q": (3,),
        "spectator": (4,),
    }
    assert_gate_parameters(gates, original_gates(original, ports, (5, 6, 7), bits))
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[::8] = torch.tensor(
        [complex(i + 1, 3 - i) for i in range(32)], dtype=torch.complex128
    )
    initial /= initial.norm()
    expected = torch.zeros_like(initial)
    for ctrl, q, spare in product(range(8), (0, 1), (0, 1)):
        amp = initial[((ctrl << 2) | (q << 1) | spare) << 3]
        if ctrl == 2:
            for out in (0, 1):
                expected[((ctrl << 2) | (out << 1) | spare) << 3] += (
                    amp
                    * (-1 if q and out else 1)
                    / 2**0.5
                    * np.exp(1j * np.pi * out / 4)
                )
        else:
            expected[((ctrl << 2) | (q << 1) | spare) << 3] = amp
    for history in product((0, 1), repeat=3):
        actual = forced_branch(initial, gates, history)
        torch.testing.assert_close(actual, expected / 2**1.5, atol=2e-12, rtol=0)
        assert torch.count_nonzero(actual.reshape(-1, 8)[:, 1:]) == 0
