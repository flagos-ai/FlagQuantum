"""Specialized adjoints reverse real returned adders and retain source channels."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_adjoint import adjoint_component_hierarchy
from flagquantum.compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
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
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_control import add_h, add_phase
from tests.unit.test_component_control_selection import (
    assert_gate_parameters,
    fixtures,
    native_fallback,
)
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_structured_control import normalize_trace, original_gates

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def add_inverse_phase(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    result = np.asarray(
        builder.call_dict(
            "cp_inverse", {"q": np.array([control, values["q"]], dtype=object)}
        )["q"]
    )
    return result[0], (result[1],)


@pytest.mark.parametrize(
    "mask,value,spectator,expanded",
    tuple(
        product(((), (0,), (1,), (0, 1)), (0, 1), ("spectator", "ctrl"), (False, True))
    ),
)
def test_specialized_masks_callback_graph_all_histories_and_complex_inverse(
    mask: tuple[int, ...], value: int, spectator: str, expanded: bool
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.mcmt.specialized_ctrl import (
        AdjointWithSpecializedCtrl,
        SpecializeOnCtrlBit,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    record, source, native, _ = fixtures(spectator, True)
    controls = (Register("added", quantum_bit_type()),)
    patterns = (ControlRegister(1, (value,)),)
    inverse = adjoint_component_hierarchy(record, {"h": "h", "p": "p_inverse"})
    inverse_fallback = native_fallback(
        inverse, {"h": add_h, "p_inverse": add_inverse_phase}
    )
    selected_calls = []
    fallback_calls = []

    def forward(
        controls: tuple[Register, ...], patterns: tuple[ControlRegister, ...]
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        selected_calls.append(True)
        return select_component_control_system(
            controls,
            patterns,
            native.get,
            native_fallback(record, {"h": add_h, "p": add_phase}),
        )

    def fallback(
        controls: tuple[Register, ...], patterns: tuple[ControlRegister, ...]
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        fallback_calls.append(True)
        return inverse_fallback(controls, patterns)

    module, adder = adjoint_component_control_system(
        record,
        controls,
        patterns,
        frozenset(mask),
        forward,
        fallback,
        {"ch": "ch", "cp": "cp_inverse", "__fq_predicate_0_x": "__fq_predicate_0_x"},
    )
    flag = SpecializeOnCtrlBit.NONE
    for bit in mask:
        flag |= SpecializeOnCtrlBit.ONE if bit else SpecializeOnCtrlBit.ZERO
    original, callback = AdjointWithSpecializedCtrl(source, flag).get_ctrl_system(
        CtrlSpec(cvs=value)
    )
    assert bool(selected_calls) == (value in mask)
    assert bool(fallback_calls) == (value not in mask)
    assert verify_program_record(module) == ()
    observer = _source(Signature([]))
    assert tuple(
        observer.signature(Signature(module.functions[0].registers or ()))
    ) == tuple(original.signature)
    signature = Signature((*controls, *(inverse.functions[0].registers or ())))
    builder = ComponentProgramBuilder(signature, module.functions, name="caller")
    upstream = _source(signature)
    values = {
        r.name: builder.inputs[r.name] for r in (inverse.functions[0].registers or ())
    }
    source_values = {
        r.name: upstream.inputs[r.name] for r in (inverse.functions[0].registers or ())
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
    if value in mask:
        caller, _ = expand_component_calls_once(
            caller,
            "caller",
            predicate=lambda _, op: op.callee is not None
            and op.callee.startswith("__fq_adjoint_control_"),
        )
    assert normalize_trace(_native_trace(caller, "caller")) == normalize_trace(
        _source_trace(upstream, composite)
    )
    assert tuple(values) == tuple(source_values)
    bound = bind_register_program(
        caller, ((0,), (1,), (2,)), clean=(3,), entry="caller"
    )
    direct = bind_register_program(module, ((0,), (1,), (2,)), clean=(3,))
    assert direct.circuit == bound.circuit
    gates = bound.circuit.instructions
    bits = {
        g.wires[0]: g.metadata["classical_bit"] for g in gates if g.name == "measure"
    }
    ports: dict[str, tuple[int, ...]] = {
        r.name: wires
        for r, wires in zip(original.signature, ((0,), (1,), (2,)), strict=True)
    }
    assert_gate_parameters(gates, original_gates(original, ports, (3,), bits))
    model = get_cost_value(original, QECGatesCost())
    assert model.measurement == 1 and model.total_t_count() == 27
    assert get_cost_value(original, QubitCount()) == 4
    if expanded:
        from flagquantum.compiler.registers.component_expansion import (
            expand_component_calls,
        )

        flat, _ = expand_component_calls(caller, "caller")
        assert (
            bind_register_program(
                flat, ((0,), (1,), (2,)), clean=(3,), entry="caller"
            ).circuit
            == bound.circuit
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
                    * np.exp(-1j * np.pi * q / 4)
                    * ((-1 if q and out else 1) / 2**0.5)
                )
            elif out == q:
                expected[((ctrl << 2) | (q << 1) | spare) << 1] += amp
    forward_gates = bind_register_program(
        native[value][0], ((0,), (1,), (2,)), clean=(3,)
    ).circuit.instructions
    for first, second in product((0, 1), repeat=2):
        computed = forced_branch(initial, forward_gates, (first,))
        restored = forced_branch(computed, gates, (second,))
        torch.testing.assert_close(restored, initial / 2, atol=2e-12, rtol=0)
    for history in product((0, 1), repeat=1):
        actual = forced_branch(initial, gates, history)
        torch.testing.assert_close(actual, expected / 2**0.5, atol=2e-12, rtol=0)
        assert torch.count_nonzero(actual[1::2]) == 0


@pytest.mark.parametrize("value", (0, 1))
def test_append_specialized_inverse_matches_all_dynamic_trajectories(
    value: int,
) -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    record, _, native, _ = fixtures("spectator", True)

    def forward(
        controls: tuple[Register, ...], patterns: tuple[ControlRegister, ...]
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        return select_component_control_system(
            controls,
            patterns,
            native.get,
            native_fallback(record, {"h": add_h, "p": add_phase}),
        )

    module, _ = adjoint_component_control_system(
        record,
        (Register("added", quantum_bit_type()),),
        (ControlRegister(1, (value,)),),
        frozenset((0, 1)),
        forward,
        forward,
        {"ch": "ch", "cp": "cp_inverse", "__fq_predicate_0_x": "__fq_predicate_0_x"},
    )
    bound = bind_register_program(module, ((0,), (1,), (2,)), clean=(3,))
    initial = torch.zeros(16, dtype=torch.complex128)
    initial[::2] = torch.tensor(
        [complex(i + 1, 3 - i) for i in range(8)], dtype=torch.complex128
    )
    initial /= initial.norm()
    expected = forced_branch(initial, bound.circuit.instructions, (0,)) * 2**0.5
    circuit = DynamicCircuit(4, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(4, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=8, seed=29, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), expected.expand(8, -1), atol=2e-12, rtol=0
    )


@pytest.mark.parametrize("mask", ((-1,), (2,), (True,)))
def test_invalid_specialization_mask_rejects(mask: tuple[int, ...]) -> None:
    record, _, _, _ = fixtures("spectator", True)
    fallback = native_fallback(record, {"h": add_h, "p": add_phase})
    with pytest.raises(ValueError, match="binary"):
        adjoint_component_control_system(
            record,
            (Register("added", quantum_bit_type()),),
            (ControlRegister(1, (1,)),),
            frozenset(mask),
            fallback,
            fallback,
            {},
        )


@pytest.mark.parametrize("count", (2, 3))
def test_multibit_adjoint_request_delegates_without_forward_query(count: int) -> None:
    record, _, _, _ = fixtures("spectator", True)
    controls = (Register("added", quantum_bit_type(), (count,)),)
    patterns = (ControlRegister(1, (0,) * count),)
    expected = native_fallback(record, {"h": add_h, "p": add_phase})
    seen = []

    def forward(
        controls: tuple[Register, ...], patterns: tuple[ControlRegister, ...]
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        pytest.fail("multi-bit adjoint pushed controls into forward specialization")

    def fallback(
        actual: tuple[Register, ...], activations: tuple[ControlRegister, ...]
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        assert actual is controls and activations is patterns
        seen.append(True)
        return expected(actual, activations)

    module, _ = adjoint_component_control_system(
        record, controls, patterns, frozenset((0, 1)), forward, fallback, {}
    )
    assert seen == [True]
    assert verify_program_record(module) == ()


def test_local_adapter_names_avoid_existing_functions() -> None:
    record, _, native, _ = fixtures("spectator", True)
    positive, name = native[1]
    sentinel = positive.functions[-1]
    positive = replace(
        positive,
        functions=(
            *positive.functions,
            replace(sentinel, name="__fq_adjoint_control_0"),
            replace(sentinel, name="__fq_adjoint_control_1_inverse"),
        ),
    )
    native[1] = (positive, name)

    def forward(
        controls: tuple[Register, ...], patterns: tuple[ControlRegister, ...]
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        return select_component_control_system(
            controls,
            patterns,
            native.get,
            native_fallback(record, {"h": add_h, "p": add_phase}),
        )

    module, _ = adjoint_component_control_system(
        record,
        (Register("added", quantum_bit_type()),),
        (ControlRegister(1, (1,)),),
        frozenset((1,)),
        forward,
        forward,
        {"ch": "ch", "cp": "cp_inverse"},
    )
    assert verify_program_record(module) == ()
    assert any(f.name == "__fq_adjoint_control_1" for f in module.functions)
    assert (
        bind_register_program(module, ((0,), (1,), (2,)), clean=(3,)).circuit.n_wires
        == 4
    )


def test_adjoint_reverses_nested_returned_adder_before_inlining() -> None:
    from qualtran import Bloq, CtrlSpec
    from qualtran.bloqs.mcmt.specialized_ctrl import (
        AdjointWithSpecializedCtrl,
        SpecializeOnCtrlBit,
    )

    record, source, native, originals = fixtures("spectator", True)
    selected, ctrl_name = native[1]
    signature = Signature(selected.functions[0].registers or ())
    wrapper = ComponentProgramBuilder(
        signature, selected.functions, name="stored_adder"
    )
    extra = wrapper.finalize(wrapper.call_dict("main", wrapper.inputs)).functions[0]
    selected = replace(selected, functions=(*selected.functions, extra))
    upstream_wrapper = _source(signature)
    original_wrapper = upstream_wrapper.builder.finalize(
        **upstream_wrapper.builder.add_d(originals[1], **upstream_wrapper.inputs)
    )

    def forward(
        controls: tuple[Register, ...], patterns: tuple[ControlRegister, ...]
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        def adder(
            builder: ComponentProgramBuilder,
            added: tuple[BindingCollection, ...],
            values: dict[str, BindingCollection],
        ) -> tuple[tuple[BindingCollection, ...], tuple[BindingCollection, ...]]:
            (control,) = added
            values[ctrl_name] = control
            outputs = builder.call_dict("stored_adder", values)
            return (outputs.pop(ctrl_name),), tuple(outputs.values())

        return selected, adder

    def original_system(component: object, ctrl_spec: object) -> Any:
        def adder(builder: Any, added: Any, values: Any) -> Any:
            (control,) = added
            values[ctrl_name] = control
            outputs = builder.add_d(original_wrapper, **values)
            return (outputs.pop(ctrl_name),), tuple(outputs.values())

        return originals[1], adder

    def decompose(component: object) -> Any:
        return source.decompose_bloq()

    wrapped = type(
        "StoredAdderBody",
        (Bloq,),
        {
            "signature": source.signature,
            "decompose_bloq": decompose,
            "get_ctrl_system": original_system,
        },
    )()
    module, adder = adjoint_component_control_system(
        record,
        (Register("added", quantum_bit_type()),),
        (ControlRegister(1, (1,)),),
        frozenset((1,)),
        forward,
        forward,
        {"ch": "ch", "cp": "cp_inverse"},
    )
    _, original_adder = AdjointWithSpecializedCtrl(
        wrapped, SpecializeOnCtrlBit.ONE
    ).get_ctrl_system(CtrlSpec())
    signature = Signature(
        (Register("added", quantum_bit_type()), *(record.functions[0].registers or ()))
    )
    builder = ComponentProgramBuilder(signature, module.functions, name="caller")
    upstream = _source(signature)
    values = {
        r.name: builder.inputs[r.name] for r in (record.functions[0].registers or ())
    }
    source_values = {
        r.name: upstream.inputs[r.name] for r in (record.functions[0].registers or ())
    }
    added, data = adder(builder, (builder.inputs["added"],), values)
    src_added, src_data = original_adder(
        upstream.builder, (upstream.inputs["added"],), source_values
    )
    caller = builder.finalize(
        dict(zip((r.name for r in signature), (*added, *data), strict=True))
    )
    composite = upstream.builder.finalize(
        **dict(zip((r.name for r in signature), (*src_added, *src_data), strict=True))
    )
    caller, _ = expand_component_calls_once(
        caller,
        "caller",
        predicate=lambda _, op: op.callee is not None
        and op.callee.startswith("__fq_adjoint_control_"),
    )
    assert normalize_trace(_native_trace(caller, "caller")) == normalize_trace(
        _source_trace(upstream, composite)
    )
    assert verify_program_record(caller) == ()
    gates = bind_register_program(
        caller, ((0,), (1,), (2,)), clean=(3,), entry="caller"
    ).circuit.instructions
    simple = adjoint_component_hierarchy(native[1][0], {"ch": "ch", "cp": "cp_inverse"})
    assert (
        gates
        == bind_register_program(
            simple, ((0,), (1,), (2,)), clean=(3,)
        ).circuit.instructions
    )
