"""Actual packed control callbacks retain directional ports and source semantics."""

from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.control.component_control_selection import (
    select_component_control_system,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_auto_partition import partition_gates
from tests.unit.test_component_control_selection import assert_gate_parameters
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_structured_control import normalize_trace
from tests.unit.test_directional_control_callbacks import directional_fixture

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_system(source: Any, child: Any, value: int) -> tuple[Any, Any]:
    from qualtran import CtrlSpec
    from qualtran.bloqs.mcmt.specialized_ctrl import _get_ctrl_system_1bit_cv

    returned, callback = _get_ctrl_system_1bit_cv(
        child.subbloq,
        CtrlSpec(cvs=value),
        current_ctrl_bit=None,
        get_ctrl_bloq_and_ctrl_reg_name=lambda cv: (source, "ctrl"),
    )
    return returned, callback


def unexpected_fallback(*args: Any) -> Any:
    raise AssertionError("direct specialization must not query fallback")


def coherent_input(width: int, work: int) -> torch.Tensor:
    initial = torch.zeros(1 << (width + work), dtype=torch.complex128)
    initial[:: 1 << work] = torch.tensor(
        [complex(i + 1, 2 - i) for i in range(1 << width)], dtype=torch.complex128
    )
    initial /= initial.norm()
    return initial


def expected_controlled(
    initial: torch.Tensor, values: tuple[int, ...], work: int
) -> torch.Tensor:
    expected = torch.zeros_like(initial)
    for controls in product((0, 1), repeat=len(values)):
        word = sum(v << i for i, v in enumerate(reversed(controls)))
        for q, spare in product((0, 1), repeat=2):
            index = (word * 4 + q * 2 + spare) << work
            if controls != values:
                expected[index] += initial[index]
                continue
            for out in (0, 1):
                target = (word * 4 + out * 2 + spare) << work
                expected[target] += (
                    initial[index]
                    * (-1 if q and out else 1)
                    / 2**0.5
                    * np.exp(1j * np.pi * out / 4)
                )
    return expected


@pytest.mark.parametrize("value,expanded", tuple(product((0, 1), repeat=2)))
def test_actual_direct_callback_keeps_wrapped_entry(value: int, expanded: bool) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    module, source, child = directional_fixture(1, True, False, value)
    selected, adder = select_component_control_system(
        (Register("added", quantum_bit_type()),),
        (ControlRegister(1, (value,)),),
        lambda cv: (module, "ctrl"),
        unexpected_fallback,
        entry="packed0",
    )
    returned, callback = source_system(source, child, value)
    assert returned == child and returned != source
    assert selected == module
    signature = Signature(
        (
            Register("added", quantum_bit_type()),
            Register("data", RegisterType("opaque", 2), side=RegisterSide.LEFT),
            Register("q", quantum_bit_type(), side=RegisterSide.RIGHT),
            Register("spectator", quantum_bit_type(), side=RegisterSide.RIGHT),
        )
    )
    builder = ComponentProgramBuilder(signature, module.functions, name="caller")
    observer = _source(signature)
    values = {"data": builder.inputs["data"]}
    src_values = {"data": observer.inputs["data"]}
    controls, data = adder(builder, (builder.inputs["added"],), values)
    src_controls, src_data = callback(
        observer.builder, (observer.inputs["added"],), src_values
    )
    names = tuple(r.name for r in signature.rights())
    caller = builder.finalize(dict(zip(names, (*controls, *data), strict=True)))
    composite = observer.builder.finalize(
        **dict(zip(names, (*src_controls, *src_data), strict=True))
    )
    assert tuple(values) == tuple(src_values) == ("data", "ctrl")
    assert verify_program_record(caller) == ()
    assert normalize_trace(_native_trace(caller, "caller")) == normalize_trace(
        _source_trace(observer, composite)
    )
    if expanded:
        caller, _ = expand_component_calls(caller, "caller")
    bound = bind_register_program(caller, ((0,), (1, 2)), clean=(3,), entry="caller")
    assert bound.outputs == ((0,), (1,), (2,))
    assert not any(e.kind == "free" and set(e.wires) & {0, 1, 2} for e in bound.events)
    bits = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in bound.circuit.instructions
        if g.name == "measure"
    }
    assert_gate_parameters(
        bound.circuit.instructions,
        partition_gates(
            composite,
            child,
            {"added": (0,), "data": (1, 2)},
            {"added": (0,), "q": (1,), "spectator": (2,)},
            (3,),
            bits,
        ),
    )
    initial = coherent_input(3, 1)
    expected = expected_controlled(initial, (value,), 1)
    for outcome in (0, 1):
        state = forced_branch(initial, bound.circuit.instructions, (outcome,))
        torch.testing.assert_close(state, expected / 2**0.5, atol=2e-12, rtol=0)
        assert torch.count_nonzero(state[1::2]) == 0
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts["t"] + counts["tdg"] == 4
    assert counts["measure"] == counts["reset"] == 1
    model = get_cost_value(composite, QECGatesCost())
    assert model.total_t_count() == 27
    assert model.measurement == 1
    assert get_cost_value(composite, QubitCount()) == bound.circuit.n_wires == 4


@pytest.mark.parametrize("value", (0, 1))
def test_direct_callback_remains_appendable(value: int) -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    module, _, _ = directional_fixture(1, True, False, value)
    bound = bind_register_program(module, ((0,), (1, 2)), clean=(3,), entry="packed0")
    initial = coherent_input(3, 1)
    circuit = DynamicCircuit(4, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(4, bound.circuit.instructions))
    actual = run_dynamic(circuit, shots=8, seed=31, strategy="trajectory")
    torch.testing.assert_close(
        actual.final_states.reshape(8, -1),
        expected_controlled(initial, (value,), 1).expand(8, -1),
        atol=2e-12,
        rtol=0,
    )
