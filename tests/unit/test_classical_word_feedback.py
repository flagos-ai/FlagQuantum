"""Typed word feedback matches source action through the normal append consumer."""

from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Any

import pytest

try:
    import numpy as np
except ModuleNotFoundError:
    pytest.importorskip("numpy")
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.classical_word_storage import classical_word_bits
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_classical_control import (
    classical_control_component_program,
)
from flagquantum.compiler.registers.component_feedback import (
    classical_bit_feedback_function,
)
from flagquantum.compiler.registers.component_measurement import measure_x_word_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import (
    classical_fixed_point_register_type,
    classical_montgomery_register_type,
    classical_ones_complement_register_type,
    classical_signed_register_type,
    classical_unsigned_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import assert_source_wire_streams
from tests.unit.test_component_classical_control import hierarchy, source_stream

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CASES = (
    ("uint", tuple(range(4))),
    ("signed", tuple(range(-2, 2))),
    ("ones_complement", (-1, 0, 1)),
    ("montgomery", tuple(range(4))),
    ("fixed_point", tuple(range(-2, 2))),
)


def types(kind: str) -> tuple[RegisterType, Any]:
    from qualtran import CFxp, CInt, CIntOnesComp, CMontgomeryUInt, CUInt

    return {
        "uint": (classical_unsigned_register_type(2), CUInt(2)),
        "signed": (classical_signed_register_type(2), CInt(2)),
        "ones_complement": (
            classical_ones_complement_register_type(2),
            CIntOnesComp(2),
        ),
        "montgomery": (classical_montgomery_register_type(2, 3), CMontgomeryUInt(2, 3)),
        "fixed_point": (
            classical_fixed_point_register_type(2, 1, True),
            CFxp(2, 1, signed=True),
        ),
    }[kind]


@pytest.mark.parametrize("kind,domain", CASES)
def test_all_activation_values_source_tensor_child_stream_and_clean_work(
    kind: str, domain: tuple[int, ...]
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    dtype, source_type = types(kind)
    module, source_child = hierarchy()
    original = source_stream(source_child)
    state = torch.complex(torch.arange(1, 9).double(), torch.arange(8).double())
    state /= state.norm()
    initial = torch.zeros(16, dtype=torch.complex128)
    initial.reshape(4, 2, 2)[:, 0, :] = state.reshape(4, 2)
    for activation in domain:
        controlled = classical_control_component_program(
            module, (((), (activation,)),), control_types=(dtype,)
        )
        assert verify_program_record(controlled) == ()
        source = ClassicallyControlled(source_child, CtrlSpec(source_type, activation))
        tensor = source._tensor_data().reshape(4, 4, 4, 4)
        with pytest.raises(NotImplementedError):
            source.decompose_bloq()
        with pytest.raises(NotImplementedError):
            get_cost_value(source, QECGatesCost())
        for value in domain:
            bound = bind_register_program(
                controlled,
                ((), (0,), (1,)),
                classical_inputs=(value, None, None),
                clean=(2,),
            )
            bits = classical_word_bits(dtype, value)
            assert Counter(g.name for g in bound.circuit.instructions) == {
                "classical_xor": 2,
                "cx": 4,
                "phase": 2,
                "h": 2,
            }
            assert bound.classical_outputs == ((0, 1), (), ())
            assert len(bound.events) == 4
            # At the comparable child boundary every primitive and operand is unchanged.
            stripped = tuple(
                replace(g, metadata={}) for g in bound.circuit.instructions[2:]
            )
            assert_source_wire_streams(stripped, original)
            circuit = DynamicCircuit(4, inputs=initial, dtype=torch.complex128)
            circuit.append(FieldOperation(4, bound.circuit.instructions))
            result = run_dynamic(circuit, shots=1, strategy="trajectory")
            index = 2 * bits[0] + bits[1]
            matrix = torch.as_tensor(
                tensor[index, :, index, :].copy(), dtype=torch.complex128
            )
            if kind == "ones_complement":
                # Source tensor indexing uses two's-complement indices for negative
                # ones'-complement values. Preserve its defect as separate evidence
                # while following the approved numerical activation contract.
                enabled = source.ctrl_spec.is_active(value)
                child_matrix = torch.as_tensor(
                    source_child.tensor_contract(), dtype=torch.complex128
                )
                action = (
                    child_matrix if enabled else torch.eye(4, dtype=torch.complex128)
                )
                expected = (action @ state.reshape(4, 2)).flatten()
                if activation == value == -1:
                    assert not torch.allclose(matrix, action)
                else:
                    torch.testing.assert_close(matrix, action, rtol=0, atol=4e-14)
            else:
                expected = (matrix @ state.reshape(4, 2)).flatten()
            actual = result.final_states[0].reshape(4, 2, 2)
            torch.testing.assert_close(
                actual[:, 0, :].flatten(), expected, rtol=0, atol=4e-14
            )
            assert actual[:, 1, :].abs().max().item() < 4e-14
            assert result.classical_bits[0].tolist() == list(bits)


def test_ones_complement_negative_zero_raw_read_matches_numerical_activation() -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled

    dtype, source_type = types("ones_complement")
    child = classical_bit_feedback_function(
        "main",
        (Register("q", quantum_bit_type()),),
        CircuitIR(1, (Instruction("x", (0,)),)),
        (),
    )
    record = classical_control_component_program(
        ProgramRecord((child,)), (((), (0,)),), control_types=(dtype,)
    )
    bound = bind_register_program(record, ((), (0,)), classical_inputs=(0, None))
    assert bound.circuit.instructions[-1].metadata["condition_clauses"] == (
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
    )
    source = ClassicallyControlled(XGate(), CtrlSpec(source_type, 0))
    state = torch.tensor([1 + 2j, 2 - 1j, -3j, 4 + 1j], dtype=torch.complex128)
    for raw in product((0, 1), repeat=2):
        # Replace only entry constants to exercise both raw representations. Entry
        # integers canonically encode zero; no production word-write API is added.
        instructions = (
            *(
                ClassicalXor((), i, bool(bit)).instruction()
                for i, bit in enumerate(raw)
            ),
            *bound.circuit.instructions[2:],
        )
        enabled = source.call_classically(ctrl=source_type.from_bits(raw), q=0)[-1]
        expected = state.reshape(2, 2).flip(0).flatten() if enabled else state
        torch.testing.assert_close(
            forced_branch(state, instructions, ()), expected, rtol=0, atol=0
        )


@pytest.mark.parametrize("activation", [-1, 0])
def test_word_conjoined_with_measured_child_control_all_histories(
    activation: int,
) -> None:
    dtype, source_type = types("signed")
    from qualtran import CBit, CtrlSpec
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled

    child = classical_bit_feedback_function(
        "leaf",
        (Register("q", quantum_bit_type()),),
        CircuitIR(1, (Instruction("x", (0,)),)),
        (((), (True,)),),
    )
    controlled = classical_control_component_program(
        ProgramRecord((child,)),
        (((), (activation,)),),
        entry="leaf",
        control_types=(dtype,),
    )
    root = controlled.functions[0]
    ports = (
        Register("word", dtype),
        Register("a", quantum_bit_type(), side=RegisterSide.LEFT),
        Register("q", quantum_bit_type()),
    )
    # Measured ownership remains governed by the existing CBit contract.
    from flagquantum.core.ir.program.register_types import classical_bit_type

    caller = ComponentProgramBuilder(
        Signature(
            (*ports, Register("history", classical_bit_type(), side=RegisterSide.RIGHT))
        ),
        (*controlled.functions, measure_x_word_function("measure")),
    )
    history = caller.call_dict("measure", {"q": caller.inputs["a"]})["c"]
    out = caller.call_dict(
        root.name,
        {"ctrl2": caller.inputs["word"], "ctrl": history, "q": caller.inputs["q"]},
    )
    module = caller.finalize(
        {"word": out["ctrl2"], "q": out["q"], "history": out["ctrl"]}
    )
    state = torch.complex(torch.arange(1, 9).double(), torch.arange(8).double())
    state /= state.norm()
    source = ClassicallyControlled(
        ClassicallyControlled(XGate(), CtrlSpec(CBit(), 1)),
        CtrlSpec(source_type, activation),
    )
    for value in range(-2, 2):
        bound = bind_register_program(
            module, ((), (0,), (1,)), classical_inputs=(value, None, None)
        )
        for outcome in (0, 1):
            enabled = source.call_classically(ctrl2=value, ctrl=outcome, q=0)[-1]
            reference = (
                Instruction("h", (0,)),
                Instruction(
                    "measure", (0,), metadata={"is_dynamic": True, "classical_bit": 2}
                ),
                *((Instruction("x", (1,)),) if enabled else ()),
            )
            torch.testing.assert_close(
                forced_branch(state, bound.circuit.instructions, (outcome,)),
                forced_branch(state, reference, (outcome,)),
                rtol=0,
                atol=3e-15,
            )


def test_shaped_mixed_words_use_row_major_msb_guards() -> None:
    from qualtran import CInt, CtrlSpec, CUInt
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled

    child = classical_bit_feedback_function(
        "main",
        (Register("q", quantum_bit_type()),),
        CircuitIR(1, (Instruction("x", (0,)),)),
        (),
    )
    control_types = (
        classical_unsigned_register_type(2),
        classical_signed_register_type(2),
    )
    pattern = (((2,), (0, 3)), ((), (-1,)))
    record = classical_control_component_program(
        ProgramRecord((child,)), pattern, control_types=control_types
    )
    source = ClassicallyControlled(
        XGate(), CtrlSpec([CUInt(2), CInt(2)], [np.array([0, 3]), -1])
    )
    state = torch.tensor([1 + 2j, 2 - 1j, -3j, 4 + 1j], dtype=torch.complex128)
    state /= state.norm()
    for a, b, c in product(range(4), range(4), range(-2, 2)):
        bound = bind_register_program(
            record, ((), (), (), (0,)), classical_inputs=(a, b, c, None)
        )
        assert bound.classical_outputs == ((0, 1), (2, 3), (4, 5), ())
        assert bound.circuit.instructions[-1].metadata["conditions"] == (
            (0, 0),
            (1, 0),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
        )
        enabled = source.call_classically(ctrl1=np.array([a, b]), ctrl2=c, q=0)[-1]
        expected = state.reshape(2, 2).flip(0).flatten() if enabled else state
        circuit = DynamicCircuit(2, inputs=state, dtype=torch.complex128)
        circuit.append(FieldOperation(2, bound.circuit.instructions))
        result = run_dynamic(circuit, shots=1, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states[0].flatten(), expected, rtol=0, atol=3e-15
        )


@pytest.mark.parametrize("shape", [(), (0,)])
def test_zero_width_and_empty_controls_have_source_vacuous_activation(
    shape: tuple[int, ...],
) -> None:
    from qualtran import CtrlSpec, CUInt
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled

    child = classical_bit_feedback_function(
        "main",
        (Register("q", quantum_bit_type()),),
        CircuitIR(1, (Instruction("x", (0,)),)),
        (),
    )
    values = (0,) if not shape else ()
    record = classical_control_component_program(
        ProgramRecord((child,)),
        ((shape, values),),
        control_types=(classical_unsigned_register_type(0),),
    )
    source_values = 0 if not shape else np.empty(0, dtype=int)
    source = ClassicallyControlled(XGate(), CtrlSpec(CUInt(0), source_values))
    assert source.ctrl_spec.is_active(source_values)
    count = len(values)
    bound = bind_register_program(
        record, ((),) * count + ((0,),), classical_inputs=(*values, None)
    )
    assert bound.circuit.instructions == (Instruction("x", (0,)),)
    assert bound.outputs == ((),) * count + ((0,),)
    assert bound.classical_outputs == (((), ()) if count else ())
    np.testing.assert_array_equal(
        source._tensor_data().reshape(2, 2), np.array([[0, 1], [1, 0]])
    )
