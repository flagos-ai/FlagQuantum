"""Nonbinary stored controls follow source callbacks and coefficient bit words."""

from collections import Counter
from dataclasses import replace
from typing import Any

import pytest
import torch

try:
    import numpy as np
except ModuleNotFoundError:
    pytest.importorskip("numpy")

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.classical_word_storage import classical_word_bits
from flagquantum.compiler.registers.component_classical_control import (
    classical_control_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_polynomial_encoding import (
    FiniteFieldPolynomialEncoding,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import assert_source_wire_streams
from tests.unit.test_component_classical_control import hierarchy, source_stream

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CASES = ((3, 1, None), (5, 1, None), (3, 2, None), (3, 1, 1), (5, 1, 1), (3, 2, 0))


def fixture(
    characteristic: int, degree: int, polynomial_degree: int | None
) -> tuple[Any, Any]:
    from qualtran import CGF, QGF, CGFPoly

    field = FiniteFieldEncoding(characteristic, degree)
    if polynomial_degree is None:
        return field, CGF(characteristic, degree, field.irreducible_poly)
    return FiniteFieldPolynomialEncoding(polynomial_degree, field), CGFPoly(
        polynomial_degree, QGF(characteristic, degree, field.irreducible_poly)
    )


def packed(codec: Any, value: Any) -> int:
    return int("".join(str(bit) for bit in codec.to_bits(value)), 2)


@pytest.mark.parametrize("characteristic,degree,polynomial_degree", CASES)
def test_all_legal_controls_source_callback_child_stream_complex_work_and_append(
    characteristic: int, degree: int, polynomial_degree: int | None
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    codec, source_type = fixture(characteristic, degree, polynomial_degree)
    dtype = codec.register_type(classical=True)
    size = int(codec.bitsize)
    domain = list(codec.get_domain())
    module, source_child = hierarchy()
    original = source_stream(source_child)
    state = torch.complex(torch.arange(1, 9).double(), torch.arange(8).double())
    state /= state.norm()
    initial = torch.zeros(16, dtype=torch.complex128)
    initial.reshape(4, 2, 2)[:, 0, :] = state.reshape(4, 2)
    child_result = forced_branch(initial, original, ())
    for activation in domain:
        controlled = classical_control_component_program(
            module, (((), (packed(codec, activation),)),), control_types=(dtype,)
        )
        source = ClassicallyControlled(
            source_child, CtrlSpec(source_type, np.array(activation, dtype=object))
        )
        with pytest.raises(NotImplementedError):
            source.decompose_bloq()
        with pytest.raises((NotImplementedError, TypeError)):
            get_cost_value(source, QECGatesCost())
        numerical = ClassicallyControlled(
            XGate(), CtrlSpec(source_type, np.array(activation, dtype=object))
        )
        tensor = source._tensor_data().reshape(1 << size, 4, 1 << size, 4)
        for value in domain:
            word = packed(codec, value)
            bits = classical_word_bits(dtype, word)
            assert (
                tuple(codec.to_bits(value)) == tuple(source_type.to_bits(value)) == bits
            )
            bound = bind_register_program(
                controlled,
                ((), (0,), (1,)),
                classical_inputs=(word, None, None),
                clean=(2,),
            )
            assert bound.classical_outputs == (tuple(range(size)), (), ())
            assert Counter(g.name for g in bound.circuit.instructions) == {
                "classical_xor": size,
                "cx": 4,
                "phase": 2,
                "h": 2,
            }
            stripped = tuple(
                replace(g, metadata={}) for g in bound.circuit.instructions[size:]
            )
            assert_source_wire_streams(stripped, original)
            enabled = numerical.call_classically(ctrl=value, q=0)[-1]
            assert enabled == int(value == activation)
            result = forced_branch(initial, bound.circuit.instructions, ())
            expected = child_result if value == activation else initial
            torch.testing.assert_close(result, expected, atol=4e-14, rtol=0)
            assert result.reshape(4, 2, 2)[:, 1, :].abs().max().item() < 4e-14
            if polynomial_degree in (None, 0):
                action = torch.as_tensor(
                    tensor[word, :, word, :].copy(), dtype=torch.complex128
                )
                torch.testing.assert_close(
                    result.reshape(4, 2, 2)[:, 0, :],
                    action @ state.reshape(4, 2),
                    atol=4e-14,
                    rtol=0,
                )
            if activation in (domain[0], domain[-1]) and value in (
                domain[0],
                domain[-1],
            ):
                circuit = DynamicCircuit(4, inputs=initial, dtype=torch.complex128)
                circuit.append(FieldOperation(4, bound.circuit.instructions))
                actual = run_dynamic(circuit, shots=1, strategy="trajectory")
                torch.testing.assert_close(
                    actual.final_states[0].flatten(), result, atol=4e-14, rtol=0
                )
                assert actual.classical_bits[0].tolist() == list(bits)


@pytest.mark.parametrize("characteristic,integer_index,word", [(3, 8, 10), (5, 24, 36)])
def test_original_polynomial_tensor_index_discrepancy_is_preserved(
    characteristic: int, integer_index: int, word: int
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled

    codec, source_type = fixture(characteristic, 1, 1)
    activation = list(codec.get_domain())[-1]
    assert int(activation) == integer_index
    assert packed(codec, activation) == word
    source = ClassicallyControlled(
        XGate(), CtrlSpec(source_type, np.array(activation, dtype=object))
    )
    size = 1 << int(codec.bitsize)
    tensor = source._tensor_data().reshape(size, 2, size, 2)
    np.testing.assert_array_equal(
        tensor[integer_index, :, integer_index, :], [[0, 1], [1, 0]]
    )
    np.testing.assert_array_equal(tensor[word, :, word, :], np.eye(2))
    assert source.call_classically(ctrl=activation, q=0)[-1] == 1


@pytest.mark.parametrize("characteristic,degree,polynomial_degree", CASES)
def test_every_unused_coefficient_word_and_activation_is_refused(
    characteristic: int, degree: int, polynomial_degree: int | None
) -> None:
    codec, _ = fixture(characteristic, degree, polynomial_degree)
    dtype = codec.register_type(classical=True)
    valid = {packed(codec, value) for value in codec.get_domain()}
    module, _ = hierarchy()
    for word in range(1 << int(codec.bitsize)):
        if word in valid:
            continue
        with pytest.raises(ValueError, match="coefficients"):
            classical_word_bits(dtype, word)
        with pytest.raises(ValueError, match="coefficients"):
            classical_control_component_program(
                module, (((), (word,)),), control_types=(dtype,)
            )
        controlled = classical_control_component_program(
            module, (((), (0,)),), control_types=(dtype,)
        )
        with pytest.raises(ValueError, match="coefficients"):
            bind_register_program(
                controlled,
                ((), (0,), (1,)),
                classical_inputs=(word, None, None),
                clean=(2,),
            )
    for word in (-1, True, 1 << int(codec.bitsize)):
        with pytest.raises(ValueError):
            classical_word_bits(dtype, word)


@pytest.mark.parametrize("characteristic,degree,polynomial_degree", CASES)
@pytest.mark.parametrize("shape", [(), (2, 2), (0, 2)])
def test_nested_transport_reuses_slots(
    monkeypatch: pytest.MonkeyPatch,
    characteristic: int,
    degree: int,
    polynomial_degree: int | None,
    shape: tuple[int, ...],
) -> None:
    from tests.unit import test_classical_field_feedback as binary

    codec, source_type = fixture(characteristic, degree, polynomial_degree)
    monkeypatch.setattr(binary, "fixture", lambda *_: (codec, source_type))
    binary.test_repeated_calls_preserve_nominal_types_empty_arrays_and_slots(
        degree, polynomial_degree, shape
    )


@pytest.mark.parametrize("characteristic,degree,polynomial_degree", CASES)
def test_nonbinary_word_and_measured_child_all_histories(
    characteristic: int, degree: int, polynomial_degree: int | None
) -> None:
    from qualtran import CBit, CtrlSpec
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled

    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.compiler.registers.component_feedback import (
        classical_bit_feedback_function,
    )
    from flagquantum.compiler.registers.component_measurement import (
        measure_x_word_function,
    )
    from flagquantum.core.ir import CircuitIR, Instruction
    from flagquantum.core.ir.program import ProgramRecord
    from flagquantum.core.ir.program.register_types import (
        classical_bit_type,
        quantum_bit_type,
    )
    from flagquantum.core.ir.program.registers import Register, RegisterSide
    from flagquantum.core.ir.program.signatures import Signature

    codec, source_type = fixture(characteristic, degree, polynomial_degree)
    dtype = codec.register_type(classical=True)
    size = int(codec.bitsize)
    domain = list(codec.get_domain())
    child = classical_bit_feedback_function(
        "leaf",
        (Register("q", quantum_bit_type()),),
        CircuitIR(1, (Instruction("x", (0,)),)),
        (((), (True,)),),
    )
    state = torch.complex(torch.arange(1, 9).double(), torch.arange(8).double())
    state /= state.norm()
    for activation in (domain[0], domain[-1]):
        controlled = classical_control_component_program(
            ProgramRecord((child,)),
            (((), (packed(codec, activation),)),),
            entry="leaf",
            control_types=(dtype,),
        )
        signature = Signature(
            (
                Register("word", dtype),
                Register("a", quantum_bit_type(), side=RegisterSide.LEFT),
                Register("q", quantum_bit_type()),
                Register("history", classical_bit_type(), side=RegisterSide.RIGHT),
            )
        )
        builder = ComponentProgramBuilder(
            signature, (*controlled.functions, measure_x_word_function("measure"))
        )
        history = builder.call_dict("measure", {"q": builder.inputs["a"]})["c"]
        result = builder.call_dict(
            "leaf",
            {
                "ctrl2": builder.inputs["word"],
                "ctrl": history,
                "q": builder.inputs["q"],
            },
        )
        module = builder.finalize(
            {"word": result["ctrl2"], "q": result["q"], "history": result["ctrl"]}
        )
        source = ClassicallyControlled(
            ClassicallyControlled(XGate(), CtrlSpec(CBit(), 1)),
            CtrlSpec(source_type, np.array(activation, dtype=object)),
        )
        for value in domain:
            bound = bind_register_program(
                module,
                ((), (0,), (1,)),
                classical_inputs=(packed(codec, value), None, None),
            )
            assert bound.classical_outputs == (tuple(range(size)), (), (size,))
            assert Counter(g.name for g in bound.circuit.instructions) == {
                "classical_xor": size,
                "h": 1,
                "measure": 1,
                "x": 1,
            }
            for outcome in (0, 1):
                enabled = source.call_classically(ctrl2=value, ctrl=outcome, q=0)[-1]
                reference = (
                    Instruction("h", (0,)),
                    Instruction(
                        "measure",
                        (0,),
                        metadata={"is_dynamic": True, "classical_bit": size},
                    ),
                    *((Instruction("x", (1,)),) if enabled else ()),
                )
                torch.testing.assert_close(
                    forced_branch(state, bound.circuit.instructions, (outcome,)),
                    forced_branch(state, reference, (outcome,)),
                    atol=3e-15,
                    rtol=0,
                )


def test_shaped_nonbinary_groups_keep_row_major_coefficients_and_feedback() -> None:
    from itertools import product

    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled

    from flagquantum.compiler.registers.component_gate_function import bit_gate_function
    from flagquantum.core.ir import CircuitIR, Instruction
    from flagquantum.core.ir.program import ProgramRecord
    from flagquantum.core.ir.program.register_types import quantum_bit_type
    from flagquantum.core.ir.program.registers import Register

    field, source_field = fixture(3, 1, None)
    polynomial, source_polynomial = fixture(3, 1, 1)
    field_values = list(field.get_domain())
    polynomial_values = list(polynomial.get_domain())
    child = bit_gate_function(
        "main",
        (Register("q", quantum_bit_type()),),
        CircuitIR(1, (Instruction("x", (0,)),)),
    )
    module = classical_control_component_program(
        ProgramRecord((child,)),
        (((2,), (0, 2)), ((), (10,))),
        control_types=(
            field.register_type(classical=True),
            polynomial.register_type(classical=True),
        ),
    )
    source = ClassicallyControlled(
        XGate(),
        CtrlSpec(
            (source_field, source_polynomial),
            (
                np.array([field_values[0], field_values[-1]], dtype=object),
                np.array(polynomial_values[-1], dtype=object),
            ),
        ),
    )
    state = torch.tensor([1 + 2j, 2 - 1j, -3j, 4 + 1j], dtype=torch.complex128)
    for a, b, c in product(field_values, field_values, polynomial_values):
        bound = bind_register_program(
            module,
            ((), (), (), (0,)),
            classical_inputs=(
                packed(field, a),
                packed(field, b),
                packed(polynomial, c),
                None,
            ),
        )
        assert bound.classical_outputs == ((0, 1), (2, 3), (4, 5, 6, 7), ())
        enabled = source.call_classically(
            ctrl1=np.array([a, b], dtype=object), ctrl2=c, q=0
        )[-1]
        expected = state.reshape(2, 2).flip(0).flatten() if enabled else state
        torch.testing.assert_close(
            forced_branch(state, bound.circuit.instructions, ()),
            expected,
            atol=0,
            rtol=0,
        )
