"""Binary field controls reuse stored word reads, source actions and append."""

from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Any

import pytest
import torch

try:
    import numpy as np
except ModuleNotFoundError:
    pytest.importorskip("numpy")

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.classical_word_storage import classical_word_bits
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_classical_control import (
    classical_control_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
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
CASES = ((1, None), (2, None), (3, None), (1, 0), (1, 1), (2, 1))


def fixture(width: int, degree: int | None) -> tuple[Any, Any]:
    from qualtran import CGF, QGF, CGFPoly

    field = FiniteFieldEncoding(2, width)
    if degree is None:
        return field, CGF(2, width, field.irreducible_poly)
    return FiniteFieldPolynomialEncoding(degree, field), CGFPoly(
        degree, QGF(2, width, field.irreducible_poly)
    )


def decoded(codec: Any, word: int) -> Any:
    return codec.from_bits(
        [(word >> i) & 1 for i in reversed(range(int(codec.bitsize)))]
    )


@pytest.mark.parametrize("width,degree", CASES)
def test_all_words_source_tensor_child_stream_complex_work_and_append(
    width: int, degree: int | None
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    codec, source_type = fixture(width, degree)
    dtype = codec.register_type(classical=True)
    size = int(codec.bitsize)
    domain = range(1 << size)
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
        source = ClassicallyControlled(
            source_child,
            CtrlSpec(source_type, np.array(decoded(codec, activation), dtype=object)),
        )
        with pytest.raises(NotImplementedError):
            source.decompose_bloq()
        # Native field scalars are unhashable in the original root cache.
        # Polynomial activations reach the source's absent decomposition.
        with pytest.raises((NotImplementedError, TypeError)):
            get_cost_value(source, QECGatesCost())
        tensor = source._tensor_data().reshape(1 << size, 4, 1 << size, 4)
        for value in domain:
            bits = classical_word_bits(dtype, value)
            assert tuple(codec.to_bits(decoded(codec, value))) == bits
            assert tuple(source_type.to_bits(decoded(codec, value))) == bits
            bound = bind_register_program(
                controlled,
                ((), (0,), (1,)),
                classical_inputs=(value, None, None),
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
            action = torch.as_tensor(
                tensor[value, :, value, :].copy(), dtype=torch.complex128
            )
            expected = (action @ state.reshape(4, 2)).flatten()
            result = forced_branch(initial, bound.circuit.instructions, ())
            torch.testing.assert_close(
                result.reshape(4, 2, 2)[:, 0, :].flatten(), expected, atol=4e-14, rtol=0
            )
            assert result.reshape(4, 2, 2)[:, 1, :].abs().max().item() < 4e-14
            if activation in (0, len(domain) - 1) and value in (0, len(domain) - 1):
                circuit = DynamicCircuit(4, inputs=initial, dtype=torch.complex128)
                circuit.append(FieldOperation(4, bound.circuit.instructions))
                actual = run_dynamic(circuit, shots=1, strategy="trajectory")
                torch.testing.assert_close(
                    actual.final_states[0].flatten(), result, atol=4e-14, rtol=0
                )
                assert actual.classical_bits[0].tolist() == list(bits)


@pytest.mark.parametrize("width,degree", CASES)
@pytest.mark.parametrize("shape", [(), (2, 2), (0, 2)])
def test_repeated_calls_preserve_nominal_types_empty_arrays_and_slots(
    width: int, degree: int | None, shape: tuple[int, ...]
) -> None:
    from tests.unit.test_program_classical_registers import identity

    codec, _ = fixture(width, degree)
    dtype = codec.register_type(classical=True)
    ports = (Register("field", dtype, shape),)
    leaf = identity("leaf", ports)
    builder = ComponentProgramBuilder(Signature(ports), (leaf,))
    carried = builder.inputs
    for _ in range(3):
        carried = builder.call_dict("leaf", carried)
    record = builder.finalize(carried)
    count = len(tuple(ports[0].all_idxs()))
    bound = bind_register_program(
        record, ((),) * count, classical_inputs=(1,) * count, clean=(0,)
    )
    assert bound.outputs == ((),) * count
    size = int(codec.bitsize)
    assert bound.classical_outputs == tuple(
        tuple(range(i * size, (i + 1) * size)) for i in range(count)
    )
    assert len(bound.circuit.instructions) == size * count


@pytest.mark.parametrize("width,degree", CASES)
def test_invalid_stored_words_fail_before_binding(
    width: int, degree: int | None
) -> None:
    codec, _ = fixture(width, degree)
    dtype = codec.register_type(classical=True)
    for value in (-1, 1 << int(codec.bitsize), True):
        with pytest.raises(ValueError):
            classical_word_bits(dtype, value)


@pytest.mark.parametrize("width,degree", CASES)
def test_field_word_and_measured_child_control_all_histories(
    width: int, degree: int | None
) -> None:
    from qualtran import CBit, CtrlSpec
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled

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
    from flagquantum.core.ir.program.registers import RegisterSide

    codec, source_type = fixture(width, degree)
    dtype = codec.register_type(classical=True)
    size = int(codec.bitsize)
    child = classical_bit_feedback_function(
        "leaf",
        (Register("q", quantum_bit_type()),),
        CircuitIR(1, (Instruction("x", (0,)),)),
        (((), (True,)),),
    )
    state = torch.complex(torch.arange(1, 9).double(), torch.arange(8).double())
    state /= state.norm()
    for activation in (0, (1 << size) - 1):
        controlled = classical_control_component_program(
            ProgramRecord((child,)),
            (((), (activation,)),),
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
            CtrlSpec(source_type, np.array(decoded(codec, activation), dtype=object)),
        )
        for value, outcome in product(range(1 << size), (0, 1)):
            bound = bind_register_program(
                module, ((), (0,), (1,)), classical_inputs=(value, None, None)
            )
            assert bound.classical_outputs == (tuple(range(size)), (), (size,))
            enabled = source.call_classically(
                ctrl2=decoded(codec, value), ctrl=outcome, q=0
            )[-1]
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


def test_shaped_field_and_polynomial_groups_use_original_row_major_activation() -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled

    from flagquantum.compiler.registers.component_gate_function import bit_gate_function
    from flagquantum.core.ir import CircuitIR, Instruction
    from flagquantum.core.ir.program import ProgramRecord
    from flagquantum.core.ir.program.register_types import quantum_bit_type

    field, source_field = fixture(2, None)
    polynomial, source_polynomial = fixture(1, 1)
    child = bit_gate_function(
        "main",
        (Register("q", quantum_bit_type()),),
        CircuitIR(1, (Instruction("x", (0,)),)),
    )
    module = classical_control_component_program(
        ProgramRecord((child,)),
        (((2,), (0, 3)), ((), (2,))),
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
                np.array([decoded(field, 0), decoded(field, 3)], dtype=object),
                np.array(decoded(polynomial, 2), dtype=object),
            ),
        ),
    )
    state = torch.tensor([1 + 2j, 2 - 1j, -3j, 4 + 1j], dtype=torch.complex128)
    for a, b, c in product(range(4), repeat=3):
        bound = bind_register_program(
            module, ((), (), (), (0,)), classical_inputs=(a, b, c, None)
        )
        assert bound.classical_outputs == ((0, 1), (2, 3), (4, 5), ())
        enabled = source.call_classically(
            ctrl1=np.array([decoded(field, a), decoded(field, b)], dtype=object),
            ctrl2=decoded(polynomial, c),
            q=0,
        )[-1]
        expected = state.reshape(2, 2).flip(0).flatten() if enabled else state
        torch.testing.assert_close(
            forced_branch(state, bound.circuit.instructions, ()),
            expected,
            atol=0,
            rtol=0,
        )
