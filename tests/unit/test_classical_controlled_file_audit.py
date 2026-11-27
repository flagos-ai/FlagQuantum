"""Locked classical-control signatures, attachments and physical feedback."""

from hashlib import sha1
from importlib.metadata import version
from itertools import product
from pathlib import Path

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_feedback import (
    call_classical_feedback,
    classical_bit_feedback_function,
    classical_control_signature,
)
from flagquantum.compiler.registers.component_measurement import measure_x_word_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    classical_unsigned_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_locked_files_classical_port_admission_and_scalar_truth_table() -> None:
    import kickmix
    from qualtran._infra.controlled import CtrlSpec
    from qualtran._infra.data_types import CBit, CUInt, QBit
    from qualtran.bloqs.basic_gates.x_basis import XGate
    from qualtran.bloqs.mcmt import classically_controlled

    assert version("qualtran") == "0.7.0"
    assert getattr(kickmix, "__version__") == "0.1" and hasattr(kickmix, "Simulator")
    source_dir = Path(classically_controlled.__file__).parent
    for name, digest in (
        ("classically_controlled.py", "a4236872f616527a1dbc92b5453cb647b656650f"),
        ("classically_controlled_test.py", "e24f6b9cc8023feedc204c1128ec458dd8476d6e"),
    ):
        raw = (source_dir / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest

    wrapper = classically_controlled.ClassicallyControlled
    ports = Signature((Register("q", quantum_bit_type()),))
    with pytest.raises(ValueError, match="Invalid type"):
        wrapper(XGate(), CtrlSpec(QBit(), cvs=1))
    with pytest.raises(ValueError, match="classical data type"):
        classical_control_signature(ports, ((quantum_bit_type(), ()),))
    assert (
        tuple(classical_control_signature(ports, ((classical_bit_type(), ()),)))[0].name
        == "ctrl"
    )

    wide = wrapper(XGate(), CtrlSpec(CUInt(2), cvs=np.asarray([1, 2], dtype=int)))
    native_wide = classical_control_signature(
        ports, ((classical_unsigned_register_type(2), (2,)),)
    )
    assert [(r.name, r.bitsize, r.shape, r.side.value) for r in native_wide] == [
        (r.name, r.bitsize, r.shape, r.side.value) for r in wide.signature
    ]

    control = wrapper(XGate(), CtrlSpec(CBit(), cvs=1))
    feedback = classical_bit_feedback_function(
        "main",
        tuple(ports),
        CircuitIR(1, (Instruction("x", (0,)),)),
        (((), (True,)),),
    )
    program = ProgramRecord((feedback,))
    for enabled, value in product((0, 1), repeat=2):
        expected = control.call_classically(ctrl=enabled, q=value)
        assert expected == (enabled, value ^ enabled)
        bound = bind_register_program(
            program, ((), (0,)), classical_inputs=(enabled, None)
        )
        assert bound.classical_outputs == ((0,), ())
        initial = torch.eye(2, dtype=torch.complex128)[value]
        circuit = DynamicCircuit(1, inputs=initial, dtype=initial.dtype)
        circuit.append(FieldOperation(1, bound.circuit.instructions))
        states = run_dynamic(circuit, shots=1, strategy="trajectory")
        assert states.classical_bits[0, 0] == expected[0]
        torch.testing.assert_close(
            states.final_states[0].flatten(),
            torch.eye(2, dtype=initial.dtype)[expected[1]],
        )


def test_shaped_control_attachment_collision_and_every_readout_branch() -> None:
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran._infra.controlled import CtrlSpec
    from qualtran._infra.data_types import CBit, QBit
    from qualtran._infra.registers import Register as SourceRegister
    from qualtran._infra.registers import Signature as SourceSignature
    from qualtran.bloqs.basic_gates.x_basis import XGate
    from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled

    q = quantum_bit_type()
    c = classical_bit_type()
    ports = (Register("ctrl", q), Register("ctrl2", q))
    source_bb, source_in = BloqBuilder.from_signature(
        SourceSignature.build(ctrl=1, ctrl2=1)
    )
    source_in["ctrl"] = source_bb.add(XGate(), q=source_in["ctrl"])
    child = source_bb.finalize(**source_in)
    cv = (np.asarray([1, 0], dtype=int), np.asarray(1, dtype=int))
    source, attach = ClassicallyControlled.make_ctrl_system(
        child, CtrlSpec((CBit(), CBit()), cvs=cv)
    )
    orig_bb, orig_in = BloqBuilder.from_signature(
        SourceSignature(
            (
                SourceRegister("flags", CBit(), (2,)),
                SourceRegister("enable", CBit()),
                SourceRegister("u", QBit()),
                SourceRegister("v", QBit()),
            )
        )
    )
    cc, payload = attach(
        orig_bb,
        (orig_in["flags"], orig_in["enable"]),
        {"ctrl": orig_in["u"], "ctrl2": orig_in["v"]},
    )
    flags, enable = cc
    u, v = payload
    original = orig_bb.finalize(flags=flags, enable=enable, u=u, v=v)

    feedback = classical_bit_feedback_function(
        "feedback",
        ports,
        CircuitIR(2, (Instruction("x", (0,)),)),
        (((2,), (True, False)), ((), (True,))),
    )
    assert feedback.registers is not None
    assert (
        tuple(r.name for r in feedback.registers)
        == (*source.ctrl_reg_names, "ctrl", "ctrl2")
        == ("ctrl3", "ctrl4", "ctrl", "ctrl2")
    )
    builder = ComponentProgramBuilder(
        Signature(
            (
                *(
                    Register(name, q, side=RegisterSide.LEFT)
                    for name in ("a", "b", "d")
                ),
                Register("flags", c, (2,), side=RegisterSide.RIGHT),
                Register("enable", c, side=RegisterSide.RIGHT),
                Register("u", q),
                Register("v", q),
            )
        ),
        (measure_x_word_function("measure"), feedback),
    )
    a, b, d = [
        builder.call_dict("measure", {"q": builder.inputs[name]})["c"]
        for name in ("a", "b", "d")
    ]
    (returned_flags, returned_enable), (returned_u, returned_v) = (
        call_classical_feedback(
            builder,
            feedback,
            (np.asarray([a, b], dtype=object), d),
            {"ctrl": builder.inputs["u"], "ctrl2": builder.inputs["v"]},
        )
    )
    record = builder.finalize(
        {
            "flags": returned_flags,
            "enable": returned_enable,
            "u": returned_u,
            "v": returned_v,
        }
    )
    bound = bind_register_program(record, ((0,), (1,), (2,), (3,), (4,)))
    assert bound.classical_outputs == ((0,), (1,), (2,), (), ())
    assert bound.outputs == ((), (), (), (3,), (4,))
    assert bound.circuit.instructions[-1] == Instruction(
        "x", (3,), metadata={"conditions": ((0, 1), (1, 0), (2, 1))}
    )

    weights = torch.arange(1, 33, dtype=torch.float64)
    amplitudes = torch.complex(weights, weights.flip(0))
    amplitudes /= amplitudes.norm()
    initial = (
        DynamicCircuit(5, inputs=amplitudes, dtype=amplitudes.dtype)
        .h(0)
        .h(1)
        .h(2)
        .state()
        .flatten()
    )
    execution = DynamicCircuit(5, inputs=initial, dtype=initial.dtype)
    execution.append(FieldOperation(5, bound.circuit.instructions))
    body = execution.to_ir().instructions
    for settings in product((0, 1), repeat=3):
        expected = torch.zeros_like(amplitudes)
        prefix = 4 * settings[0] + 2 * settings[1] + settings[2]
        for input_u, input_v in product((0, 1), repeat=2):
            result = original.call_classically(
                flags=np.asarray(settings[:2]), enable=settings[2], u=input_u, v=input_v
            )
            np.testing.assert_array_equal(result[0], settings[:2])
            assert result[1] == settings[2]
            expected[4 * prefix + 2 * result[2] + result[3]] = amplitudes[
                4 * prefix + 2 * input_u + input_v
            ]
        torch.testing.assert_close(
            forced_branch(initial, body, settings), expected, atol=2e-12, rtol=0
        )
