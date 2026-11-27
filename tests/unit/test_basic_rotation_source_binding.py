"""Bind the complete basic rotation family at its original source boundaries."""

from functools import partial
from hashlib import sha1
from pathlib import Path
from typing import Literal

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import RotationCall
from flagquantum.compiler.registers.primitives.pauli_rotation import (
    PauliAxis,
    pauli_power_ir,
    pauli_rotation_ir,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from flagquantum.simulation.unitary import get_unitary
from tests._reference_dependencies import unavailable_reference

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.basic_gates import rotation

    for name, blob in (
        ("rotation.py", "69d2f19d636db2de0105919fb6b53421ba5b94b4"),
        ("rotation_test.py", "63480cc29919427e516c71f7453e8bdfa7cf01a3"),
    ):
        raw = (Path(rotation.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("axis", ("x", "y", "z"))
@pytest.mark.parametrize("power", (False, True))
@pytest.mark.parametrize("parameter", (0.4, 0.5, 1.0))
def test_actual_atomic_rotation_source_stream_and_native_replacement(
    axis: PauliAxis,
    power: bool,
    parameter: float,
) -> None:
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    from qualtran._infra.bloq import DecomposeTypeError
    from qualtran.bloqs.basic_gates import rotation
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    from flagquantum.ecosystem.cirq._feedback_import import (
        _has_explicit_matrix,
        _lower_feedback_gate,
        import_cirq_feedback,
    )
    from tests.unit.test_controlled_unitary_synthesis import assert_source_stream

    source_cls = getattr(rotation, axis.upper() + "PowGate" if power else "R" + axis)
    source = source_cls(parameter)
    emitter = import_register_emitter(source)
    assert emitter.layout == (("q", 1),) and emitter.requirements() == (0, 0)
    with pytest.raises(DecomposeTypeError, match="atomic"):
        source.decompose_bloq()
    opcodes: dict[PauliAxis, Literal["xpow", "ypow", "zpow"]] = {
        "x": "xpow",
        "y": "ypow",
        "z": "zpow",
    }
    radian_codes: dict[PauliAxis, Literal["rx", "ry", "rz"]] = {
        "x": "rx",
        "y": "ry",
        "z": "rz",
    }
    for inverse in (False, True):
        original = source.adjoint() if inverse else source
        qubit = cirq.LineQubit(0)
        leaves = cirq.decompose(
            original.cirq_gate.on(qubit),
            keep=lambda op: isinstance(op.gate, cirq.GlobalPhaseGate)
            or _lower_feedback_gate(cirq, op.gate, [], 0) is not None
            or _has_explicit_matrix(cirq, op.gate),
        )
        direct = import_cirq_feedback(
            cirq.Circuit(leaves, strategy=cirq.InsertStrategy.NEW), qubit_order=(qubit,)
        ).ir
        snapshot = CircuitIR.from_json(emitter.snapshots[int(inverse)].to_json())
        assert_source_stream(snapshot, direct)
        signed = -parameter if inverse else parameter
        native = (
            pauli_power_ir(axis, 0, signed)
            if power
            else pauli_rotation_ir(axis, 0, signed)
        )
        expected = torch.as_tensor(original.tensor_contract(), dtype=torch.complex128)
        torch.testing.assert_close(
            get_unitary(snapshot, dtype=torch.complex128), expected, rtol=0, atol=3e-12
        )
        circuit = Circuit(1, dtype=torch.complex128).append(FieldOperation(1, native))
        torch.testing.assert_close(get_unitary(circuit), expected, rtol=0, atol=3e-12)
        # Rebind to an entangled spectator and preserve absolute phase.
        state = torch.tensor([1 + 2j, 3 - 1j, -2j, 4 + 1j], dtype=torch.complex128)
        state /= state.norm()
        actual = Circuit(2, inputs=state, dtype=torch.complex128).append(
            FieldOperation(2, emitter({"q": (1,)}, (), (), inverse))
        )
        torch.testing.assert_close(
            actual.state().flatten(),
            (expected @ state.reshape(2, 2).T).T.flatten(),
            rtol=0,
            atol=3e-12,
        )
        actual.append(FieldOperation(2, emitter({"q": (1,)}, (), (), not inverse)))
        torch.testing.assert_close(actual.state().flatten(), state, rtol=0, atol=3e-12)
        key = RotationCall(opcodes[axis] if power else radian_codes[axis], signed)
        assert (
            query_qec_gate_cost(key).asdict()
            == get_cost_value(original, QECGatesCost()).asdict()
        )
        assert get_cost_value(original, QubitCount()) == 1


@pytest.mark.parametrize("axis", ("x", "y", "z"))
def test_original_radian_adapter_round_trips(axis: PauliAxis) -> None:
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    from qualtran.bloqs.basic_gates import rotation

    from flagquantum.ecosystem.cirq import from_cirq, to_cirq
    from flagquantum.ecosystem.pennylane import from_pennylane, to_pennylane

    qml = pytest.importorskip("pennylane")
    source = getattr(rotation, "R" + axis)(0.4)
    ir = CircuitIR(1, (Instruction("r" + axis, (0,), {"theta": 0.4}),))
    assert to_cirq(ir) == cirq.Circuit(source.cirq_gate.on(cirq.LineQubit(0)))
    from tests.unit.test_controlled_unitary_synthesis import assert_source_stream

    assert_source_stream(from_cirq(to_cirq(ir)), ir)
    script = to_pennylane(ir)
    assert script.operations[0] == source.as_pl_op(wires=[0])
    assert from_pennylane(script).instructions == ir.instructions
    torch.testing.assert_close(
        torch.as_tensor(qml.matrix(script.operations[0]), dtype=torch.complex128),
        torch.as_tensor(source.tensor_contract()),
        rtol=0,
        atol=3e-12,
    )


@pytest.mark.parametrize("axis", ("x", "y", "z"))
@pytest.mark.parametrize("power", (False, True))
@pytest.mark.parametrize("symbolic", (False, True))
def test_complete_atomic_source_display(
    axis: PauliAxis, power: bool, symbolic: bool
) -> None:
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran.bloqs.basic_gates import rotation
    from qualtran.drawing.musical_score import get_musical_score_data
    from sympy import Symbol

    from flagquantum.core.ir.program.register_types import opaque_register_type
    from flagquantum.core.ir.program.registers import Register
    from flagquantum.drawer.register_score import RegisterStep, layout_register_score
    from flagquantum.drawer.register_score_data import register_score_data
    from flagquantum.drawer.rotation_symbols import rotation_wire_symbol
    from tests.unit.test_drawer_component_score import assert_source_geometry

    parameter = Symbol("a") if symbolic else 1.0
    source = getattr(rotation, axis.upper() + "PowGate" if power else "R" + axis)(
        parameter
    )
    reg = Register("q", opaque_register_type(1))
    step = RegisterStep(
        str(source), ("q",), ("q",), input_registers=(reg,), output_registers=(reg,)
    )
    supplier = partial(
        rotation_wire_symbol, axis=axis, parameter=parameter, power=power
    )
    builder, inputs = BloqBuilder.from_signature(source.signature)
    original = builder.finalize(**builder.add_d(source, **inputs))
    score = layout_register_score({"q": 1}, ("q",), (step,))
    assert_source_geometry(
        register_score_data(score, register_wire_symbol=supplier),
        get_musical_score_data(original),
        ordered_ports=True,
    )
    label = supplier(step, reg, ())[1]["text"]
    assert str(source) == (str(label).replace("^", "**") if power else label)


@pytest.mark.parametrize("axis", ("x", "y"))
@pytest.mark.parametrize("shift", (-0.5, 0.2))
def test_shifted_power_atoms_keep_source_absolute_phase(
    axis: PauliAxis, shift: float
) -> None:
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    from qualtran.bloqs.basic_gates.rotation import XPowGate, YPowGate

    from flagquantum.ecosystem.cirq._feedback_import import import_cirq_feedback
    from tests.unit.test_controlled_unitary_synthesis import assert_source_stream

    source = (XPowGate if axis == "x" else YPowGate)(0.37, global_shift=shift)
    emitter = import_register_emitter(source)
    for inverse in (False, True):
        original = source.adjoint() if inverse else source
        expected = torch.as_tensor(original.tensor_contract())
        assert original.global_shift == shift
        qubit = cirq.LineQubit(0)
        direct = import_cirq_feedback(
            cirq.Circuit(original.cirq_gate.on(qubit)), qubit_order=(qubit,)
        ).ir
        assert_source_stream(emitter.snapshots[int(inverse)], direct)
        torch.testing.assert_close(
            get_unitary(emitter.snapshots[int(inverse)], dtype=torch.complex128),
            expected,
            rtol=0,
            atol=3e-12,
        )
        native = pauli_power_ir(axis, 0, -0.37 if inverse else 0.37, global_shift=shift)
        circuit = Circuit(1, dtype=torch.complex128).append(FieldOperation(1, native))
        torch.testing.assert_close(get_unitary(circuit), expected, rtol=0, atol=3e-12)


@pytest.mark.parametrize("controlled", (False, True))
def test_zpower_native_signed_repetition_and_source_power(controlled: bool) -> None:
    from qualtran.bloqs.basic_gates.rotation import CZPowGate, ZPowGate

    from flagquantum.compiler.registers.primitives.controlled_rotation import (
        controlled_z_power_ir,
    )
    from tests.unit.measurement_checks import forced_branch

    source = (CZPowGate if controlled else ZPowGate)(0.37, eps=1e-9)
    scaled = source**-3
    assert scaled.exponent == source.exponent * -3 and scaled.eps == source.eps
    exponent = float(scaled.exponent)
    if not controlled:
        circuit = Circuit(1, dtype=torch.complex128).append(
            FieldOperation(1, pauli_power_ir("z", 0, exponent))
        )
        torch.testing.assert_close(
            get_unitary(circuit),
            torch.as_tensor(scaled.tensor_contract()),
            rtol=0,
            atol=3e-12,
        )
        return
    emitter = import_register_emitter(scaled)
    assert emitter.layout == (("q", 2),) and emitter.requirements() == (1, 1)
    expected_gates = controlled_z_power_ir(0, 1, exponent, 2, 0)
    from tests.unit.test_register_subtraction import assert_source

    assert_source(emitter.snapshots[0].instructions, expected_gates)
    coherent = torch.tensor([1 + 2j, 3 - 1j, -2j, 4 + 1j], dtype=torch.complex128)
    coherent /= coherent.norm()
    initial = torch.zeros(8, dtype=torch.complex128)
    initial[::2] = coherent
    expected = torch.zeros_like(initial)
    expected[::2] = torch.as_tensor(scaled.tensor_contract()) @ coherent / 2**0.5
    for first in (0, 1):
        branch = forced_branch(initial, emitter.snapshots[0].instructions, (first,))
        torch.testing.assert_close(branch, expected, rtol=0, atol=3e-12)
        assert not torch.count_nonzero(branch[1::2])
        for second in (0, 1):
            torch.testing.assert_close(
                forced_branch(branch, emitter.snapshots[1].instructions, (second,)),
                initial / 2,
                rtol=0,
                atol=3e-12,
            )


def test_actual_crz_named_capture_and_all_original_factories() -> None:
    from qualtran.bloqs.basic_gates import rotation

    from flagquantum.compiler.registers.primitives.controlled_rotation import (
        controlled_rz_ir,
    )
    from tests.unit.test_register_subtraction import assert_source

    source = rotation.CRz(0.37, eps=1e-9)
    emitter = import_register_emitter(source)
    assert emitter.layout == (("ctrl", 1), ("q", 1))
    assert emitter.requirements() == (0, 0)
    for inverse in (False, True):
        original = source.adjoint() if inverse else source
        from flagquantum.compiler.registers.primitives.power import register_power_ir

        native = register_power_ir(
            CircuitIR(2, controlled_rz_ir(0, 1, 0.37)), 1, adjoint=inverse
        )
        assert_source(emitter.snapshots[int(inverse)].instructions, native)
        torch.testing.assert_close(
            get_unitary(emitter.snapshots[int(inverse)], dtype=torch.complex128),
            torch.as_tensor(original.tensor_contract()),
            rtol=0,
            atol=3e-12,
        )
    for name in ("_z_pow", "_cz_pow", "_x_pow", "_y_pow", "_rx", "_ry", "_rz", "_crz"):
        example = getattr(rotation, name).make()
        # Original symbolic examples bind through the approved parameter contract.
        if name in {"_rz", "_crz"}:
            from qualtran.resource_counting._bloq_counts import QECGatesCost
            from qualtran.resource_counting._costing import get_cost_value

            from flagquantum.compiler._component_trace.resource_generalization import (
                ControlledZRotationCall,
            )

            key = (
                ControlledZRotationCall(example.angle, example.eps)
                if name == "_crz"
                else RotationCall("rz", example.angle, example.eps)
            )
            assert (
                query_qec_gate_cost(key).asdict()
                == get_cost_value(example, QECGatesCost()).asdict()
            )
            symbolic_emitter = import_register_emitter(example)
            width = 2 if name == "_crz" else 1
            source_class = rotation.CRz if name == "_crz" else rotation.Rz
            for inverse in (False, True):
                circuit = (
                    Circuit(width, dtype=torch.complex128)
                    .append(
                        FieldOperation(
                            width, symbolic_emitter.snapshots[int(inverse)].instructions
                        )
                    )
                    .bind_parameters({str(example.angle): 0.4})
                )
                matrix = source_class(-0.4 if inverse else 0.4).tensor_contract()
                torch.testing.assert_close(
                    get_unitary(circuit),
                    torch.as_tensor(matrix),
                    rtol=0,
                    atol=3e-12,
                )
            continue
        bound = import_register_emitter(example)
        assert sum(width for _, width in bound.layout) in (1, 2)
