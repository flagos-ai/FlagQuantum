"""Complete CNOT source capture, scalar display and SDK assertions."""

from dataclasses import replace
from hashlib import sha1
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import CliffordCall
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from flagquantum.simulation.unitary import get_unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.basic_gates import cnot

    for name, blob in (
        ("cnot.py", "d2c2002a226debd2cdf382bbbec5c6f027bf90cc"),
        ("cnot_test.py", "cb4f3767b80d5f3d2220ebb0892964dec725ca92"),
    ):
        raw = (Path(cnot.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def test_actual_factory_atom_binding_adjoint_and_model() -> None:
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.basic_gates.cnot import _CNOT_DOC, _cnot
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = _cnot.make()
    assert source == CNOT() and _CNOT_DOC.examples == (_cnot,)
    assert source.adjoint() is source
    with pytest.raises(DecomposeTypeError, match="atomic"):
        source.decompose_bloq()
    _, leaves = source.call_graph()
    assert leaves == {source: 1}
    emitter = import_register_emitter(source)
    assert emitter.layout == (("ctrl", 1), ("target", 1))
    assert emitter.requirements() == (0, 0) and emitter.binding_swaps == (0, 0)
    emitter = replace(
        emitter,
        snapshots=(
            CircuitIR.from_json(emitter.snapshots[0].to_json()),
            CircuitIR.from_json(emitter.snapshots[1].to_json()),
        ),
    )
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    for ir in emitter.snapshots:
        assert ir.instructions == (Instruction("cx", (0, 1)),)
        torch.testing.assert_close(
            get_unitary(ir, dtype=torch.complex128), matrix, rtol=0, atol=2e-12
        )
    # Reordered physical wires with a coherent, entangled spectator.
    wires = (2, 0)
    order = (*wires, 1)
    inverse_order = tuple(order.index(q) for q in range(3))
    state = torch.tensor(
        [complex(i + 1, 3 - 2 * i) for i in range(8)], dtype=torch.complex128
    )
    state /= state.norm()
    expected = (
        (matrix @ state.reshape((2,) * 3).permute(order).reshape(4, 2))
        .reshape((2,) * 3)
        .permute(inverse_order)
        .flatten()
    )
    for inverse in (False, True):
        gates = emitter({"ctrl": wires[:1], "target": wires[1:]}, (), (), inverse)
        assert gates == (Instruction("cx", wires),)
        circuit = Circuit(3, inputs=state, dtype=torch.complex128)
        circuit.append(FieldOperation(3, gates))
        torch.testing.assert_close(
            circuit.state().flatten(), expected, rtol=0, atol=2e-12
        )
        circuit.append(FieldOperation(3, gates))
        torch.testing.assert_close(circuit.state().flatten(), state, rtol=0, atol=2e-12)
    assert (
        query_qec_gate_cost(CliffordCall("cx", 2)).asdict()
        == get_cost_value(source, QECGatesCost()).asdict()
    )
    assert get_cost_value(source, QubitCount()) == 2


def test_original_cirq_and_pennylane_assertions() -> None:
    import cirq
    import pennylane as qml
    from qualtran.bloqs.basic_gates import CNOT

    from flagquantum.ecosystem.cirq import from_cirq, to_cirq
    from flagquantum.ecosystem.pennylane import from_pennylane, to_pennylane

    source = CNOT()
    ir = CircuitIR(2, (Instruction("cx", (0, 1)),))
    expected = cirq.Circuit(cirq.CNOT(*cirq.LineQubit.range(2)))
    assert to_cirq(ir) == expected
    assert from_cirq(expected).instructions == ir.instructions
    script = to_pennylane(ir)
    assert script.operations[0] == source.as_pl_op(wires=[0, 1])
    assert from_pennylane(script).instructions == ir.instructions
    torch.testing.assert_close(
        torch.as_tensor(qml.matrix(script.operations[0]), dtype=torch.complex128),
        torch.as_tensor(source.tensor_contract()),
        rtol=0,
        atol=2e-12,
    )
    twice = Circuit(2, dtype=torch.complex128)
    operation = FieldOperation(2, ir.instructions)
    twice.append(operation).append(operation)
    assert torch.equal(twice.state().flatten(), torch.eye(4, dtype=torch.complex128)[0])
    cirq.testing.assert_has_diagram(
        to_cirq(CircuitIR(2, ir.instructions * 2)),
        "0: ───@───@───\n      │   │\n1: ───X───X───",
    )


def test_complete_scalar_score_and_source_unknown_register() -> None:
    from qualtran import BloqBuilder, QBit
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.drawing import get_musical_score_data

    from flagquantum.core.ir.program.register_types import opaque_register_type
    from flagquantum.core.ir.program.registers import Register
    from flagquantum.drawer.controlled_not_symbols import controlled_not_wire_symbol
    from flagquantum.drawer.register_score import RegisterStep, layout_register_score
    from flagquantum.drawer.register_score_data import register_score_data
    from tests.unit.test_drawer_component_score import assert_source_geometry
    from tests.unit.test_register_tensor import native_signature

    source = CNOT()
    registers = tuple(native_signature(source.signature))
    assert tuple(reg.shape for reg in registers) == ((), ())
    names = ("ctrl", "target")
    step = RegisterStep(
        str(source), names, names, input_registers=registers, output_registers=registers
    )
    score = layout_register_score(dict.fromkeys(names, 1), names, (step,))
    builder, inputs = BloqBuilder.from_signature(source.signature)
    original = builder.finalize(**builder.add_d(source, **inputs))
    assert_source_geometry(
        register_score_data(score, register_wire_symbol=controlled_not_wire_symbol),
        get_musical_score_data(original),
        ordered_ports=True,
    )
    assert controlled_not_wire_symbol(step, registers[0], ()) == (
        "Circle",
        {"filled": True},
    )
    unknown = Register("unknown", opaque_register_type(1))
    with pytest.raises(ValueError, match="Unknown wire symbol register name: unknown"):
        controlled_not_wire_symbol(step, unknown, ())
    with pytest.raises(ValueError, match="Unknown wire symbol register name: unknown"):
        source.wire_symbol(SourceRegister("unknown", QBit()))
