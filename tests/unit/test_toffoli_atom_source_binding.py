"""Complete Toffoli source capture, shaped display and SDK assertions."""

from dataclasses import replace
from hashlib import sha1
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import FixedTCall
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from flagquantum.simulation.unitary import get_unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.basic_gates import toffoli

    for name, blob in (
        ("toffoli.py", "a007be2ef25aa7dc5fda4d793498041a8f6b5a48"),
        ("toffoli_test.py", "ff83e793112a8c7503ec526715e76647ce6f2196"),
    ):
        raw = (Path(toffoli.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def test_actual_factory_atom_binding_adjoint_and_model() -> None:
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.basic_gates import Toffoli
    from qualtran.bloqs.basic_gates.toffoli import _TOFFOLI_DOC, _toffoli
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = _toffoli.make()
    assert source == Toffoli() and _TOFFOLI_DOC.examples == (_toffoli,)
    assert source.adjoint() is source
    with pytest.raises(DecomposeTypeError, match="atomic"):
        source.decompose_bloq()
    _, leaves = source.call_graph()
    assert leaves == {source: 1}
    emitter = import_register_emitter(source)
    assert emitter.layout == (("ctrl", 2), ("target", 1))
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
        assert ir.instructions == (Instruction("ccx", (0, 1, 2)),)
        torch.testing.assert_close(
            get_unitary(ir, dtype=torch.complex128), matrix, rtol=0, atol=2e-12
        )
    # Reordered physical wires with a coherent, entangled spectator.
    wires = (2, 0, 3)
    order = (*wires, 1)
    inverse_order = tuple(order.index(q) for q in range(4))
    state = torch.tensor(
        [complex(i + 1, 3 - 2 * i) for i in range(16)], dtype=torch.complex128
    )
    state /= state.norm()
    expected = (
        (matrix @ state.reshape((2,) * 4).permute(order).reshape(8, 2))
        .reshape((2,) * 4)
        .permute(inverse_order)
        .flatten()
    )
    for inverse in (False, True):
        gates = emitter({"ctrl": wires[:2], "target": wires[2:]}, (), (), inverse)
        assert gates == (Instruction("ccx", wires),)
        circuit = Circuit(4, inputs=state, dtype=torch.complex128)
        circuit.append(FieldOperation(4, gates))
        torch.testing.assert_close(
            circuit.state().flatten(), expected, rtol=0, atol=2e-12
        )
        circuit.append(FieldOperation(4, gates))
        torch.testing.assert_close(circuit.state().flatten(), state, rtol=0, atol=2e-12)
    assert (
        query_qec_gate_cost(FixedTCall("toffoli")).asdict()
        == get_cost_value(source, QECGatesCost()).asdict()
    )
    assert get_cost_value(source, QubitCount()) == 3


def test_original_cirq_and_pennylane_assertions() -> None:
    import cirq
    import pennylane as qml
    from qualtran.bloqs.basic_gates import Toffoli

    from flagquantum.ecosystem.cirq import from_cirq, to_cirq
    from flagquantum.ecosystem.pennylane import from_pennylane, to_pennylane

    source = Toffoli()
    ir = CircuitIR(3, (Instruction("ccx", (0, 1, 2)),))
    expected = cirq.Circuit(cirq.CCNOT(*cirq.LineQubit.range(3)))
    assert to_cirq(ir) == expected
    assert from_cirq(expected).instructions == ir.instructions
    script = to_pennylane(ir)
    assert script.operations[0] == source.as_pl_op(wires=[0, 1, 2])
    assert from_pennylane(script).instructions == ir.instructions
    torch.testing.assert_close(
        torch.as_tensor(qml.matrix(script.operations[0]), dtype=torch.complex128),
        torch.as_tensor(source.tensor_contract()),
        rtol=0,
        atol=2e-12,
    )
    twice = Circuit(3, dtype=torch.complex128)
    operation = FieldOperation(3, ir.instructions)
    twice.append(operation).append(operation)
    assert torch.equal(twice.state().flatten(), torch.eye(8, dtype=torch.complex128)[0])
    cirq.testing.assert_has_diagram(
        to_cirq(CircuitIR(3, ir.instructions * 2)),
        "0: ───@───@───\n      │   │\n1: ───@───@───\n      │   │\n2: ───X───X───",
    )


def test_complete_shaped_score_and_source_unknown_register() -> None:
    from qualtran import BloqBuilder, QBit
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.basic_gates import Toffoli
    from qualtran.drawing import get_musical_score_data

    from flagquantum.core.ir.program.register_types import opaque_register_type
    from flagquantum.core.ir.program.registers import Register
    from flagquantum.drawer.controlled_not_symbols import controlled_not_wire_symbol
    from flagquantum.drawer.register_score import RegisterStep, layout_register_score
    from flagquantum.drawer.register_score_data import register_score_data
    from tests.unit.test_drawer_component_score import assert_source_geometry
    from tests.unit.test_register_tensor import native_signature

    source = Toffoli()
    registers = tuple(native_signature(source.signature))
    assert tuple(reg.shape for reg in registers) == ((2,), ())
    names = ("ctrl[0]", "ctrl[1]", "target")
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
    for index in ((0,), (1,)):
        assert controlled_not_wire_symbol(step, registers[0], index) == (
            "Circle",
            {"filled": True},
        )
    unknown = Register("unknown", opaque_register_type(1))
    with pytest.raises(ValueError, match="Unknown wire symbol register name: unknown"):
        controlled_not_wire_symbol(step, unknown, ())
    with pytest.raises(ValueError, match="Unknown wire symbol register name: unknown"):
        source.wire_symbol(SourceRegister("unknown", QBit()))
