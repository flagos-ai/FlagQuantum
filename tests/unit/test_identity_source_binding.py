"""Bind complete Identity calls and distinguish atom, primitive and model counts."""

from hashlib import sha1
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.primitives.identity import register_identity_ir
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem.cirq import from_cirq, to_cirq
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from flagquantum.simulation.unitary import get_unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.basic_gates import identity

    for name, blob in (
        ("identity.py", "a78c6c7d91295613958772a8fcab1845446da417"),
        ("identity_test.py", "23e70493c487bccb4e539befdd7ee6d0ff4a19e3"),
    ):
        raw = (Path(identity.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("width", (1, 2, 3, 4))
@pytest.mark.parametrize("controlled", (False, True))
def test_actual_identity_source_capture_and_native_replacement(
    width: int, controlled: bool
) -> None:
    import numpy as np
    from qualtran import CtrlSpec, QBit, QUInt
    from qualtran.bloqs.basic_gates import Identity
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = Identity(width)
    if controlled:
        source = source.controlled(
            CtrlSpec((QBit(), QUInt(2)), (np.asarray([0, 1]), np.asarray(2)))
        )
    logical = width + 4 * int(controlled)
    emitter = import_register_emitter(source)
    assert emitter.layout == (("q", logical),) and emitter.requirements() == (0, 0)
    assert emitter.binding_swaps == (0, 0)
    assert emitter.snapshots[0].instructions == (
        (Instruction("i", (0,)),) if logical == 1 else ()
    )
    graph, leaves = source.call_graph()
    assert list(graph.nodes) == [source] and leaves == {source: 1}
    assert get_cost_value(source, QECGatesCost()).asdict() == {}
    assert get_cost_value(source, QubitCount()) == logical
    # The foreign multiwire atom has no primitive body after Cirq decomposition.
    # Native register identity explicitly emits one I per physical wire.
    wires = tuple(reversed(range(logical)))
    native = register_identity_ir(wires)
    assert len(native) == logical
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    for snapshot in emitter.snapshots:
        restored = CircuitIR.from_json(snapshot.to_json())
        actual = emitter({"q": wires}, (), (), snapshot is emitter.snapshots[1])
        assert restored.instructions == snapshot.instructions
        circuit = Circuit(logical, dtype=torch.complex128)
        circuit.append(FieldOperation(logical, actual))
        torch.testing.assert_close(get_unitary(circuit), expected, rtol=0, atol=0)
        circuit.append(FieldOperation(logical, native))
        torch.testing.assert_close(get_unitary(circuit), expected, rtol=0, atol=0)


def test_identity_single_wire_cirq_and_pennylane_round_trip() -> None:
    import cirq
    from qualtran.bloqs.basic_gates import Identity

    from flagquantum.ecosystem.pennylane import from_pennylane, to_pennylane

    qml = pytest.importorskip("pennylane")
    circuit = Circuit(1, dtype=torch.complex128).append(
        FieldOperation(1, (Instruction("i", (0,)),))
    )
    ir = circuit.to_ir()
    assert to_cirq(ir) == cirq.Circuit(cirq.I(cirq.LineQubit(0)))
    assert from_cirq(to_cirq(ir)).instructions == ir.instructions
    script = to_pennylane(ir)
    assert script.operations[0] == Identity().as_pl_op(wires=[0])
    assert from_pennylane(script).instructions == ir.instructions
    torch.testing.assert_close(
        get_unitary(circuit),
        torch.as_tensor(qml.matrix(script.operations[0]), dtype=torch.complex128),
        rtol=0,
        atol=0,
    )


def test_zero_width_source_export_and_unbound_capture_failures() -> None:
    from qualtran.bloqs.basic_gates import Identity
    from sympy import Symbol

    with pytest.raises(TypeError, match="missing 1 required positional argument: 'q'"):
        import_register_emitter(Identity(0))
    with pytest.raises(ValueError, match="concrete positive shapes"):
        import_register_emitter(Identity(Symbol("n")))


def test_classical_control_dispatch_is_not_quantum_register_capture() -> None:
    import numpy as np
    from qualtran import CBit, CtrlSpec
    from qualtran.bloqs.basic_gates import Identity
    from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled

    source = Identity(2).controlled(CtrlSpec(qdtypes=CBit()))
    assert isinstance(source, ClassicallyControlled)
    np.testing.assert_array_equal(source.tensor_contract(), np.eye(8))
    with pytest.raises(ValueError, match="Cannot convert classical control spec"):
        import_register_emitter(source)


@pytest.mark.parametrize("polarity", (0, 1))
def test_identity_classical_control_uses_shared_tensor_and_feedback(
    polarity: int,
) -> None:
    import numpy as np
    from qualtran import CBit, CtrlSpec
    from qualtran.bloqs.basic_gates import Identity

    from flagquantum.compiler.registers.control.control_predicate import ControlRegister
    from flagquantum.compiler.registers.primitives.classical_controlled import (
        classically_controlled_ir,
    )
    from flagquantum.core.ir.program.signatures import Signature
    from flagquantum.simulation.tensor_network.controlled_tensor import (
        controlled_register_tensor_node,
    )
    from flagquantum.simulation.tensor_network.register_tensor_binding import (
        contract_register_tensors,
    )
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_register_tensor import native_signature

    leaf = Identity(2)
    source = leaf.controlled(CtrlSpec(qdtypes=CBit(), cvs=polarity))
    controls, targets = native_signature(source.ctrl_regs), native_signature(
        leaf.signature
    )
    incoming, outgoing = {"ctrl": ((0,),), "q": ((1, 2),)}, {
        "ctrl": ((3,),),
        "q": ((4, 5),),
    }
    node = controlled_register_tensor_node(
        controls,
        targets,
        [np.array(polarity)],
        lambda: np.asarray(leaf.tensor_contract(), dtype=np.complex128),
        incoming,
        outgoing,
        torch.empty((), dtype=torch.complex128),
    )
    tensor = contract_register_tensors(
        (node,), Signature((*controls, *targets)), incoming, outgoing
    )
    np.testing.assert_array_equal(tensor.numpy(), source.tensor_contract())
    measured = Instruction(
        "measure", (0,), metadata={"classical_bit": 0, "is_dynamic": True}
    )
    body = classically_controlled_ir(
        ((0,),), (ControlRegister(1, (polarity,)),), register_identity_ir((1, 2))
    )
    initial = torch.tensor(
        [complex(i + 1, 2 - i) for i in range(8)], dtype=torch.complex128
    )
    initial /= initial.norm()
    for history in (0, 1):
        expected = torch.zeros_like(initial)
        expected[history * 4 : (history + 1) * 4] = initial[
            history * 4 : (history + 1) * 4
        ]
        torch.testing.assert_close(
            forced_branch(initial, (measured, *body), (history,)),
            expected,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("width", (1, 3))
def test_identity_whole_atom_score_matches_original(width: int) -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import Identity
    from qualtran.drawing import get_musical_score_data

    from flagquantum.core.ir.program.register_types import opaque_register_type
    from flagquantum.core.ir.program.registers import Register
    from flagquantum.drawer.identity_symbols import identity_wire_symbol
    from flagquantum.drawer.register_score import RegisterStep, layout_register_score
    from flagquantum.drawer.register_score_data import register_score_data
    from tests.unit.test_drawer_component_score import assert_source_geometry

    original = Identity(width)
    builder, inputs = BloqBuilder.from_signature(original.signature)
    source = builder.finalize(**builder.add_d(original, **inputs))
    reg = Register("q", opaque_register_type(width))
    step = RegisterStep(
        "I", ("q",), ("q",), gate="i", input_registers=(reg,), output_registers=(reg,)
    )
    score = layout_register_score({"q": width}, ("q",), (step,))
    actual = register_score_data(score, register_wire_symbol=identity_wire_symbol)
    assert_source_geometry(actual, get_musical_score_data(source), ordered_ports=True)
