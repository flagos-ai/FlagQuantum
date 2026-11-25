"""Bind actual Hadamard atoms, their display and external adapter assertions."""

from dataclasses import replace
from hashlib import sha1
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.compound_control_resource import (
    CONTROLLED_HADAMARD_COST,
)
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import CliffordCall
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from flagquantum.simulation.unitary import get_unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.basic_gates import hadamard

    for name, blob in (
        ("hadamard.py", "371672f6e39c8a5ee95e8ede95d6e3cb3c29b95c"),
        ("hadamard_test.py", "9af68dcaca8ef829fe4a65a6299cb4956294fb0c"),
    ):
        raw = (Path(hadamard.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("controlled", (False, True))
def test_actual_atoms_named_binding_adjoint_and_independent_models(
    controlled: bool,
) -> None:
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.basic_gates.hadamard import _chadamard, _hadamard
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = _chadamard.make() if controlled else _hadamard.make()
    assert source.adjoint() is source
    with pytest.raises(DecomposeTypeError, match="atomic"):
        source.decompose_bloq()
    emitter = import_register_emitter(source)
    layout = (("ctrl", 1), ("target", 1)) if controlled else (("q", 1),)
    assert emitter.layout == layout and emitter.requirements() == (0, 0)
    assert emitter.binding_swaps == (0, 0)
    emitter = replace(
        emitter,
        snapshots=(
            CircuitIR.from_json(emitter.snapshots[0].to_json()),
            CircuitIR.from_json(emitter.snapshots[1].to_json()),
        ),
    )
    leaf = Instruction("ch", (0, 1)) if controlled else Instruction("h", (0,))
    assert all(ir.instructions == (leaf,) for ir in emitter.snapshots)
    # Reverse physical binding while retaining an entangled spectator.
    bindings = {"ctrl": (2,), "target": (0,)} if controlled else {"q": (2,)}
    wires = (2, 0) if controlled else (2,)
    width = 3
    order = (*wires, *(q for q in range(width) if q not in wires))
    inverse_order = tuple(order.index(q) for q in range(width))
    state = torch.tensor(
        [complex(i + 1, 2 - 3 * i) for i in range(8)], dtype=torch.complex128
    )
    state /= state.norm()
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    expected = (
        (
            matrix
            @ state.reshape((2,) * width).permute(order).reshape(1 << len(wires), -1)
        )
        .reshape((2,) * width)
        .permute(inverse_order)
        .reshape(-1)
    )
    for inverse in (False, True):
        gates = emitter(bindings, (), (), inverse)
        assert gates == (Instruction(leaf.name, wires),)
        circuit = Circuit(width, inputs=state, dtype=torch.complex128)
        circuit.append(FieldOperation(width, gates))
        torch.testing.assert_close(
            circuit.state().flatten(), expected, rtol=0, atol=2e-12
        )
        circuit.append(FieldOperation(width, emitter(bindings, (), (), not inverse)))
        torch.testing.assert_close(circuit.state().flatten(), state, rtol=0, atol=2e-12)
    # Both model paths are independent of the single emitted atom.
    model = CONTROLLED_HADAMARD_COST if controlled else CliffordCall("h", 1)
    assert (
        query_qec_gate_cost(model).asdict()
        == get_cost_value(source, QECGatesCost()).asdict()
    )
    assert get_cost_value(source, QubitCount()) == len(wires)
    circuit = Circuit(len(wires), dtype=torch.complex128).append(
        FieldOperation(len(wires), (leaf,))
    )
    torch.testing.assert_close(get_unitary(circuit), matrix, rtol=0, atol=2e-12)


def test_original_hadamard_pennylane_and_cirq_round_trips() -> None:
    import cirq
    from qualtran.bloqs.basic_gates import Hadamard
    from qualtran.cirq_interop import cirq_gate_to_bloq

    from flagquantum.ecosystem.cirq import from_cirq, to_cirq
    from flagquantum.ecosystem.pennylane import from_pennylane, to_pennylane

    qml = pytest.importorskip("pennylane")
    source = Hadamard()
    ir = CircuitIR(1, (Instruction("h", (0,)),))
    assert to_cirq(ir) == cirq.Circuit(cirq.H(cirq.LineQubit(0)))
    assert from_cirq(to_cirq(ir)).instructions == ir.instructions
    script = to_pennylane(ir)
    assert script.operations[0] == source.as_pl_op(wires=[0]) == qml.Hadamard(wires=[0])
    assert from_pennylane(script).instructions == ir.instructions
    torch.testing.assert_close(
        torch.as_tensor(qml.matrix(script.operations[0]), dtype=torch.complex128),
        torch.as_tensor(source.tensor_contract()),
        rtol=0,
        atol=2e-12,
    )
    assert cirq_gate_to_bloq(cirq.H.controlled()) == source.controlled()
    controlled = CircuitIR(2, (Instruction("ch", (0, 1)),))
    assert from_cirq(to_cirq(controlled)).instructions == controlled.instructions


@pytest.mark.parametrize("controlled", (False, True))
def test_whole_atom_source_score_and_unknown_register(controlled: bool) -> None:
    from qualtran import BloqBuilder, QBit
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.basic_gates import Hadamard
    from qualtran.drawing import get_musical_score_data

    from flagquantum.core.ir.program.register_types import opaque_register_type
    from flagquantum.core.ir.program.registers import Register
    from flagquantum.drawer.hadamard_symbols import (
        controlled_hadamard_wire_symbol,
        hadamard_wire_symbol,
    )
    from flagquantum.drawer.register_score import RegisterStep, layout_register_score
    from flagquantum.drawer.register_score_data import register_score_data
    from tests.unit.test_drawer_component_score import assert_source_geometry

    original = Hadamard().controlled() if controlled else Hadamard()
    builder, inputs = BloqBuilder.from_signature(original.signature)
    source = builder.finalize(**builder.add_d(original, **inputs))
    names = ("ctrl", "target") if controlled else ("q",)
    regs = tuple(Register(name, opaque_register_type(1)) for name in names)
    step = RegisterStep(
        "CH" if controlled else "H",
        names,
        names,
        gate="ch" if controlled else "h",
        input_registers=regs,
        output_registers=regs,
    )
    supplier = controlled_hadamard_wire_symbol if controlled else hadamard_wire_symbol
    score = layout_register_score(dict.fromkeys(names, 1), names, (step,))
    assert_source_geometry(
        register_score_data(score, register_wire_symbol=supplier),
        get_musical_score_data(source),
        ordered_ports=True,
    )
    unknown = Register("unknown", opaque_register_type(1))
    if controlled:
        with pytest.raises(ValueError, match="Unknown register"):
            supplier(step, unknown, ())
        with pytest.raises(ValueError, match="Unknown register"):
            original.wire_symbol(SourceRegister("unknown", QBit()))
    else:
        assert supplier(step, unknown, ()) == ("TextBox", {"text": "H"})
