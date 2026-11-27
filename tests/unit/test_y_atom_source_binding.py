"""Actual Y/CY source captures preserve phases, presentation and adapter limits."""

from functools import partial
from hashlib import sha1
from pathlib import Path
from typing import Literal

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
    from qualtran.bloqs.basic_gates import y_gate

    for name, blob in (
        ("y_gate.py", "a3b305679692809a00407b630e2cc9f4227caebe"),
        ("y_gate_test.py", "9c7dab6b15ec1055f3a1d90fa08bb5d3a211a1a6"),
    ):
        raw = (Path(y_gate.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("gate", ("y", "cy"))
def test_actual_factories_atom_capture_and_complete_complex_action(
    gate: Literal["y", "cy"],
) -> None:
    from qualtran._infra.bloq import DecomposeTypeError
    from qualtran.bloqs.basic_gates.y_gate import (
        _CY_GATE_DOC,
        _Y_GATE_DOC,
        _cy_gate,
        _y_gate,
    )
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    factory, document = (
        (_y_gate, _Y_GATE_DOC) if gate == "y" else (_cy_gate, _CY_GATE_DOC)
    )
    source = factory.make()
    assert document.examples == (factory,) and document.call_graph_example is None
    assert source.adjoint() is source
    with pytest.raises(DecomposeTypeError, match="atomic"):
        source.decompose_bloq()
    _, leaves = source.call_graph()
    assert leaves == {source: 1}
    emitter = import_register_emitter(source)
    width = 1 if gate == "y" else 2
    assert emitter.requirements() == (0, 0) and emitter.binding_swaps == (0, 0)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    for snapshot in emitter.snapshots:
        restored = CircuitIR.from_json(snapshot.to_json())
        assert restored.instructions == (Instruction(gate, tuple(range(width))),)
        torch.testing.assert_close(
            get_unitary(restored, dtype=torch.complex128), matrix, rtol=0, atol=2e-12
        )
    wires = (1,) if gate == "y" else (2, 0)
    spectator = 0 if gate == "y" else 1
    order = (*wires, spectator)
    inverse_order = tuple(order.index(q) for q in range(width + 1))
    initial = torch.tensor(
        [complex(i + 1, 2 - i) for i in range(1 << (width + 1))], dtype=torch.complex128
    )
    initial /= initial.norm()
    expected = (
        (
            matrix
            @ initial.reshape((2,) * (width + 1)).permute(order).reshape(1 << width, 2)
        )
        .reshape((2,) * (width + 1))
        .permute(inverse_order)
        .flatten()
    )
    ports = {
        name: (wire,) for (name, _), wire in zip(emitter.layout, wires, strict=True)
    }
    for inverse in (False, True):
        gates = emitter(ports, (), (), inverse)
        assert gates == (Instruction(gate, wires),)
        circuit = Circuit(width + 1, inputs=initial, dtype=torch.complex128)
        circuit.append(FieldOperation(width + 1, gates))
        torch.testing.assert_close(
            circuit.state().flatten(), expected, rtol=0, atol=2e-12
        )
        circuit.append(FieldOperation(width + 1, gates))
        torch.testing.assert_close(
            circuit.state().flatten(), initial, rtol=0, atol=2e-12
        )
    assert (
        query_qec_gate_cost(CliffordCall(gate, width)).asdict()
        == get_cost_value(source, QECGatesCost()).asdict()
    )
    assert get_cost_value(source, QubitCount()) == width


@pytest.mark.parametrize("gate", ("y", "cy"))
def test_complete_native_score_matches_source(gate: Literal["y", "cy"]) -> None:
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran._infra.data_types import QBit
    from qualtran._infra.registers import Register as SourceRegister
    from qualtran.bloqs.basic_gates.y_gate import CYGate, YGate
    from qualtran.drawing.musical_score import TextBox, get_musical_score_data

    from flagquantum.core.ir.program.register_types import quantum_bit_type
    from flagquantum.core.ir.program.registers import Register
    from flagquantum.drawer.basis_symbols import pauli_basis_wire_symbol
    from flagquantum.drawer.register_score import RegisterStep, layout_register_score
    from flagquantum.drawer.register_score_data import register_score_data
    from tests.unit.test_drawer_component_score import assert_source_geometry
    from tests.unit.test_register_tensor import native_signature

    source = YGate() if gate == "y" else CYGate()
    registers = tuple(native_signature(source.signature))
    names = tuple(reg.name for reg in registers)
    step = RegisterStep(
        str(source), names, names, input_registers=registers, output_registers=registers
    )
    score = layout_register_score(dict.fromkeys(names, 1), names, (step,))
    builder, inputs = BloqBuilder.from_signature(source.signature)
    original = builder.finalize(**builder.add_d(source, **inputs))
    callback = partial(pauli_basis_wire_symbol, gate=gate)
    assert_source_geometry(
        register_score_data(score, register_wire_symbol=callback),
        get_musical_score_data(original),
        ordered_ports=True,
    )
    unknown = Register("unknown", quantum_bit_type())
    if gate == "y":
        assert callback(step, unknown, ()) == ("TextBox", {"text": "Y"})
        symbol = source.wire_symbol(SourceRegister("unknown", QBit()))
        assert isinstance(symbol, TextBox)
        assert symbol.text == "Y"
    else:
        with pytest.raises(ValueError, match="Unknown register"):
            callback(step, unknown, ())
        with pytest.raises(ValueError, match="Unknown register"):
            source.wire_symbol(SourceRegister("unknown", QBit()))


def test_original_cy_inverse_mapping_failure_is_retained() -> None:
    import cirq
    from qualtran.bloqs.basic_gates.y_gate import CYGate
    from qualtran.cirq_interop._cirq_to_bloq import CirqGateAsBloq, cirq_gate_to_bloq

    canonical = cirq.Y.controlled()
    converted = cirq_gate_to_bloq(canonical)
    if isinstance(canonical, cirq.ControlledGate):
        assert converted == CYGate()
    else:
        assert converted != CYGate()
        assert isinstance(converted, CirqGateAsBloq)
    assert cirq_gate_to_bloq(cirq.ControlledGate(cirq.Y)) == CYGate()
