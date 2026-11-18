"""Bind complete basis-file display and concrete Pauli source interfaces."""

from functools import partial
from hashlib import sha1
from pathlib import Path
from typing import Literal

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from flagquantum.simulation.unitary import get_unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.basic_gates import x_basis

    for name, blob in (
        ("x_basis.py", "8b903d048e307789d71539441088efeec2868f99"),
        ("x_basis_test.py", "cf8ce512efec89ede19cddf60a27cdaff6d1cddf"),
        ("z_basis.py", "48231b5d507f076ed3ea95a4d8486f375cca7d01"),
        ("z_basis_test.py", "35de62edcda9ce3230d31012f1fe757c4572bc38"),
    ):
        raw = (Path(x_basis.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("gate", ("x", "z", "cz"))
def test_actual_pauli_atom_capture_and_complete_complex_action(
    gate: Literal["x", "z", "cz"],
) -> None:
    from qualtran.bloqs.basic_gates import CZ, XGate, ZGate

    source = {"x": XGate(), "z": ZGate(), "cz": CZ()}[gate]
    emitter = import_register_emitter(source)
    width = 2 if gate == "cz" else 1
    assert emitter.requirements() == (0, 0) and emitter.binding_swaps == (0, 0)
    assert source.adjoint() is source
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    for snapshot in emitter.snapshots:
        restored = CircuitIR.from_json(snapshot.to_json())
        assert restored.instructions == (Instruction(gate, tuple(range(width))),)
        torch.testing.assert_close(
            get_unitary(restored, dtype=torch.complex128), matrix, rtol=0, atol=2e-12
        )
    wires = tuple(reversed(range(width)))
    order = (*wires, width)
    inverse_order = tuple(order.index(q) for q in range(width + 1))
    initial = torch.tensor(
        [complex(i + 1, 2 - i) for i in range(1 << (width + 1))],
        dtype=torch.complex128,
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


@pytest.mark.parametrize(
    "kind",
    (
        "plus_state",
        "minus_state",
        "plus_effect",
        "minus_effect",
        "zero_state",
        "one_state",
        "zero_effect",
        "one_effect",
        "int_state",
        "int_effect",
        "x",
        "z",
        "cz",
        "meas_x",
        "meas_z",
    ),
)
def test_complete_basis_source_score_and_factories(
    kind: Literal[
        "plus_state",
        "minus_state",
        "plus_effect",
        "minus_effect",
        "zero_state",
        "one_state",
        "zero_effect",
        "one_effect",
        "int_state",
        "int_effect",
        "x",
        "z",
        "cz",
        "meas_x",
        "meas_z",
    ],
) -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import XGate, x_basis, z_basis
    from qualtran.drawing import get_musical_score_data

    from flagquantum.drawer.basis_symbols import (
        basis_vector_wire_symbol,
        pauli_basis_wire_symbol,
    )
    from flagquantum.drawer.register_score import RegisterStep, layout_register_score
    from flagquantum.drawer.register_score_data import register_score_data
    from tests.unit.test_classical_action_tensor import signature_from_source
    from tests.unit.test_drawer_component_score import assert_source_geometry

    module = (
        x_basis
        if kind
        in {"plus_state", "minus_state", "plus_effect", "minus_effect", "meas_x"}
        else z_basis
    )
    source = (
        XGate()
        if kind == "x"
        else getattr(module, "_" + ("zgate" if kind == "z" else kind)).make()
    )
    registers = tuple(signature_from_source(source.signature))
    left = tuple(r.name for r in registers if r.side.name != "RIGHT")
    right = tuple(r.name for r in registers if r.side.name != "LEFT")
    step = RegisterStep(
        str(source),
        left,
        right,
        input_registers=tuple(r for r in registers if r.name in left),
        output_registers=tuple(r for r in registers if r.name in right),
    )
    widths = {r.name: int(r.bitsize) for r in registers}
    score = layout_register_score(widths, left, (step,))
    builder, inputs = BloqBuilder.from_signature(source.signature)
    original = builder.finalize(**builder.add_d(source, **inputs))
    if kind == "x" or kind == "z" or kind == "cz":
        supplier = partial(pauli_basis_wire_symbol, gate=kind)
        actual = register_score_data(score, register_wire_symbol=supplier)
    elif "state" in kind or "effect" in kind:
        label = str(source)[1:-1]
        actual = register_score_data(
            score, register_wire_symbol=partial(basis_vector_wire_symbol, label=label)
        )
    else:
        actual = register_score_data(score)
    assert_source_geometry(actual, get_musical_score_data(original), ordered_ports=True)


def test_original_vector_and_measurement_capture_refusals() -> None:
    import cirq
    from qualtran.bloqs.basic_gates import (
        IntState,
        MeasureX,
        MeasureZ,
        MinusState,
        OneState,
        PlusState,
        ZeroState,
    )

    for source in (ZeroState(), OneState(), PlusState(), MinusState()):
        with pytest.raises(ValueError, match="There is no Cirq equivalent"):
            source.adjoint().as_cirq_op(qubit_manager=cirq.SimpleQubitManager())
        with pytest.raises(ValueError, match="There is no Cirq equivalent"):
            import_register_emitter(
                source, local_ports=tuple(r.name for r in source.signature)
            )
    import numpy as np

    integer = IntState(2, 3)
    manager = cirq.SimpleQubitManager()
    wires = np.asarray(manager.qalloc(3))
    operation, outputs = integer.adjoint().as_cirq_op(
        qubit_manager=manager,
        val=wires,
    )
    assert outputs == {}
    with pytest.raises(ValueError, match="There is no Cirq equivalent for <0"):
        cirq.decompose(operation)
    with pytest.raises(ValueError, match="There is no Cirq equivalent for <0"):
        import_register_emitter(integer, local_ports=("val",))
    for measurement in (MeasureX(), MeasureZ()):
        with pytest.raises(ValueError, match="requires through ports"):
            import_register_emitter(measurement)


def test_pauli_unknown_register_and_vector_direction() -> None:
    from qualtran import QBit
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.basic_gates import CZ, XGate, ZGate

    from flagquantum.core.ir.program.register_types import quantum_bit_type
    from flagquantum.core.ir.program.registers import Register, RegisterSide
    from flagquantum.drawer.basis_symbols import (
        basis_vector_wire_symbol,
        pauli_basis_wire_symbol,
    )
    from flagquantum.drawer.register_score import RegisterStep

    reg = Register("unknown", quantum_bit_type())
    step = RegisterStep("test")
    assert pauli_basis_wire_symbol(step, reg, (), gate="x") == ("ModPlus", {})
    assert pauli_basis_wire_symbol(step, reg, (), gate="z") == (
        "TextBox",
        {"text": "Z"},
    )
    assert (
        type(XGate().wire_symbol(SourceRegister("unknown", QBit()))).__name__
        == "ModPlus"
    )
    assert ZGate().wire_symbol(SourceRegister("unknown", QBit())).text == "Z"
    with pytest.raises(ValueError, match="Unknown wire symbol register name: unknown"):
        pauli_basis_wire_symbol(step, reg, (), gate="cz")
    with pytest.raises(ValueError, match="Unknown wire symbol register name: unknown"):
        CZ().wire_symbol(SourceRegister("unknown", QBit()))
    assert basis_vector_wire_symbol(step, reg, (), label="a") == (
        "TextBox",
        {"text": "a"},
    )
    assert basis_vector_wire_symbol(
        step, Register("q", quantum_bit_type(), side=RegisterSide.LEFT), (), label="1"
    ) == ("RarrowTextBox", {"text": "1"})
