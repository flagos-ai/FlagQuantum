"""Complete bitwise factories use native capture, phases and score symbols."""

from functools import partial
from hashlib import sha1
from pathlib import Path
from typing import Literal

import pytest
import torch
from sympy import Symbol

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.bits.bitwise import (
    bitwise_not_ir,
    xor_constant_ir,
    xor_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.value_math import SymbolicInt
from flagquantum.drawer.bitwise_symbols import bitwise_wire_symbol
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import register_score_data
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.test_register_tensor import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.arithmetic import bitwise

    for filename, blob in (
        ("bitwise.py", "24835990709339b96cde9b2302a2525c09345ab5"),
        ("bitwise_test.py", "a8149d90052a692728bc0f39b99f10779ef66036"),
    ):
        raw = (Path(bitwise.__file__).parent / filename).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("name", ("_xor", "_xork", "_bitwise_not"))
def test_actual_factories_capture_binding_json_and_complete_complex_action(
    name: str,
) -> None:
    from qualtran.bloqs.arithmetic import bitwise
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = getattr(bitwise, name).make()
    assert source.adjoint() is source
    emitter = import_register_emitter(source)
    assert emitter.requirements() == (0, 0) and emitter.binding_swaps == (0, 0)
    width = sum(size for _, size in emitter.layout)
    logical_wires = tuple(reversed(range(width)))
    ports = {}
    offset = 0
    for port, size in emitter.layout:
        ports[port] = logical_wires[offset : offset + size]
        offset += size
    if name == "_xor":
        native = xor_ir(ports["x"], ports["y"])
        assert bitwise._XOR_DOC.examples[0].make() == source
    elif name == "_xork":
        native = xor_constant_ir(ports["x"], source.k)
    else:
        native = bitwise_not_ir(ports["x"])
        assert bitwise._BITWISE_NOT_DOC.examples[0].make() == source
    initial = torch.tensor(
        [complex(i + 1, 3 - 2 * i) for i in range(1 << (width + 1))],
        dtype=torch.complex128,
    )
    initial /= initial.norm()
    # Read the original value callback independently of the emitted gate sequence.
    mapped = []
    for basis in range(1 << (width + 1)):
        inputs = {
            port: sum(
                ((basis >> (width - wire)) & 1) << (len(wires) - index - 1)
                for index, wire in enumerate(wires)
            )
            for port, wires in ports.items()
        }
        outputs = source.on_classical_vals(**inputs)
        result = basis
        for port, wires in ports.items():
            for index, wire in enumerate(wires):
                shift = width - wire
                bit = (outputs[port] >> (len(wires) - index - 1)) & 1
                result = (result & ~(1 << shift)) | (bit << shift)
        mapped.append(result)
    expected = torch.zeros_like(initial)
    expected[mapped] = initial
    for snapshot in emitter.snapshots:
        restored = CircuitIR.from_json(snapshot.to_json())
        assert restored.instructions == snapshot.instructions
        assert len(restored.instructions) == len(native)
    for inverse in (False, True):
        actual = emitter(ports, (), (), inverse)
        assert actual == native
        circuit = Circuit(width + 1, inputs=initial, dtype=torch.complex128)
        circuit.append(FieldOperation(width + 1, actual))
        torch.testing.assert_close(
            circuit.state().flatten(), expected, rtol=0, atol=2e-12
        )
        circuit.append(FieldOperation(width + 1, actual))
        torch.testing.assert_close(
            circuit.state().flatten(), initial, rtol=0, atol=2e-12
        )
    model = get_cost_value(source, QECGatesCost())
    assert model.total_t_count() == model.measurement == 0
    assert model.clifford == len(native)
    assert get_cost_value(source, QubitCount()) == width


@pytest.mark.parametrize("name", ("_xor_symb", "_bitwise_not_symb"))
def test_actual_symbolic_factories_retain_physical_capture_boundary(name: str) -> None:
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.arithmetic import bitwise

    source = getattr(bitwise, name).make()
    if name == "_xor_symb":
        with pytest.raises(DecomposeTypeError, match="concrete"):
            source.decompose_bloq()
    else:
        from qualtran.bloqs.basic_gates import OnEach, XGate

        children = tuple(source.decompose_bloq().bloq_instances)
        assert len(children) == 1
        child = children[0].bloq
        assert child == OnEach(source.dtype.num_qubits, XGate(), source.dtype)
        with pytest.raises(DecomposeTypeError, match="symbolic bitsize"):
            child.decompose_bloq()
    with pytest.raises(ValueError, match="concrete positive shapes"):
        import_register_emitter(source)


@pytest.mark.parametrize("operation", ("xor", "xor_constant", "bitwise_not"))
@pytest.mark.parametrize("constant", (-3, 6, Symbol("k")))
def test_complete_native_score_matches_source(
    operation: Literal["xor", "xor_constant", "bitwise_not"],
    constant: SymbolicInt,
) -> None:
    from qualtran import BloqBuilder, QUInt
    from qualtran.bloqs.arithmetic.bitwise import BitwiseNot, Xor, XorK
    from qualtran.drawing import get_musical_score_data

    from flagquantum.core.ir.program.register_types import quantum_bit_type
    from flagquantum.core.ir.program.registers import Register
    from tests.unit.test_drawer_component_score import assert_source_geometry

    source = (
        Xor(QUInt(4))
        if operation == "xor"
        else (
            XorK(QUInt(4), constant)
            if operation == "xor_constant"
            else BitwiseNot(QUInt(4))
        )
    )
    registers = tuple(native_signature(source.signature))
    names = tuple(reg.name for reg in registers)
    step = RegisterStep(
        str(source), names, names, input_registers=registers, output_registers=registers
    )
    score = layout_register_score(dict.fromkeys(names, 4), names, (step,))
    builder, inputs = BloqBuilder.from_signature(source.signature)
    original = builder.finalize(**builder.add_d(source, **inputs))
    callback = partial(bitwise_wire_symbol, operation=operation, constant=constant)
    assert_source_geometry(
        register_score_data(score, register_wire_symbol=callback),
        get_musical_score_data(original),
        ordered_ports=True,
    )
    unknown = Register("unknown", quantum_bit_type())
    assert callback(step, unknown, ()) == (
        "TextBox",
        {"text": source.wire_symbol(unknown).text},
    )
