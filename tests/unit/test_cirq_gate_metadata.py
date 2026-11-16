"""Compare Cirq wrapper ports, tensors and glyphs against the frozen source."""

import numpy as np
import pytest

from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._gate_metadata import (
    cirq_gate_signature,
    cirq_gate_tensor,
    cirq_gate_wire_symbol,
)
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_gate

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_raw_gate_signature_and_tensor_match_scalar_and_binary_sources() -> None:
    cirq = pytest.importorskip("cirq")
    source = pytest.importorskip("qualtran.cirq_interop._cirq_to_bloq")
    for gate in (cirq.GlobalPhaseGate(-1j), cirq.X, cirq.rx(0.123), cirq.TOFFOLI):
        wrapper = source.CirqGateAsBloq(gate)
        actual = cirq_gate_signature(gate)
        assert [(r.name, r.bitsize, r.shape, r.side.name) for r in actual] == [
            (r.name, r.bitsize, r.shape, r.side.name) for r in wrapper.signature
        ]
        matrix = wrapper.tensor_contract()
        tensor = cirq_gate_tensor(gate)
        np.testing.assert_allclose(tensor, np.reshape(matrix, tensor.shape), atol=2e-12)
    with pytest.raises(ValueError, match="binary quantum ports"):
        cirq_gate_signature(cirq.IdentityGate(qid_shape=(3,)))
    with pytest.raises(NotImplementedError, match="only supported for unitary gates"):
        source.CirqGateAsBloq(cirq.ResetChannel()).as_composite_bloq().tensor_contract()
    with pytest.raises(NotImplementedError, match="require a unitary"):
        cirq_gate_tensor(cirq.ResetChannel())


def test_tensor_projects_sided_words_and_preserves_bit_axis_order() -> None:
    cirq = pytest.importorskip("cirq")
    source = pytest.importorskip("qualtran.simulation.tensor._tensor_data_manipulation")
    from qualtran import QAny, QBit
    from qualtran import Register as SourceRegister
    from qualtran import Side as SourceSide
    from qualtran import Signature as SourceSignature

    sided_left, sided_right = (
        Register("left", quantum_bit_type(), side=RegisterSide.LEFT),
        Register("right", quantum_bit_type(), side=RegisterSide.RIGHT),
    )
    sided = Signature((sided_left, sided_right))
    ref_sided = SourceSignature(
        [
            SourceRegister("left", QBit(), side=SourceSide.LEFT),
            SourceRegister("right", QBit(), side=SourceSide.RIGHT),
        ]
    )
    expected = source.tensor_data_from_unitary_and_signature(
        cirq.unitary(cirq.SWAP), ref_sided
    )
    assert expected.shape == (2, 2)
    np.testing.assert_array_equal(cirq_gate_tensor(cirq.SWAP, sided), expected)
    np.testing.assert_array_equal(expected, np.eye(2))
    word = Signature((Register("word", opaque_register_type(2)),))
    ref_word = SourceSignature([SourceRegister("word", QAny(2))])
    expected = source.tensor_data_from_unitary_and_signature(
        cirq.unitary(cirq.CNOT), ref_word
    )
    actual = cirq_gate_tensor(cirq.CNOT, word)
    assert actual.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(actual, expected.reshape((2, 2, 2, 2)))
    with pytest.raises(ValueError, match="shape"):
        cirq_gate_tensor(cirq.CNOT, Signature((sided_left,)))


def test_wire_glyphs_match_source_controls_sides_and_shaped_slice() -> None:
    cirq = pytest.importorskip("cirq")
    source = pytest.importorskip("qualtran.cirq_interop._cirq_to_bloq")
    from qualtran import QAny, QBit
    from qualtran import Register as SourceRegister
    from qualtran import Side as SourceSide
    from qualtran import Signature as SourceSignature

    for gate in (cirq.CNOT, cirq.ControlledGate(cirq.X, control_values=(0,))):
        native_sig = cirq_gate_signature(gate)
        ref_sig = source.CirqGateAsBloq(gate).signature
        for bit in range(2):
            ref = source._wire_symbol_from_gate(gate, ref_sig, ref_sig[0], (bit,))
            name, props = cirq_gate_wire_symbol(gate, native_sig, native_sig[0], (bit,))
            assert name == type(ref).__name__
            if name == "Circle":
                assert props == {"filled": ref.filled}
            elif name == "ModPlus":
                assert props == {}
            else:
                assert name == "TextBox" and props == {"text": ref.text}

    class Labelled(cirq.Gate):  # type: ignore[name-defined, misc]
        def _num_qubits_(self) -> int:
            return 5

        def _circuit_diagram_info_(self, _args: object) -> tuple[str, ...]:
            return "in", "word-first", "word-second", "ignored-2", "ignored-3"

    gate = Labelled()
    native_sig = Signature(
        (
            Register("input", quantum_bit_type(), side=RegisterSide.LEFT),
            Register("words", opaque_register_type(2), (2,), RegisterSide.RIGHT),
        )
    )
    ref_sig = SourceSignature(
        [
            SourceRegister("input", QBit(), side=SourceSide.LEFT),
            SourceRegister("words", QAny(2), (2,), SourceSide.RIGHT),
        ]
    )
    for port, index in ((0, ()), (1, (0,)), (1, (1,))):
        ref = source._wire_symbol_from_gate(gate, ref_sig, ref_sig[port], index)
        name, props = cirq_gate_wire_symbol(gate, native_sig, native_sig[port], index)
        assert name == type(ref).__name__ and props == {"text": ref.text}


def test_explicit_gate_signature_defaults_to_operation_operand_order() -> None:
    cirq = pytest.importorskip("cirq")
    source = pytest.importorskip("qualtran.cirq_interop._cirq_to_bloq_test")
    first, second = cirq.NamedQubit.range(2, prefix="wire")
    signature = Signature(
        (Register("junk", opaque_register_type(2), side=RegisterSide.LEFT),)
    )
    operation = source.LeftOnlyGate().on(second, first)
    native = import_cirq_register_gate(operation, signature=signature)
    manual = import_cirq_register_gate(
        operation,
        signature=signature,
        in_quregs={"junk": (second, first)},
        out_quregs={},
    )
    assert native is not None and native == manual
    bound = bind_register_program(
        native, ((0, 1),), input_ownership=((0, False), (1, False))
    )
    assert [i.name for i in bound.circuit.instructions] == ["cx", "reset", "reset"]
    too_short = Signature((Register("one", quantum_bit_type()),))
    with pytest.raises(ValueError, match="operands do not match"):
        import_cirq_register_gate(operation, signature=too_short)
