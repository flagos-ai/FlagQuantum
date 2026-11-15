"""Check bit-preserving register views against pinned Qualtran Cast."""

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler._component_trace.qec_gate_cost import (
    QECGateCost,
    query_qec_gate_cost,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    BookkeepingCall,
)
from flagquantum.compiler.registers.layout.register_cast import RegisterCast

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
def test_original_thirty_two_bit_fixed_point_view_matches_source():
    import cirq
    import numpy as np
    from qualtran.bloqs.bookkeeping.cast import _cast
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.simulation.fixed_point_encoding import fixed_point_real_value

    original = _cast()
    native = RegisterCast(32, 32)
    wires = tuple(reversed(range(32)))
    foreign_wires = np.array([cirq.LineQubit(wire) for wire in wires])
    operation, outputs = original.as_cirq_op(None, reg=foreign_wires)
    assert operation is None and outputs["reg"] is foreign_wires
    assert (
        native.bind(wires)
        == native.adjoint().bind(wires)
        == tuple(qubit.x for qubit in outputs["reg"])
    )
    assert native.logical_width == get_cost_value(original, QubitCount()) == 32
    assert get_cost_value(original, QECGatesCost()).asdict() == {}
    for value in (-(1 << 31), -1, 0, 1, (1 << 31) - 1):
        converted = native.reinterpret_integer(value, source_signed=True)
        assert original.call_classically(reg=value) == (converted,)
        assert fixed_point_real_value(converted, 32, 32) == (
            original.out_dtype.float_from_fixed_width_int(converted)
        )


@pytest.mark.ftqc_reference
def test_original_cast_add_cast_has_only_the_add_stream_and_restores_work():
    from itertools import product
    from math import sqrt

    from qualtran.bloqs.for_testing import TestCastToFrom

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.registers.arithmetic.addition import add_ir
    from flagquantum.compiler.registers.arithmetic.arithmetic_resource_model import (
        source_add_cost,
    )
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.simulation.classical_addition import register_add_classical_values
    from flagquantum.simulation.fixed_point_encoding import fixed_point_real_value
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_indexed_component_cost_profiles import _assert_model
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    source = TestCastToFrom()
    cast = RegisterCast(4, 4)
    a, b, work, slots = (0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10), (12, 13, 14)
    gates = add_ir(a, cast.bind(b), work, slots)
    assert cast.adjoint().bind(b) == b
    assert_source(
        gates,
        reference_arithmetic_gates(
            source, {"a": a, "b": b}, work, dict(zip(work, slots, strict=True))
        ),
    )
    for legacy in (False, True):
        _assert_model(source_add_cost(4, 4, legacy_shims=legacy), source, legacy)
    assert sum(gate.name in {"t", "tdg"} for gate in gates) == 12
    assert sum(gate.name == "measure" for gate in gates) == 3
    values = torch.arange(1, 257, dtype=torch.float64)
    ket = torch.complex(values, (values % 13) - 6)
    ket /= ket.norm()
    clean = torch.zeros(8, dtype=torch.complex128)
    clean[0] = 1
    original = torch.as_tensor(source.tensor_contract(), dtype=ket.dtype).reshape(
        256, 256
    )
    expected = torch.kron(original @ ket, clean)
    circuit = DynamicCircuit(11, dtype=ket.dtype).append(FieldOperation(11, gates))
    initial = torch.kron(ket, clean)
    for outcomes in product((0, 1), repeat=3):
        torch.testing.assert_close(
            forced_branch(initial, circuit.to_ir().instructions, outcomes),
            expected / sqrt(8),
            rtol=0,
            atol=3e-12,
        )

    raw = RegisterCast(8, 8)
    left, right = register_add_classical_values(7, raw.reinterpret_integer(2), 8)
    result = raw.adjoint().reinterpret_integer(right)
    assert TestCastToFrom(bitsize=8).call_classically(a=7, b=2) == (left, result)
    assert fixed_point_real_value(result, 8, 8) == 9 / 256


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "input_type,output_type,source_signed,target_signed",
    [
        ("uint", "int", False, True),
        ("int", "uint", True, False),
        ("uint", "ufxp", False, False),
        ("sfxp", "ufxp", True, False),
        ("ufxp", "sfxp", False, True),
        ("sfxp", "int", True, True),
    ],
)
def test_integer_cast_matches_every_source_bit_pattern(
    input_type: str, output_type: str, source_signed: bool, target_signed: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QFxp, QInt, QUInt
    from qualtran.bloqs.bookkeeping import Cast

    width = 5
    dtypes = {
        "int": QInt(width),
        "uint": QUInt(width),
        "ufxp": QFxp(width, 5),
        "sfxp": QFxp(width, 2, signed=True),
    }
    source = Cast(dtypes[input_type], dtypes[output_type])
    native = RegisterCast(width, width)
    for value in dtypes[input_type].get_classical_domain():
        converted = native.reinterpret_integer(
            value, source_signed=source_signed, target_signed=target_signed
        )
        assert source.call_classically(reg=value) == (converted,)
        assert (
            native.adjoint().reinterpret_integer(
                converted, source_signed=target_signed, target_signed=source_signed
            )
            == value
        )


@pytest.mark.ftqc_reference
def test_quantum_cast_matches_scalar_source_complex_action_and_resources() -> None:
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    from qualtran import QFxp, QInt, Side
    from qualtran.bloqs.bookkeeping import Cast
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = Cast(QInt(3), QFxp(3, 2, signed=True))
    assert tuple(reg.side for reg in source.signature) == (Side.LEFT, Side.RIGHT)
    op, source_ports = source.as_cirq_op(
        None, reg=np.array([cirq.LineQubit(wire) for wire in (3, 1, 0)])
    )
    assert op is None
    native = RegisterCast(3, 3)
    wires = native.bind((3, 1, 0))
    assert wires == tuple(wire.x for wire in source_ports["reg"])
    assert native.logical_width == get_cost_value(source, QubitCount()) == 3
    assert get_cost_value(source, QECGatesCost()).asdict() == {}
    assert query_qec_gate_cost(BookkeepingCall("cast", 3)) == QECGateCost()

    values = torch.arange(1, 17, dtype=torch.float64)
    state = torch.complex(values, values.remainder(5) - 2)
    state /= state.norm()
    circuit = fq.Circuit(4, inputs=state, dtype=state.dtype)
    original = state.reshape((2,) * 4).permute(2, 3, 1, 0).reshape(2, 8)
    actual = circuit.state().reshape((2,) * 4).permute(2, *wires).reshape(2, 8)
    tensor = torch.as_tensor(source.tensor_contract(), dtype=state.dtype).reshape(8, 8)
    torch.testing.assert_close(actual, original @ tensor.T, rtol=0, atol=2e-12)
    assert not circuit.to_ir().instructions


@pytest.mark.ftqc_reference
def test_shaped_symbolic_and_mixed_cast_boundaries() -> None:
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    from qualtran import CBit, QBit, QInt, QUInt
    from qualtran.bloqs.bookkeeping import Cast
    from qualtran.resource_counting import QubitCount, get_cost_value
    from sympy import Symbol

    source = Cast(QInt(3), QUInt(3), shape=(2,))
    shaped = RegisterCast(3, 3, shape=(2,))
    wires = (8, 1, 2, 4, 0, 3)
    source_wires = np.array([cirq.LineQubit(wire) for wire in wires]).reshape(2, 3)
    op, outputs = source.as_cirq_op(None, reg=source_wires)
    assert op is None
    assert shaped.bind(wires) == tuple(wire.x for wire in outputs["reg"].flat)
    assert shaped.logical_width == get_cost_value(source, QubitCount()) == 6
    assert shaped.adjoint().bind(wires) == wires
    assert source.adjoint().shape == ()
    with pytest.raises(TypeError, match="unhashable"):
        source.tensor_contract()

    n = Symbol("n", integer=True, positive=True)
    for target_width in (n, 5):
        symbolic = RegisterCast(n, target_width, shape=(2, 3))
        assert symbolic.logical_width == get_cost_value(
            Cast(QInt(n), QUInt(target_width), shape=(2, 3)), QubitCount()
        )
        with pytest.raises(ValueError, match="concrete widths"):
            symbolic.bind(())
    with pytest.raises(ValueError, match="opt-in"):
        RegisterCast(1, 1, target_quantum=False)
    with pytest.raises(ValueError, match="potentially dangerous"):
        Cast(QBit(), CBit())
    mixed = RegisterCast(1, 1, target_quantum=False, allow_quantum_to_classical=True)
    source_mixed = Cast(QBit(), CBit(), allow_quantum_to_classical=True)
    assert mixed.logical_width == get_cost_value(source_mixed, QubitCount()) == 1
    assert mixed.bind((4,)) == mixed.adjoint().bind((4,)) == (4,)
    reverse_mixed = Cast(CBit(), QBit(), allow_quantum_to_classical=True)
    for value in (0, 1):
        assert source_mixed.call_classically(reg=value) == (
            mixed.reinterpret_integer(value),
        )
        assert reverse_mixed.call_classically(reg=value) == (
            mixed.adjoint().reinterpret_integer(value),
        )
    assert RegisterCast(
        1, 1, source_quantum=False, target_quantum=False
    ).logical_width == (get_cost_value(Cast(CBit(), CBit()), QubitCount()))
    with pytest.raises(ValueError, match="potentially dangerous"):
        source_mixed.adjoint()


def test_cast_rejects_width_mismatch_and_invalid_bindings() -> None:
    with pytest.raises(ValueError, match="preserve"):
        RegisterCast(2, 3)
    with pytest.raises(ValueError, match="nonnegative"):
        RegisterCast(2, 2, shape=(-1,))
    with pytest.raises(ValueError, match="declared layout"):
        RegisterCast(2, 2).bind((1,))
    with pytest.raises(ValueError, match="distinct"):
        RegisterCast(2, 2).bind((1, 1))
    assert RegisterCast(0, 0).bind(()) == ()
