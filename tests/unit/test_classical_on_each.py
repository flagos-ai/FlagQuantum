"""Bound OnEach evaluation follows original full-graph ordering and bit storage."""

from dataclasses import asdict
from itertools import product

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_on_each import on_each_classical_value
from flagquantum.simulation.integer_encoding import integer_from_bits, integer_to_bits
from tests.unit.test_classical_comparison import same_value

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


class SymbolicInteger(int):
    def is_symbolic(self):
        return True


def original_child(kind="xz", inverse=False):
    from qualtran import Bloq, Signature
    from qualtran.bloqs.basic_gates import XGate, ZGate

    if kind == "x":
        return XGate()
    if kind == "z":
        return ZGate()

    class BitRecipe(Bloq):
        @property
        def signature(self):
            return Signature.build(q=1)

        def build_composite_bloq(self, bb, q):
            for gate in ((ZGate(), XGate()) if inverse else (XGate(), ZGate())):
                q = bb.add(gate, q=q)
            return {"q": q}

    return BitRecipe()


def native_child(kind="xz", inverse=False):
    from examples.algorithms.classical_on_each_values import bit_xz
    from flagquantum.simulation.classical_pauli_values import (
        x_classical_value,
        z_basis_phase,
    )

    if kind == "x":
        return lambda bit: (x_classical_value(bit), 1.0)
    if kind == "z":
        return lambda bit: (bit, 1.0 * z_basis_phase(bit))
    return lambda bit: bit_xz(bit, inverse=inverse)


def codec_pair(kind, width):
    from flagquantum.simulation.classical_registers import bind_integer_register_values
    from flagquantum.simulation.ones_complement_encoding import (
        ones_complement_from_bits,
        ones_complement_to_bits,
        validate_ones_complement_value,
    )

    def validate(value):
        if kind == "opaque":
            return
        if kind == "ones":
            if not isinstance(value, sympy.Expr):
                validate_ones_complement_value(value, width)
        else:
            bind_integer_register_values(
                "q", {"q": value}, width, signed=kind == "signed"
            )

    if kind == "ones":
        return (
            lambda v: ones_complement_to_bits(v, width),
            lambda b: ones_complement_from_bits(b, width),
            validate,
        )
    return (
        lambda v: integer_to_bits(v, width, signed=kind == "signed"),
        lambda b: integer_from_bits(b, width, signed=kind == "signed"),
        validate,
    )


def source_type(kind, width):
    from qualtran import QInt, QIntOnesComp, QUInt

    return (
        None
        if kind == "opaque"
        else {"uint": QUInt, "signed": QInt, "ones": QIntOnesComp}[kind](width)
    )


@pytest.mark.parametrize(
    "width,kind,gate,value",
    tuple(
        product(
            (2, 3),
            ("opaque", "uint", "signed", "ones"),
            ("x", "z", "xz"),
            (-4, -1, 0, 1, 3, 7, np.int8(1), 1.0, sympy.Integer(1)),
        )
    ),
)
def test_original_complete_graph_dtype_values_phases_and_diagnostics(
    width, kind, gate, value
):
    from qualtran.bloqs.basic_gates import OnEach
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    source = OnEach(width, original_child(gate), target_dtype=source_type(kind, width))
    encode, decode, validate = codec_pair(kind, width)
    try:
        expected, phase = do_phased_classical_simulation(source, {"q": value})
    except Exception as error:
        with pytest.raises(type(error)):
            on_each_classical_value(
                value, width, encode, decode, native_child(gate), validate
            )
    else:
        actual, actual_phase = on_each_classical_value(
            value, width, encode, decode, native_child(gate), validate
        )
        same_value(actual, expected["q"])
        same_value(actual_phase, phase)


@pytest.mark.parametrize(
    "width",
    [
        0,
        -1,
        True,
        np.int64(2),
        sympy.Integer(2),
        sympy.Symbol("n"),
        SymbolicInteger(2),
        SymbolicInteger(0),
    ],
)
@pytest.mark.parametrize("kind", ["uint", "signed"])
def test_concrete_layout_and_expression_guard_follow_original_after_input_binding(
    width, kind
):
    from qualtran.bloqs.basic_gates import OnEach
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    source = OnEach(width, original_child("x"), target_dtype=source_type(kind, width))
    encode, decode, validate = codec_pair(kind, width)
    try:
        expected, phase = do_phased_classical_simulation(source, {"q": 0})
    except Exception as error:
        with pytest.raises(type(error)):
            on_each_classical_value(
                0, width, encode, decode, native_child("x"), validate
            )
    else:
        actual, actual_phase = on_each_classical_value(
            0, width, encode, decode, native_child("x"), validate
        )
        same_value(actual, expected["q"])
        same_value(actual_phase, phase)


def test_zero_width_original_independent_join_runs_before_split_encoder():
    from qualtran import QUInt
    from qualtran.bloqs.basic_gates import OnEach, XGate
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    source_seen, native_seen = [], []

    class RecordedUInt(QUInt):
        def to_bits(self, value):
            source_seen.append("encode")
            return super().to_bits(value)

        def from_bits(self, bits):
            source_seen.append(("decode", list(bits)))
            return super().from_bits(bits)

    def encode(value):
        native_seen.append("encode")
        return integer_to_bits(value, 0)

    def decode(bits):
        native_seen.append(("decode", list(bits)))
        return integer_from_bits(bits, 0)

    with pytest.raises(ValueError):
        do_phased_classical_simulation(
            OnEach(0, XGate(), target_dtype=RecordedUInt(0)), {"q": 0}
        )
    with pytest.raises(ValueError):
        on_each_classical_value(
            0,
            0,
            encode,
            decode,
            lambda bit: pytest.fail("No child"),
            lambda value: None,
        )
    assert source_seen == native_seen == [("decode", [])]


def test_zero_width_successful_custom_join_still_executes_later_split():
    from qualtran import QUInt
    from qualtran.bloqs.basic_gates import OnEach, XGate
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    source_seen, native_seen = [], []

    class EmptyCodec(QUInt):
        def to_bits(self, value):
            source_seen.append("encode")
            return []

        def from_bits(self, bits):
            source_seen.append("decode")
            return 0

    def encode(value):
        native_seen.append("encode")
        return []

    def decode(bits):
        native_seen.append("decode")
        return 0

    expected, phase = do_phased_classical_simulation(
        OnEach(0, XGate(), target_dtype=EmptyCodec(0)), {"q": 0}
    )
    actual, actual_phase = on_each_classical_value(
        0,
        0,
        encode,
        decode,
        lambda bit: pytest.fail("No child"),
        lambda value: None,
    )
    same_value(actual, expected["q"])
    same_value(actual_phase, phase)
    assert source_seen == native_seen == ["decode", "encode"]


@pytest.mark.parametrize("width", [0, 1, True, np.int64(1), 2, -1, SymbolicInteger(1)])
def test_fixed_bit_dtype_does_not_inherit_iteration_count_symbolic_parameters(width):
    from qualtran import QBit
    from qualtran.bloqs.basic_gates import OnEach, XGate
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    from flagquantum.simulation.bit_encoding import (
        bit_from_bits,
        bit_to_bits,
        validate_bit_value,
    )

    try:
        expected, phase = do_phased_classical_simulation(
            OnEach(width, XGate(), target_dtype=QBit()), {"q": 1}
        )
    except Exception as error:
        with pytest.raises(type(error)):
            on_each_classical_value(
                1,
                width,
                bit_to_bits,
                bit_from_bits,
                native_child("x"),
                validate_bit_value,
                register_width=1,
            )
    else:
        actual, actual_phase = on_each_classical_value(
            1,
            width,
            bit_to_bits,
            bit_from_bits,
            native_child("x"),
            validate_bit_value,
            register_width=1,
        )
        same_value(actual, expected["q"])
        same_value(actual_phase, phase)


def test_bit_results_retain_source_float_type_until_uint8_join_collection(monkeypatch):
    from qualtran import Bloq, QUInt, Signature
    from qualtran.bloqs.basic_gates import OnEach
    from qualtran.simulation.classical_sim import PhasedClassicalSimState

    import flagquantum.simulation.classical_on_each as native

    class FloatIdentity(Bloq):
        @property
        def signature(self):
            return Signature.build(q=1)

        def on_classical_vals(self, q):
            return {"q": float(q)}

        def basis_state_phase(self, q):
            return None

    original = OnEach(3, FloatIdentity(), target_dtype=QUInt(3)).decompose_bloq()
    state = PhasedClassicalSimState.from_cbloq(original, {"q": 5})
    expected = state.simulate()
    child_values = [
        value
        for soq, value in state.soq_assign.items()
        if isinstance(getattr(soq.binst, "bloq", None), FloatIdentity)
    ]
    assert child_values and all(type(value) is float for value in child_values)
    seen, collect = [], native.collect_integer_register_value

    def collecting(assignments, width, **kwargs):
        seen.extend(assignments.values())
        assert all(type(value) is float for value in assignments.values())
        result = collect(assignments, width, **kwargs)
        assert result.dtype == np.uint8
        return result

    monkeypatch.setattr(native, "collect_integer_register_value", collecting)
    encode, decode, validate = codec_pair("uint", 3)
    actual, phase = native.on_each_classical_value(
        5, 3, encode, decode, lambda bit: (float(bit), 1.0), validate
    )
    same_value(actual, expected["q"])
    same_value(phase, state.phase)
    assert seen == [1.0, 0.0, 1.0]


@pytest.mark.parametrize("inverse", [False, True])
def test_noncontiguous_append_stream_tensor_complex_phases_inverse_and_models(inverse):
    import torch
    from qualtran import QUInt
    from qualtran.bloqs.basic_gates import OnEach, XGate, ZGate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    import flagquantum as fq
    from examples.algorithms.classical_on_each_values import EachBitXZ, evaluate_word
    from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
    from flagquantum.compiler._component_trace.resource_generalization import (
        CliffordCall,
        OnEachCall,
    )
    from flagquantum.core.ir import Instruction
    from tests.unit.register_source_graph import walk_source_graph

    source = OnEach(3, original_child("xz", inverse), target_dtype=QUInt(3))
    circuit = fq.Circuit(4, dtype=torch.complex128)
    circuit.append(EachBitXZ(inverse=inverse))
    gates = tuple(circuit.to_ir().instructions)

    def source_leaf(bloq, inputs, desired):
        if isinstance(bloq, (XGate, ZGate)):
            opcode = "x" if isinstance(bloq, XGate) else "z"
            return (Instruction(opcode, inputs["q"]),), inputs
        return None

    original, ports = walk_source_graph(
        source, {"q": (0, 2, 3)}, {"q": (0, 2, 3)}, source_leaf
    )
    assert ports == {"q": (0, 2, 3)} and gates == original
    grid = torch.arange(1, 17, dtype=torch.float64)
    amplitudes = torch.complex(grid, 3 * grid - 5)
    amplitudes /= amplitudes.norm()
    initial, expected = (torch.zeros(16, dtype=torch.complex128) for _ in range(2))
    logical = torch.zeros((8, 2), dtype=torch.complex128)
    logical_expected = torch.zeros_like(logical)
    for word, spectator in product(range(8), (0, 1)):
        result, phase = evaluate_word(word, 3, inverse=inverse)
        before = ((word >> 2) << 3) | (spectator << 2) | (word & 3)
        after = ((result >> 2) << 3) | (spectator << 2) | (result & 3)
        initial[before] = logical[word, spectator] = amplitudes[(word << 1) | spectator]
        expected[after] = logical_expected[result, spectator] = phase * initial[before]
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        8, 8
    )
    torch.testing.assert_close(matrix @ logical, logical_expected, rtol=0, atol=2e-12)
    executed = fq.Circuit(4, inputs=initial, dtype=torch.complex128)
    executed.append(EachBitXZ(inverse=inverse))
    torch.testing.assert_close(executed.state().flatten(), expected, rtol=0, atol=2e-12)
    executed.append(EachBitXZ(inverse=not inverse))
    torch.testing.assert_close(executed.state().flatten(), initial, rtol=0, atol=2e-12)
    for legacy in (False, True):
        native = {}
        for kind in ("x", "z"):
            counts = asdict(
                query_qec_gate_cost(
                    OnEachCall(CliffordCall(kind, 1), 3), legacy_shims=legacy
                )
            )
            for name, value in counts.items():
                native[name] = native.get(name, 0) + value
        model = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
        assert {
            name: value for name, value in native.items() if value
        } == model.asdict()
        assert (
            model.clifford == len(gates) == 6
            and model.measurement == model.total_t_count() == 0
        )
        assert get_cost_value(source, QubitCount()) == 3
