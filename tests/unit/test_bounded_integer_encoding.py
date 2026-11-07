"""Bounded codecs and their existing measured SELECT consumer match pinned source."""

from collections import Counter
from itertools import product

import numpy as np
import pytest
import sympy

from flagquantum.simulation.bounded_integer_encoding import (
    bounded_integer_array_from_bits,
    bounded_integer_array_to_bits,
    bounded_integer_from_bits,
    bounded_integer_to_bits,
)
from flagquantum.simulation.bounded_integer_values import bounded_integer_domain
from tests.unit.test_bounded_integer_values import SHAPES, source
from tests.unit.test_integer_encoding import compare_call

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width,bound", SHAPES)
@pytest.mark.parametrize("method", ["encode", "decode"])
def test_scalar_order_unsigned_checks_and_postdecode_bound(width, bound, method):
    native = (
        bounded_integer_to_bits if method == "encode" else bounded_integer_from_bits
    )
    original = "to_bits" if method == "encode" else "from_bits"
    values = [-1, 0, 1, 3, 7, 8, 9, np.int8(1), True, np.bool_(True), 1.0]
    if method == "decode":
        values = [[], [0], [1], [0, 1], [1, 0, 1], [1, 0, 0, 0], [1, 9], [True, False]]
    for value in values:
        compare_call(
            lambda x: getattr(source(width, bound), original)(x),
            lambda x: native(x, width, bound),
            value,
        )


@pytest.mark.parametrize("width,bound", SHAPES)
@pytest.mark.parametrize("method", ["encode", "decode"])
@pytest.mark.parametrize(
    "values",
    [
        np.array(0),
        np.array([0, 1]),
        np.array([-1, 0]),
        np.array([8], dtype=np.uint8),
        np.array([0.0, 1.0]),
        np.array([[1, 0, 1], [1, 1, 1]], dtype=np.uint8),
        np.empty((0, 3), dtype=np.int64),
    ],
)
def test_default_array_vectorization_preserves_each_scalar_boundary(
    width, bound, method, values
):
    native = (
        bounded_integer_array_to_bits
        if method == "encode"
        else bounded_integer_array_from_bits
    )
    original = "to_bits_array" if method == "encode" else "from_bits_array"
    compare_call(
        lambda x: getattr(source(width, bound), original)(x),
        lambda x: native(x, width, bound),
        values,
    )


@pytest.mark.parametrize("name", ["BQUInt", "BCUInt"])
def test_public_quantum_and_classical_sources_share_bounded_codec(name):
    import qualtran

    dtype = getattr(qualtran, name)(2, 3)
    for value in (0, 1, 2, 3):
        compare_call(dtype.to_bits, lambda x: bounded_integer_to_bits(x, 2, 3), value)
        compare_call(
            dtype.from_bits,
            lambda x: bounded_integer_from_bits(x, 2, 3),
            [value // 2, value % 2],
        )
    compare_call(
        dtype.to_bits_array,
        lambda x: bounded_integer_array_to_bits(x, 2, 3),
        np.array([0, 1, 2]),
    )


def test_symbolic_literal_bound_does_not_add_a_decoder_width_guard():
    # Symbolic metadata bypasses construction capacity checks. The raw unsigned
    # decoder ignores width; the bounded decoder only checks its decoded value.
    assert bounded_integer_from_bits([1, 0, 0, 0], 3, sympy.Integer(9)) == 8
    with pytest.raises(ValueError):
        bounded_integer_to_bits(8, 3, sympy.Integer(9))
    with pytest.raises(ValueError):
        bounded_integer_array_to_bits(np.array([8], dtype=np.uint8), 3, 8)


def test_bounded_array_calls_shared_scalar_owners_and_default_vectorizer(monkeypatch):
    import flagquantum.simulation.bounded_integer_encoding as native

    calls = []

    def encode(value, width):
        calls.append(("encode", value, width))
        return [1, 0]

    def decode(bits, width):
        calls.append(("decode", tuple(bits), width))
        return 2

    monkeypatch.setattr(native, "integer_to_bits", encode)
    monkeypatch.setattr(native, "integer_from_bits", decode)
    np.testing.assert_array_equal(
        native.bounded_integer_array_to_bits(np.array([1]), 2, 3), [[1, 0]]
    )
    np.testing.assert_array_equal(
        native.bounded_integer_array_from_bits(np.array([[0, 1]]), 2, 3), [2]
    )
    assert calls == [("encode", 1, 2), ("decode", (0, 1), 2)]
    # A replacement decoder must still pass the bounded consumer's own checks.
    monkeypatch.setattr(native, "integer_from_bits", lambda bits, width: 3)
    with pytest.raises(ValueError, match="Too-large"):
        native.bounded_integer_from_bits([0, 0], 2, 3)


def test_valid_bounded_addresses_source_stream_all_histories_and_complex_cleanup():
    import cirq
    import torch
    from qualtran.bloqs.multiplexers.select_pauli_lcu import (
        SelectPauliLCU as SourceSelect,
    )

    from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import (
        SelectPauliLCU,
    )
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.qec.resources.estimation import estimate_ftqc_resources
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_first_quantization_resource_model import assert_source_model
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_relative_controlled_swap import assert_source_dependencies

    words = ("X", "Y", "Z")
    op = SelectPauliLCU(2, 1, words).make_bounded_on(
        selection=(0, 1), target=(2,), workspace=(3,), classical_bits=(7,)
    )
    circuit = DynamicCircuit(5).append(op)
    gates = circuit.to_ir().instructions
    original = SourceSelect(2, 1, tuple(cirq.DensePauliString(word) for word in words))
    source_gates = reference_arithmetic_gates(
        original,
        {"selection": (0, 1), "target": (2,)},
        (3,),
        {3: 7},
        cirq_style=True,
        keep_pauli_leaves=True,
    )
    assert_source_dependencies(gates, source_gates)
    grid = torch.arange(1, 13, dtype=torch.float64)
    weights = torch.complex(grid, 3 * grid - 5)
    weights /= weights.norm()
    initial = torch.zeros(32, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for address in bounded_integer_domain(2, 3):
        encoded = bounded_integer_to_bits(address, 2, 3)
        stored = bounded_integer_from_bits(encoded, 2, 3)
        assert stored == address
        for target, spectator in product((0, 1), repeat=2):
            amplitude = weights[4 * address + 2 * target + spectator]
            initial[(stored << 3) | (target << 2) | spectator] = amplitude
            output = target ^ (address != 2)
            phase = (
                (1j if target == 0 else -1j)
                if address == 1
                else (-1 if address == 2 and target else 1)
            )
            expected[(stored << 3) | (output << 2) | spectator] = phase * amplitude
    measurements = Counter(gate.name for gate in gates)["measure"]
    for history in product((0, 1), repeat=measurements):
        native_state = forced_branch(initial, gates, history)
        source_state = forced_branch(initial, source_gates, history)
        torch.testing.assert_close(native_state, source_state, rtol=0, atol=2e-12)
        torch.testing.assert_close(
            native_state, expected / 2 ** (measurements / 2), rtol=0, atol=2e-12
        )
        assert torch.count_nonzero(native_state.reshape(4, 2, 2, 2)[:, :, 1, :]) == 0
    assert_source_model(op.resource_model(), original)
    report = estimate_ftqc_resources(op, emitted_ir=circuit.to_ir())
    assert report.emitted is not None
    assert report.emitted.t_count == report.logical.gates.total_t_count()
    assert report.emitted.unconditional["measure"] == report.logical.gates.measurement
