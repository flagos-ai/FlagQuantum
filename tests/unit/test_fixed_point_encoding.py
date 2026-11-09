"""Stored fixed-point values follow unchanged raw integer and default array paths."""

from itertools import islice

import numpy as np
import pytest
import sympy

from flagquantum.simulation.fixed_point_encoding import (
    fixed_point_array_from_bits,
    fixed_point_array_to_bits,
    fixed_point_domain,
    fixed_point_from_bits,
    fixed_point_integer_width,
    fixed_point_real_value,
    fixed_point_to_bits,
    validate_fixed_point_array,
    validate_fixed_point_shape,
    validate_fixed_point_value,
)
from tests.unit.test_integer_encoding import compare_call

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
SHAPES = [
    (3, 2, False),
    (3, 2, True),
    (1, 1, False),
    (1, 0, True),
    (0, 0, False),
    (0, 0, True),
    (0, -1, True),
    (-1, -2, False),
    (3, 4, False),
    (3, 3, True),
    (np.int64(3), np.int64(2), False),
    (np.int64(3), np.int64(2), np.bool_(True)),
    (sympy.Integer(3), sympy.Integer(2), True),
    (sympy.Symbol("n", integer=True), 2, False),
    (3, sympy.Symbol("f", integer=True), True),
    (3, -2, True),
    (3.0, 2, False),
    (65, 2, False),
]


def source(width, fraction, signed):
    from qualtran._infra.data_types import _Fxp

    return _Fxp(width, fraction, signed)


@pytest.mark.parametrize("width,fraction,signed", SHAPES)
def test_shape_metadata_and_raw_domain_order(width, fraction, signed):
    compare_call(
        lambda _: source(width, fraction, signed).num_int,
        lambda _: fixed_point_integer_width(width, fraction, signed=signed),
        None,
    )
    compare_call(
        lambda _: tuple(islice(source(width, fraction, signed).get_domain(), 5)),
        lambda _: tuple(islice(fixed_point_domain(width, fraction, signed=signed), 5)),
        None,
    )
    compare_call(
        lambda _: source(width, fraction, signed) and None,
        lambda _: validate_fixed_point_shape(width, fraction, signed=signed),
        None,
    )


@pytest.mark.parametrize("width,fraction,signed", SHAPES)
@pytest.mark.parametrize("method", ["validate", "encode", "decode", "real"])
def test_scalar_raw_codec_types_bounds_and_unchecked_scale(
    width, fraction, signed, method
):
    native = {
        "validate": validate_fixed_point_value,
        "encode": fixed_point_to_bits,
        "decode": fixed_point_from_bits,
        "real": fixed_point_real_value,
    }[method]
    original_name = {
        "validate": "assert_valid_val",
        "encode": "to_bits",
        "decode": "from_bits",
        "real": "float_from_fixed_width_int",
    }[method]
    values = [-9, -1, 0, 1, 7, 8, 2**64, np.int8(-1), np.uint8(1), True, 1.5]
    if method == "decode":
        values = [[], [0], [1], [0, 1], [1, 0, 1], [1, 9], [True, False]]
    for value in values:
        compare_call(
            lambda x: getattr(source(width, fraction, signed), original_name)(x),
            lambda x: native(x, width, fraction, signed=signed),
            value,
        )


@pytest.mark.parametrize("width,fraction,signed", SHAPES)
@pytest.mark.parametrize("method", ["validate", "encode", "decode"])
@pytest.mark.parametrize(
    "values",
    [
        np.array(1),
        np.array([-1, 0, 1]),
        np.array([0.0, 1.0]),
        np.array([[1, 0, 1], [1, 1, 1]], dtype=np.uint8),
        np.empty((0, 3), dtype=np.int64),
        np.array([0, 2**64], dtype=object),
    ],
)
def test_default_array_codec_preserves_scalar_rules_and_inference(
    width, fraction, signed, method, values
):
    native = {
        "validate": validate_fixed_point_array,
        "encode": fixed_point_array_to_bits,
        "decode": fixed_point_array_from_bits,
    }[method]
    original_name = {
        "validate": "assert_valid_val_array",
        "encode": "to_bits_array",
        "decode": "from_bits_array",
    }[method]
    compare_call(
        lambda x: getattr(source(width, fraction, signed), original_name)(x),
        lambda x: native(x, width, fraction, signed=signed),
        values,
    )


@pytest.mark.parametrize("name", ["QFxp", "CFxp"])
@pytest.mark.parametrize("signed", [False, True])
def test_public_source_types_share_the_same_stored_integer_path(name, signed):
    import qualtran

    dtype = getattr(qualtran, name)(3, 2, signed)
    for value in (-4, -1, 0, 1, 3, 7):
        compare_call(
            dtype.to_bits, lambda x: fixed_point_to_bits(x, 3, 2, signed=signed), value
        )
    values = np.array([-1, 0, 1])
    compare_call(
        dtype.to_bits_array,
        lambda x: fixed_point_array_to_bits(x, 3, 2, signed=signed),
        values,
    )


def test_unsigned_fixed_point_default_is_not_unsigned_mask_encoding():
    from qualtran import QFxp, QUInt

    from flagquantum.simulation.integer_encoding import integer_array_to_bits

    values = np.array([8], dtype=np.uint8)
    np.testing.assert_array_equal(QUInt(3).to_bits_array(values), [[0, 0, 0]])
    np.testing.assert_array_equal(integer_array_to_bits(values, 3), [[0, 0, 0]])
    with pytest.raises(ValueError):
        QFxp(3, 2).to_bits_array(values)
    with pytest.raises(ValueError):
        fixed_point_array_to_bits(values, 3, 2)
    with pytest.raises(ValueError):
        fixed_point_array_to_bits(np.array([0.0, 1.0]), 3, 2)
    assert fixed_point_real_value(99, 3, 2) == 24.75  # No raw range validation.


def test_shape_construction_precedes_lazy_raw_domain_arithmetic():
    with pytest.raises(ValueError):
        source(1, 0, True).get_domain()
    with pytest.raises(ValueError):
        fixed_point_domain(1, 0, signed=True)
    original = source(0, -1, True).get_domain()
    native = fixed_point_domain(0, -1, signed=True)
    assert type(native) is type(original)
    with pytest.raises(ValueError):
        next(original)
    with pytest.raises(ValueError):
        next(native)


def test_array_and_integer_implementations_can_be_replaced_without_consumer_changes(
    monkeypatch,
):
    import flagquantum.simulation.fixed_point_encoding as native

    calls = []

    def encode(value, width, *, signed):
        calls.append(("encode", value, width, signed))
        return [1, 0, 1]

    def decode(bits, width, *, signed):
        calls.append(("decode", tuple(bits), width, signed))
        return -3

    monkeypatch.setattr(native, "integer_to_bits", encode)
    monkeypatch.setattr(native, "integer_from_bits", decode)
    np.testing.assert_array_equal(
        native.fixed_point_array_to_bits(np.array([0]), 3, 2), [[1, 0, 1]]
    )
    np.testing.assert_array_equal(
        native.fixed_point_array_from_bits(np.array([[0, 0, 0]]), 3, 2), [-3]
    )
    assert calls == [("encode", 0, 3, False), ("decode", (0, 0, 0), 3, False)]


def test_raw_values_full_complex_phase_gradient_and_no_work():
    import torch
    from qualtran.bloqs.rotations.phase_gradient import PhaseGradientUnitary
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    import flagquantum as fq
    from flagquantum.compiler._component_trace.resource_comparison import (
        count_emitted_instructions,
    )
    from flagquantum.compiler.registers.phase.phase_gradient import (
        phase_gradient_unitary_ir,
    )
    from flagquantum.core.ir import CircuitIR

    weights = torch.arange(1, 17, dtype=torch.float64)
    amplitudes = torch.complex(weights, 3 * weights - 5)
    amplitudes /= amplitudes.norm()
    words = tuple(fixed_point_domain(3, 3))
    assert words == tuple(range(8))
    for word in words:
        assert fixed_point_from_bits(fixed_point_to_bits(word, 3, 3), 3, 3) == word
    source_gate = PhaseGradientUnitary(3)
    source_matrix = torch.as_tensor(source_gate.tensor_contract().copy())
    real_values = torch.tensor([fixed_point_real_value(word, 3, 3) for word in words])
    expected = (
        amplitudes.reshape(8, 2)
        * torch.exp(2j * torch.pi * real_values.to(torch.float64))[:, None]
    )
    torch.testing.assert_close(
        source_matrix @ amplitudes.reshape(8, 2), expected, rtol=0, atol=2e-12
    )
    gates = phase_gradient_unitary_ir((0, 1, 2))
    circuit = fq.Circuit(4, inputs=amplitudes, dtype=torch.complex128).compose(
        CircuitIR(4, gates)
    )
    torch.testing.assert_close(
        circuit.state().reshape(8, 2), expected, rtol=0, atol=2e-12
    )
    inverse = phase_gradient_unitary_ir((0, 1, 2), exponent=-1)
    circuit.compose(CircuitIR(4, inverse))
    torch.testing.assert_close(
        circuit.state().flatten(), amplitudes, rtol=0, atol=2e-12
    )
    emitted = count_emitted_instructions(CircuitIR(4, gates))
    assert emitted.unconditional == {"phase": 3} and emitted.t_count == 0
    model = get_cost_value(source_gate, QECGatesCost())
    assert model.clifford == 2 and model.t == 1 and model.measurement == 0


@pytest.mark.parametrize("backend", ["source", "native"])
def test_shared_core_shape_guard_can_replace_real_numeric_consumer(
    backend, monkeypatch
):
    import flagquantum.simulation.fixed_point_encoding as native

    original_guard = native.validate_fixed_point_parameters
    events = []

    def guard(width, fractional_width, *, signed=False):
        events.append((width, fractional_width, signed))
        if backend == "source":
            source(width, fractional_width, signed)
        else:
            original_guard(width, fractional_width, signed=signed)

    monkeypatch.setattr(native, "validate_fixed_point_parameters", guard)
    values = np.array([-4, -1, 0, 3])
    packed = native.fixed_point_array_to_bits(values, 3, 2, signed=True)
    np.testing.assert_array_equal(packed, source(3, 2, True).to_bits_array(values))
    np.testing.assert_array_equal(
        native.fixed_point_array_from_bits(packed, 3, 2, signed=True), values
    )
    assert native.fixed_point_real_value(-1, 3, 2, signed=True) == -0.25
    assert tuple(native.fixed_point_domain(3, 2, signed=True)) == tuple(range(-4, 4))
    assert events == [(3, 2, True)] * 4
    with pytest.raises(ValueError, match="bitsize must be > 1"):
        native.fixed_point_to_bits(0, 1, 0, signed=True)
