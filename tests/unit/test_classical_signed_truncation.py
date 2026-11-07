"""Original signed truncation casts, prefix checks and shared codec ownership."""

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_signed_truncation import (
    signed_truncate_classical_value,
)
from tests.unit.test_integer_encoding import compare_call

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
WIDTHS = [
    (4, 2),
    (3, 1),
    (2, 1),
    (3, 2),
    (4, 4),
    (2, 4),
    (0, -1),
    (0, 0),
    (-1, -2),
    (4, 0),
    (4, -1),
    (np.int64(4), np.int64(2)),
    (sympy.Integer(4), 2),
    (4, sympy.Integer(2)),
    (4, sympy.Integer(1)),
    (3, sympy.Symbol("n")),
    (sympy.Symbol("n"), 2),
    (3, sympy.Integer(4)),
]


def source(input_width, output_width, ones):
    from qualtran import QInt, QIntOnesComp
    from qualtran.bloqs.arithmetic.conversions.sign_extension import SignTruncate

    dtype = QIntOnesComp if ones else QInt
    return SignTruncate(dtype(input_width), dtype(output_width))


@pytest.mark.parametrize("ones", [False, True])
@pytest.mark.parametrize("input_width,output_width", WIDTHS)
@pytest.mark.parametrize(
    "value",
    [
        -9,
        -8,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        7,
        8,
        2**70,
        np.int8(-1),
        np.uint8(255),
        True,
        np.bool_(True),
        1.9,
        float("nan"),
        sympy.Integer(-1),
        sympy.Symbol("x"),
        np.array([1]),
        np.array([1, 2]),
    ],
)
def test_original_constructor_value_conversion_and_prefix_failures(
    input_width, output_width, ones, value
):
    compare_call(
        lambda x: source(input_width, output_width, ones).on_classical_vals(x)["y"],
        lambda x: signed_truncate_classical_value(
            x, input_width, output_width, ones_complement=ones
        ),
        value,
    )


@pytest.mark.parametrize("ones", [False, True])
def test_metadata_failure_precedes_value_integer_conversion(ones):
    class Value:
        def __int__(self):
            pytest.fail("Invalid construction must fail before converting the value")

    for input_width, output_width in ((4, 4), (2, 4)):
        with pytest.raises(ValueError):
            source(input_width, output_width, ones).on_classical_vals(Value())
        with pytest.raises(ValueError):
            signed_truncate_classical_value(
                Value(), input_width, output_width, ones_complement=ones
            )
    if ones:
        with pytest.raises(ValueError):
            source(4, 1, True).on_classical_vals(Value())
        with pytest.raises(ValueError):
            signed_truncate_classical_value(Value(), 4, 1, ones_complement=True)


@pytest.mark.parametrize("ones", [False, True])
def test_codec_replacement_preserves_value_cast_and_unconverted_width(
    ones, monkeypatch
):
    import flagquantum.simulation.classical_signed_truncation as native

    calls = []

    def encode(value, width, **kwargs):
        calls.append(("encode", value, width, kwargs))
        assert type(value) is int
        return [1, 1, 1, 0]

    def decode(bits, width, **kwargs):
        calls.append(("decode", bits, width, kwargs))
        assert type(width) is sympy.Integer
        return 9

    prefix = "ones_complement_" if ones else "integer_"
    monkeypatch.setattr(native, prefix + "to_bits", encode)
    monkeypatch.setattr(native, prefix + "from_bits", decode)
    assert (
        native.signed_truncate_classical_value(
            np.int8(-2), 4, sympy.Integer(2), ones_complement=ones
        )
        == 9
    )
    options = {} if ones else {"signed": True}
    assert calls == [
        ("encode", -2, 4, options),
        ("decode", [1, 0], sympy.Integer(2), options),
    ]


@pytest.mark.parametrize("ones", [False, True])
def test_prefix_mismatch_fails_before_output_decoder(ones, monkeypatch):
    import flagquantum.simulation.classical_signed_truncation as native

    def unexpected(*args, **kwargs):
        pytest.fail("A nonredundant sign prefix must not be decoded")

    monkeypatch.setattr(
        native, "ones_complement_from_bits" if ones else "integer_from_bits", unexpected
    )
    with pytest.raises(ValueError, match="prefix"):
        native.signed_truncate_classical_value(3, 4, 2, ones_complement=ones)


@pytest.mark.parametrize("ones", [False, True])
def test_all_raw_sign_words_source_stream_spectator_and_cleared_prefix(ones):
    from collections import Counter

    import torch
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    import flagquantum as fq
    from flagquantum.compiler.registers.sign_extension import sign_extend_ir
    from flagquantum.core.ir import CircuitIR
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    original = source(4, 2, ones)
    gates = sign_extend_ir((2, 3), (0, 1))
    expected_gates = reference_arithmetic_gates(
        original, {"x": (0, 1, 2, 3)}, (), {}, expected_outputs={"y": (2, 3)}
    )
    assert_source(gates, expected_gates)
    grid = torch.arange(1, 9, dtype=torch.float64)
    weights = torch.complex(grid, 3 * grid - 5)
    weights /= weights.norm()
    initial = torch.zeros(32, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for word in range(4):
        prefix = 3 if word & 2 else 0
        for spectator in (0, 1):
            initial[(((prefix << 2) | word) << 1) | spectator] = weights[
                2 * word + spectator
            ]
            expected[(word << 1) | spectator] = weights[2 * word + spectator]
    circuit = fq.Circuit(5, inputs=initial, dtype=torch.complex128).compose(
        CircuitIR(5, gates)
    )
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)
    assert torch.count_nonzero(circuit.state().flatten().reshape(4, 8)[1:]) == 0
    circuit.compose(CircuitIR(5, gates))
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=2e-12)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(original, QECGatesCost())
    assert counts == {"cx": 3}
    assert Counter(g.name for g in expected_gates) == counts
    # The source QEC classifier treats the fanout component as one Clifford;
    # both emitted primitive streams contain the same three CNOT instructions.
    assert model.clifford == 1 and model.total_t_count() == model.measurement == 0
    if ones:
        # Both zero encodings are present coherently; neither raw word is
        # replaced by the canonical scalar-zero encoding in the gate path.
        assert (
            original.on_classical_vals(0)["y"]
            == signed_truncate_classical_value(0, 4, 2, ones_complement=True)
            == 0
        )
