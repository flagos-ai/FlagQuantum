"""Unmodified bit-array window action, shared codecs and staged lookup failures."""

from copy import deepcopy

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_montgomery_window import (
    montgomery_window_bits_classical_values,
)
from flagquantum.simulation.integer_encoding import integer_to_bits
from tests.unit.test_classical_product_accumulation import same_value

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CASES = [(2, 1, 3), (3, 1, 3), (4, 2, 5), (4, 4, 5), (5, 5, 13)]


def original(width, window, modulus):
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import SingleWindowModMul

    return SingleWindowModMul(window, width, modulus)


def compare_values(source, values, params):
    source_values = deepcopy(values)
    native_values = deepcopy(values)
    try:
        expected = source.on_classical_vals(**source_values)
    except Exception as error:
        with pytest.raises(type(error)):
            montgomery_window_bits_classical_values(
                native_values["x"],
                native_values["y"],
                native_values["target"],
                native_values["qrom_index"],
                *params,
            )
        return
    actual = montgomery_window_bits_classical_values(
        native_values["x"],
        native_values["y"],
        native_values["target"],
        native_values["qrom_index"],
        *params,
    )
    assert tuple(actual) == tuple(expected)
    for key in expected:
        same_value(actual[key], expected[key])
    assert actual["x"] is native_values["x"] and actual["y"] is native_values["y"]
    for key in source_values:
        same_value(native_values[key], source_values[key])


@pytest.mark.parametrize("params", CASES)
def test_original_all_controls_selected_operands_accumulators_and_dirty_index(params):
    width, window, modulus = params
    source = original(*params)
    for left in range(1 << window):
        for right in (0, 1, modulus - 1):
            for accumulator in (0, 1, (1 << (width + window)) - 1):
                for index in (0, 3):
                    values = dict(
                        x=np.asarray(
                            integer_to_bits(left, window)[::-1], dtype=np.uint8
                        ),
                        y=right,
                        target=np.asarray(
                            integer_to_bits(accumulator, width + window), dtype=np.uint8
                        ),
                        qrom_index=index,
                    )
                    compare_values(source, values, params)


@pytest.mark.parametrize(
    "kind",
    [
        "short_left",
        "empty_target",
        "nonbinary_target",
        "string_target",
        "bool_target",
        "nonbinary_control",
        "negative_y",
        "numpy_y",
        "numpy_dimensions",
        "symbolic_modulus",
        "even_modulus",
        "zero_modulus",
        "invalid_dimensions",
        "symbolic_width",
        "symbolic_literal_width",
        "negative_width",
        "zero_width",
    ],
)
def test_source_diagnostic_values_and_construction_failures(kind):
    params = (4, 2, 5)
    values = dict(x=[1, 0], y=2, target=[0, 0, 0, 0, 0, 1], qrom_index=7)
    if kind == "short_left":
        values["x"] = [1]
    elif kind == "empty_target":
        values["target"] = []
    elif kind == "nonbinary_target":
        values["target"] = [2]
    elif kind == "string_target":
        values["target"] = ["1", "0"]
    elif kind == "bool_target":
        values["target"] = [True, False]
    elif kind == "nonbinary_control":
        values["x"] = [2, -1]
    elif kind == "negative_y":
        values["y"] = -3
    elif kind == "numpy_y":
        values["y"] = np.uint8(250)
    elif kind == "numpy_dimensions":
        params = (np.int64(4), 2, 5)
    elif kind == "symbolic_modulus":
        params = (4, 2, sympy.Symbol("p"))
    elif kind == "even_modulus":
        params = (4, 2, 4)
    elif kind == "zero_modulus":
        params = (4, 2, 0)
    elif kind == "invalid_dimensions":
        params = (4, 3, 5)
    elif kind == "symbolic_width":
        params = (sympy.Symbol("n"), 2, 5)
    elif kind == "symbolic_literal_width":
        params = (sympy.Integer(4), 2, 5)
    elif kind == "negative_width":
        params = (-2, 2, 5)
    elif kind == "zero_width":
        params = (0, 2, 5)
    try:
        source = original(*params)
    except Exception as error:
        with pytest.raises(type(error)):
            montgomery_window_bits_classical_values(
                values["x"],
                values["y"],
                values["target"],
                values["qrom_index"],
                *params,
            )
    else:
        compare_values(source, values, params)


def test_codecs_and_table_are_actual_replaceable_dependencies(monkeypatch):
    import flagquantum.simulation.classical_montgomery_window as module

    calls = []
    decode, encode, table = (
        module.integer_from_bits,
        module.integer_to_bits,
        module._inversion_data,
    )

    def from_bits(bits, width):
        calls.append(("decode", width))
        return decode(bits, width)

    def data(window, modulus):
        calls.append(("table", window, modulus))
        np.testing.assert_array_equal(
            table(window, modulus), original(4, 2, 5).qrom.data[0]
        )
        return table(window, modulus)

    def to_bits(value, width):
        calls.append(("encode", width))
        return encode(value, width)

    monkeypatch.setattr(module, "integer_from_bits", from_bits)
    monkeypatch.setattr(module, "integer_to_bits", to_bits)
    monkeypatch.setattr(module, "_inversion_data", data)
    result = module.montgomery_window_bits_classical_values(
        [1, 0], 2, [0] * 6, 7, 4, 2, 5
    )
    assert calls == [("decode", 6), ("table", 2, 5), ("encode", 6)]
    assert result["qrom_index"] == 2  # Incoming 7 is overwritten, not XORed.
    monkeypatch.setattr(
        module, "_inversion_data", lambda w, p: np.zeros(1 << w, dtype=np.int64)
    )
    replaced = module.montgomery_window_bits_classical_values(
        [1, 0], 2, [0] * 6, 7, 4, 2, 5
    )
    assert replaced["target"] != result["target"]


def test_original_high_level_and_gate_overflow_witness_is_not_repaired():
    values = dict(
        x=np.array([1], dtype=np.uint8),
        y=2,
        target=np.array([1, 0, 1], dtype=np.uint8),
        qrom_index=0,
    )
    source = original(2, 1, 3)
    expected = source.on_classical_vals(**values)
    actual = montgomery_window_bits_classical_values(
        values["x"], values["y"], values["target"], 0, 2, 1, 3
    )
    assert expected["target"] == actual["target"] == [1, 0, 1]
    # The unchanged decomposed source yields [0,0,1]; the existing quantum
    # suite separately preserves that graph and its exact adjoint behavior.
    assert tuple(source.decompose_bloq().on_classical_vals(**values)["target"]) == (
        0,
        0,
        1,
    )
