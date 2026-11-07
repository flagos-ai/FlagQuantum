"""Original simulator initialization, range-only arrays and row-major bindings."""

from itertools import product
from types import MappingProxyType

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_registers import bind_integer_register_values

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
n = sympy.Symbol("n", positive=True, integer=True)


def original_bind(name, values, width, shape, signed):
    import networkx as nx
    from qualtran import QInt, QUInt, Register, Signature
    from qualtran.simulation.classical_sim import ClassicalSimState

    # Initialization is the actual source binding path. No execution is claimed
    # for this empty instruction graph.
    signature = Signature(
        [Register(name, (QInt if signed else QUInt)(width), shape=shape)]
    )
    state = ClassicalSimState(signature, nx.DiGraph(), values)
    return {soq.idx: value for soq, value in state.soq_assign.items()}


def compare_binding(values, width, shape, signed, name="x"):
    try:
        expected = original_bind(name, values, width, shape, signed)
    except (ValueError, TypeError, OverflowError) as exc:
        with pytest.raises(type(exc)) as actual:
            bind_integer_register_values(
                name, values, width, shape=shape, signed=signed
            )
        assert type(actual.value.__cause__) is type(exc.__cause__)
    else:
        actual = bind_integer_register_values(
            name, values, width, shape=shape, signed=signed
        )
        assert tuple(actual) == tuple(expected)
        for idx in expected:
            assert type(actual[idx]) is type(expected[idx])
            if expected[idx] is np.ma.masked:
                assert actual[idx] is np.ma.masked
            else:
                np.testing.assert_equal(actual[idx], expected[idx])
        return actual


def fixture(kind):
    if kind == "scalar":
        return {"x": np.int64(1)}, ()
    if kind == "zero_dim_array":
        return {"x": np.array(1)}, ()
    if kind == "scalar_float":
        return {"x": 1.5}, ()
    if kind == "scalar_symbol":
        return {"x": n}, ()
    if kind == "scalar_symbolic_literal":
        return {"x": sympy.Integer(99)}, ()
    if kind == "list":
        return {"x": [[0, 1], [2, 3]]}, (2, 2)
    if kind == "strided":
        return {"x": np.arange(8, dtype=np.int64).reshape(2, 4)[:, ::2]}, (2, 2)
    if kind == "masked":
        return {
            "x": np.ma.array([[0, 99], [2, 3]], mask=[[False, True], [False, False]])
        }, (2, 2)
    if kind == "float_nan":
        return {"x": np.array([[1.5, np.nan]])}, (1, 2)
    if kind == "symbolic_elements":
        return {"x": np.array([sympy.Integer(0), sympy.Integer(1)], dtype=object)}, (2,)
    if kind == "bad_range":
        return {"x": np.array([0, 99])}, (2,)
    if kind == "bad_shape":
        return {"x": np.array([0, 1])}, (2, 2)
    if kind == "empty":
        return {"x": np.empty((2, 0), dtype=np.int64)}, (2, 0)
    if kind == "missing":
        return {"other": 1}, ()
    raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind,signed,width",
    tuple(
        product(
            (
                "scalar",
                "zero_dim_array",
                "scalar_float",
                "scalar_symbol",
                "scalar_symbolic_literal",
                "list",
                "strided",
                "masked",
                "float_nan",
                "symbolic_elements",
                "bad_range",
                "bad_shape",
                "empty",
                "missing",
            ),
            (False, True),
            (0, 3, sympy.Integer(3), n),
        )
    ),
)
def test_source_initialization_values_types_order_and_exception_causes(
    kind, signed, width
):
    values, shape = fixture(kind)
    compare_binding(MappingProxyType(values), width, shape, signed)


@pytest.mark.parametrize("shape", ((sympy.Integer(2),), (n,), (-1,)))
def test_symbolic_or_invalid_shape_cannot_be_enumerated(shape):
    compare_binding({"x": [0, 1]}, 3, shape, False)


def test_binding_keeps_scalar_objects_extra_fields_and_masked_array_subclass():
    symbol = sympy.Rational(1, 2)
    assert (
        bind_integer_register_values("", {"": symbol, "unused": object()}, 1)[()]
        is symbol
    )
    values, _ = fixture("masked")
    before = values["x"].copy()
    actual = compare_binding(values, 3, (2, 2), False)
    assert actual[(0, 1)] is np.ma.masked
    np.testing.assert_equal(values["x"], before)
    assert tuple(actual) == ((0, 0), (0, 1), (1, 0), (1, 1))


def test_same_float_value_is_rejected_in_scalar_but_retained_in_array_protocol():
    with pytest.raises(ValueError):
        bind_integer_register_values("x", {"x": 1.5}, 3)
    bound = bind_integer_register_values("x", {"x": [1.5, np.nan]}, 3, shape=(2,))
    assert bound[(0,)] == 1.5 and np.isnan(bound[(1,)])


def test_bound_two_dimensional_words_keep_all_complex_amplitudes_and_source_xor_values():
    import torch
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.bitwise import XorK
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.circuit import Circuit
    from flagquantum.compiler.registers.bitwise import xor_constant_ir
    from flagquantum.simulation.classical_bitwise import xor_constant_classical_value

    source = XorK(QUInt(2), 1)
    data = torch.arange(512, dtype=torch.int64)
    coeff = data.to(torch.float64)
    amplitudes = torch.complex(coeff + 1, 3 * coeff - 5)
    amplitudes /= amplitudes.norm()
    expected = torch.zeros_like(amplitudes)
    for label in data.tolist():
        word = label >> 1
        values = np.array([(word >> shift) & 3 for shift in (6, 4, 2, 0)]).reshape(2, 2)
        native = bind_integer_register_values("x", {"x": values}, 2, shape=(2, 2))
        original = original_bind("x", {"x": values}, 2, (2, 2), False)
        mapped = 0
        for idx in native:
            assert native[idx] == original[idx]
            result = xor_constant_classical_value(native[idx], 1)
            assert result == source.on_classical_vals(x=original[idx])["x"]
            mapped = (mapped << 2) | int(result)
        expected[(mapped << 1) | (label & 1)] = amplitudes[label]
    gates = tuple(
        g for start in (0, 2, 4, 6) for g in xor_constant_ir((start, start + 1), 1)
    )
    assert len(gates) == 4 and all(g.name == "x" for g in gates)
    cost = get_cost_value(source, QECGatesCost())
    assert cost.clifford * 4 == 4 and cost.total_t_count() == cost.measurement == 0
    circuit = Circuit(9, inputs=amplitudes, dtype=torch.complex128).append(
        FieldOperation(9, gates)
    )
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        circuit.append(FieldOperation(9, gates)).state().flatten(),
        amplitudes,
        rtol=0,
        atol=1e-12,
    )
