"""XOR predicates retain original raw numeric, NumPy and symbolic behavior."""

from copy import deepcopy
from itertools import product

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_comparison import (
    equal_classical_values,
    less_equal_classical_values,
    less_than_constant_classical_values,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
PAIRS = [
    (0, 0),
    (0, 1),
    (3, 2),
    (-1, 2),
    (2**100, 2**90),
    (np.int8(-1), np.int8(2)),
    (np.uint8(255), np.uint8(1)),
    (np.array([1]), np.array([2])),
    (np.array([0, 2]), np.array([1, 1])),
    (np.array([[0], [2]]), np.array([1, 2])),
    (sympy.Integer(1), sympy.Integer(2)),
    (sympy.Symbol("x"), 0),
    (1.5, 2.0),
    (float("nan"), 1),
    (np.bool_(True), np.bool_(False)),
]
TARGETS = [0, 1, 2, -1, True, np.uint8(1), np.bool_(True), np.array([0, 1]), 1.0, None]


def same_value(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_equal(actual, expected)


def compare_calls(original, native, values):
    original_inputs, native_inputs = deepcopy(values), deepcopy(values)
    try:
        expected = original(*original_inputs)
    except Exception as error:
        with pytest.raises(type(error)):
            native(*native_inputs)
    else:
        actual = native(*native_inputs)
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            same_value(a, e)
    for a, e in zip(native_inputs, original_inputs, strict=True):
        same_value(a, e)


def source_call(kind, width=3):
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.comparison import (
        Equals,
        LessThanConstant,
        LessThanEqual,
    )

    if kind == "constant":
        return lambda x, y, target: tuple(
            LessThanConstant(width, y).on_classical_vals(x=x, target=target).values()
        )
    if kind == "less_equal":
        return lambda x, y, target: tuple(
            LessThanEqual(width, width)
            .on_classical_vals(x=x, y=y, target=target)
            .values()
        )
    return lambda x, y, target: tuple(
        Equals(QUInt(width)).on_classical_vals(x, y, target).values()
    )


def native_call(kind):
    if kind == "constant":
        return lambda x, y, target: less_than_constant_classical_values(x, target, y)
    return {"less_equal": less_equal_classical_values, "equal": equal_classical_values}[
        kind
    ]


@pytest.mark.parametrize("kind", ["constant", "less_equal", "equal"])
@pytest.mark.parametrize("left,right", PAIRS)
@pytest.mark.parametrize("target", TARGETS)
def test_original_predicate_xor_values_types_broadcasts_and_failures(
    kind, left, right, target
):
    compare_calls(source_call(kind), native_call(kind), (left, right, target))


@pytest.mark.parametrize("kind", ["constant", "less_equal", "equal"])
@pytest.mark.parametrize(
    "width", [0, -1, 3, np.int64(3), sympy.Integer(3), sympy.Symbol("n"), 3.0]
)
def test_source_unused_metadata_does_not_add_numerical_width_checks(kind, width):
    compare_calls(source_call(kind, width), native_call(kind), (3, 2, 1))


@pytest.mark.parametrize("name", ["QUInt", "QInt", "QMontgomeryUInt", "QAny"])
def test_equality_dtype_does_not_reinterpret_raw_values(name):
    import qualtran
    from qualtran.bloqs.arithmetic.comparison import Equals

    original = Equals(getattr(qualtran, name)(3))
    for left, right in ((-1, 7), (-1, -1), (3, 3)):
        compare_calls(
            lambda x, y, t: tuple(original.on_classical_vals(x, y, t).values()),
            equal_classical_values,
            (left, right, 1),
        )


def test_original_operands_retain_identity_and_arrays_are_not_mutated():
    left, right, target = np.array([0, 1]), np.array([1, 1]), np.array([0, 1])
    for function in (less_equal_classical_values, equal_classical_values):
        result = function(left, right, target)
        assert result[0] is left and result[1] is right
    result = less_than_constant_classical_values(left, target, 1)
    assert result[0] is left
    np.testing.assert_array_equal(left, [0, 1])
    np.testing.assert_array_equal(right, [1, 1])
    np.testing.assert_array_equal(target, [0, 1])


@pytest.mark.parametrize("kind", ["constant", "less_equal", "equal"])
def test_native_values_connect_to_original_stream_full_complex_branches_and_clean_work(
    kind,
):
    import torch
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.comparison import (
        Equals,
        LessThanConstant,
        LessThanEqual,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.compiler.registers.comparison.equality import equality_ir
    from flagquantum.compiler.registers.comparison.less_equal import (
        less_equal_ir,
        less_equal_requirements,
    )
    from flagquantum.compiler.registers.comparison.less_than import (
        less_than_constant_ir,
        less_than_constant_workspace_size,
    )
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    left = (0, 1)
    if kind == "constant":
        data_bits, target = 3, 2
        work = tuple(range(3, 3 + less_than_constant_workspace_size(2, 3)))
        slots = (40, 41)
        gates = less_than_constant_ir(left, 3, target, work, slots)
        original = LessThanConstant(2, 3)
        ports = {"x": left, "target": (target,)}
    else:
        right, target, data_bits = (2, 3), 4, 5
        count = less_equal_requirements(2, 2)[0] if kind == "less_equal" else 1
        work = tuple(range(5, 5 + count))
        slots = tuple(
            range(
                40,
                40 + (less_equal_requirements(2, 2)[1] if kind == "less_equal" else 1),
            )
        )
        gates = (less_equal_ir if kind == "less_equal" else equality_ir)(
            left, right, target, work, slots
        )
        original = LessThanEqual(2, 2) if kind == "less_equal" else Equals(QUInt(2))
        ports = {"x": left, "y": right, "target": (target,)}
    measured = {
        gate.wires[0]: gate.metadata["classical_bit"]
        for gate in gates
        if gate.name == "measure"
    }
    expected_gates = reference_arithmetic_gates(
        original, ports, work, measured, cirq_style=kind != "equal"
    )
    assert_source(gates, expected_gates)
    total = data_bits + len(work) + 1
    grid = torch.arange(1, (1 << (data_bits + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(grid, 3 * grid - 5)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for data, spectator in product(range(1 << data_bits), (0, 1)):
        old_target = data & 1
        a = data >> 1 if kind == "constant" else data >> 3
        b = 3 if kind == "constant" else (data >> 1) & 3
        updated = native_call(kind)(a, b, old_target)[-1]
        output = (data & ~1) | updated
        initial[(data << (len(work) + 1)) | spectator] = amplitudes[
            2 * data + spectator
        ]
        expected[(output << (len(work) + 1)) | spectator] = amplitudes[
            2 * data + spectator
        ]
    measurements = len(measured)
    for history in product((0, 1), repeat=measurements):
        actual = forced_branch(initial, gates, history)
        torch.testing.assert_close(
            actual, expected / 2 ** (measurements / 2), rtol=0, atol=2e-12
        )
        torch.testing.assert_close(
            actual, forced_branch(initial, expected_gates, history), rtol=0, atol=2e-12
        )
        assert (
            torch.count_nonzero(
                actual.reshape(1 << data_bits, 1 << len(work), 2)[:, 1:, :]
            )
            == 0
        )
    counts = {
        name: sum(g.name == name for g in gates) for name in ("t", "tdg", "measure")
    }
    model = get_cost_value(original, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count()
    assert counts["measure"] == model.measurement
