"""Original integer conversions, source classical ports and coherent gate checks."""

from itertools import product

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.arithmetic import add_ir
from flagquantum.compiler.registers.out_of_place_addition import out_of_place_add_ir
from flagquantum.simulation.classical_addition import (
    add_constant_classical_value,
    add_integer_values,
    out_of_place_sum_classical_values,
    register_add_classical_values,
)
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def equal(actual, expected):
    assert type(actual) is type(expected)
    np.testing.assert_array_equal(actual, expected)
    if isinstance(actual, np.ndarray):
        assert actual.dtype == expected.dtype


@pytest.mark.parametrize(
    "width,signed,kind",
    tuple(
        product(
            (None, 0, 1, 3, 8),
            (False, True),
            ("int", "int8", "uint8", "int64", "array"),
        )
    ),
)
def test_original_shared_add_ints_types_and_overflow(width, signed, kind):
    from qualtran.simulation.classical_sim import add_ints

    if kind == "array":
        left = np.array([0, 1, 7], dtype=np.int64)
        right = np.array([1, 3, 4], dtype=np.int64)
    else:
        cls = int if kind == "int" else getattr(np, kind)
        left, right = cls(7), cls(3)
    try:
        expected = add_ints(left, right, num_bits=width, is_signed=signed)
    except (ValueError, OverflowError, TypeError) as exc:
        with pytest.raises(type(exc)):
            add_integer_values(left, right, num_bits=width, signed=signed)
    else:
        equal(add_integer_values(left, right, num_bits=width, signed=signed), expected)


@pytest.mark.parametrize(
    "name,width", tuple(product(("QUInt", "QInt", "QMontgomeryUInt"), (1, 2, 3)))
)
def test_all_legal_classical_add_and_constant_values(name, width):
    import qualtran
    from qualtran.bloqs.arithmetic.addition import Add, AddK

    dt = getattr(qualtran, name)(width)
    signed = name == "QInt"
    domain = (
        range(-(1 << (width - 1)), 1 << (width - 1)) if signed else range(1 << width)
    )
    for left, right in product(domain, repeat=2):
        expected = Add(dt).on_classical_vals(a=left, b=right)
        assert register_add_classical_values(left, right, width, signed=signed) == (
            expected["a"],
            expected["b"],
        )
    for constant, target in product((-2, 0, 1, 3), domain):
        expected = AddK(dt, constant).on_classical_vals(x=target)["x"]
        actual = add_constant_classical_value(target, constant, width, signed=signed)
        equal(actual, expected)


@pytest.mark.parametrize(
    "width,msb,adjoint", tuple(product((0, 1, 2, 3), (False, True), (False, True)))
)
def test_out_of_place_classical_right_and_left_ports(width, msb, adjoint):
    from qualtran.bloqs.arithmetic.addition import OutOfPlaceAdder

    source = OutOfPlaceAdder(
        width, is_adjoint=adjoint, include_most_significant_bit=msb
    )
    for left, right in product(range(1 << width), repeat=2):
        output = (left + right) % (1 << (width + int(msb))) if adjoint else None
        expected = source.on_classical_vals(a=left, b=right, c=output)
        actual = out_of_place_sum_classical_values(
            left, right, width, include_msb=msb, adjoint=adjoint, output=output
        )
        assert actual == expected


def test_source_integer_casts_symbolic_boundaries_and_unchecked_sum_erase():
    import sympy
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.addition import Add, AddK, OutOfPlaceAdder

    width = sympy.Integer(3)
    assert (
        register_add_classical_values(7, 2, width)
        == tuple(Add(QUInt(width)).on_classical_vals(a=7, b=2).values())
        == (7, 1)
    )
    for w, k in ((width, 1), (3, sympy.Integer(1)), (sympy.Symbol("n"), 1)):
        with pytest.raises(ValueError):
            AddK(QUInt(w), k).on_classical_vals(x=1)
        with pytest.raises(ValueError):
            add_constant_classical_value(1, k, w)
    for w in (width, sympy.Symbol("n")):
        with pytest.raises(ValueError):
            OutOfPlaceAdder(w).on_classical_vals(a=1, b=2)
        with pytest.raises(ValueError):
            out_of_place_sum_classical_values(1, 2, w)
    # Original classical adjoint discards even a wrong sum. It proves no cleanup.
    expected = OutOfPlaceAdder(2, is_adjoint=True).on_classical_vals(a=1, b=2, c=0)
    assert (
        out_of_place_sum_classical_values(1, 2, 2, adjoint=True, output=0)
        == expected
        == {"a": 1, "b": 2}
    )


@pytest.mark.parametrize("adjoint,output", ((False, 0), (True, None)))
def test_original_assertion_contract_on_sum_output(adjoint, output):
    from qualtran.bloqs.arithmetic.addition import OutOfPlaceAdder

    with pytest.raises(AssertionError):
        OutOfPlaceAdder(2, is_adjoint=adjoint).on_classical_vals(a=1, b=2, c=output)
    with pytest.raises(AssertionError):
        out_of_place_sum_classical_values(1, 2, 2, adjoint=adjoint, output=output)


@pytest.mark.parametrize(
    "m,n,signed",
    (
        (1, 1, False),
        (1, 2, False),
        (2, 2, False),
        (2, 3, False),
        (3, 3, False),
        (1, 1, True),
        (2, 2, True),
        (3, 3, True),
    ),
)
def test_full_complex_add_output_matches_classical_model_and_every_work_branch(
    m, n, signed
):
    left = tuple(range(m))
    right = tuple(range(m, m + n))
    work = tuple(range(m + n, m + 2 * n - 1))
    slots = tuple(range(n - 1))
    gates = add_ir(left, right, work, slots)
    data = torch.arange(1 << (m + n + 1), dtype=torch.int64)
    coeff = data.to(torch.float64)
    amplitudes = torch.complex(coeff + 1, 3 * coeff - 5)
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << (m + 2 * n), dtype=torch.complex128)
    state[((data >> 1) << n) | (data & 1)] = amplitudes
    expected = torch.zeros_like(state)
    for label in data.tolist():
        a = label >> (n + 1)
        b = (label >> 1) & ((1 << n) - 1)
        if signed:
            a -= (1 << m) if a >= 1 << (m - 1) else 0
            b -= (1 << n) if b >= 1 << (n - 1) else 0
        x, y = register_add_classical_values(a, b, n, signed=signed)
        logical = ((x % (1 << m)) << n) | (y % (1 << n))
        expected[(logical << n) | (label & 1)] = amplitudes[label]
    for history in product((0, 1), repeat=n - 1):
        torch.testing.assert_close(
            forced_branch(state, gates, history),
            expected / 2 ** ((n - 1) / 2),
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.parametrize("n,msb", tuple(product((1, 2, 3), (False, True))))
def test_correct_sum_classical_ports_and_quantum_erase_all_branches(n, msb):
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.circuit import Circuit

    out_width = n + int(msb)
    total = 2 * n + out_width + 1
    left = tuple(range(n))
    right = tuple(range(n, 2 * n))
    output = tuple(range(2 * n, total - 1))
    forward = out_of_place_add_ir(left, right, output, include_msb=msb)
    inverse = out_of_place_add_ir(
        left,
        right,
        output,
        include_msb=msb,
        adjoint=True,
        classical_bits=tuple(range(out_width - 1)),
    )
    data = torch.arange(1 << (2 * n + 1), dtype=torch.int64)
    coeff = data.to(torch.float64)
    amplitudes = torch.complex(coeff + 1, 3 * coeff - 5)
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << total, dtype=torch.complex128)
    state[((data >> 1) << (out_width + 1)) | (data & 1)] = amplitudes
    expected = torch.zeros_like(state)
    for label in data.tolist():
        a = label >> (n + 1)
        b = (label >> 1) & ((1 << n) - 1)
        values = out_of_place_sum_classical_values(a, b, n, include_msb=msb)
        assert out_of_place_sum_classical_values(
            a, b, n, include_msb=msb, adjoint=True, output=values["c"]
        ) == {"a": a, "b": b}
        expected[
            ((label >> 1) << (out_width + 1)) | (values["c"] << 1) | (label & 1)
        ] = amplitudes[label]
    actual = (
        Circuit(total, inputs=state, dtype=torch.complex128)
        .append(FieldOperation(total, forward))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    for history in product((0, 1), repeat=out_width - 1):
        torch.testing.assert_close(
            forced_branch(expected, inverse, history),
            state / 2 ** ((out_width - 1) / 2),
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "kind,constant",
    tuple(product(("int8", "uint8", "int64", "array1", "array2"), (False, True))),
)
def test_add_consumers_keep_original_python_cast_policy(kind, constant):
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.addition import Add, AddK

    value = (
        np.array([3] if kind == "array1" else [3, 4])
        if kind.startswith("array")
        else getattr(np, kind)(3)
    )
    original = AddK(QUInt(3), 2) if constant else Add(QUInt(3))

    def source():
        return original.on_classical_vals(
            **({"x": value} if constant else {"a": value, "b": 2})
        )

    def native():
        return (
            {"x": add_constant_classical_value(value, 2, 3)}
            if constant
            else dict(
                zip(("a", "b"), register_add_classical_values(value, 2, 3), strict=True)
            )
        )

    try:
        expected = source()
    except (TypeError, ValueError) as exc:
        with pytest.raises(type(exc)):
            native()
    else:
        result = native()
        for name in expected:
            equal(result[name], expected[name])


def test_add_unresolved_width_conversion_and_zero_classical_word_remain_source_paths():
    import sympy
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.addition import Add

    symbol = sympy.Symbol("n", positive=True, integer=True)
    with pytest.raises(TypeError):
        Add(QUInt(symbol)).on_classical_vals(a=1, b=2)
    with pytest.raises(TypeError):
        register_add_classical_values(1, 2, symbol)
    assert (
        register_add_classical_values(0, 0, 0)
        == tuple(Add(QUInt(0)).on_classical_vals(a=0, b=0).values())
        == (0, 0)
    )
