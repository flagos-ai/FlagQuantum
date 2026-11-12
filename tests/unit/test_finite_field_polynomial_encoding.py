"""Polynomial numerical methods follow pinned quantum/classical source codecs."""

from collections import Counter
from collections.abc import Callable
from functools import partial
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.core.value_math import SymbolicInt
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_polynomial_encoding import (
    FiniteFieldPolynomialEncoding,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CASES = [(2, 1, 0), (2, 1, 2), (2, 2, 1), (2, 3, 1), (3, 1, 1), (3, 2, 0)]


def pair(
    p: int, m: int, degree: SymbolicInt, classical: bool
) -> tuple[FiniteFieldPolynomialEncoding, Any]:
    from qualtran import QGF, CGFPoly, QGFPoly

    return FiniteFieldPolynomialEncoding(degree, FiniteFieldEncoding(p, m)), (
        CGFPoly if classical else QGFPoly
    )(degree, QGF(p, m))


def same_call(native: Callable[[], object], source: Callable[[], object]) -> None:
    try:
        expected = source()
    except (ValueError, TypeError, IndexError, OverflowError) as error:
        with pytest.raises(type(error)) as caught:
            native()
        assert str(caught.value) == str(error)
    else:
        actual = native()
        assert type(actual) is type(expected)
        if isinstance(actual, np.ndarray):
            assert isinstance(expected, np.ndarray)
            assert actual.dtype == expected.dtype and actual.shape == expected.shape
            for a, b in zip(actual.flat, expected.flat, strict=True):
                assert type(a) is type(b) and a == b
        else:
            assert actual == expected


@pytest.mark.parametrize("p,m,degree", CASES)
@pytest.mark.parametrize("classical", [False, True])
def test_complete_domains_coefficients_and_scalar_codecs(
    p: int, m: int, degree: int, classical: bool
) -> None:
    encoding, source = pair(p, m, degree, classical)
    assert encoding.bitsize == source.num_bits
    actual = list(encoding.get_domain())
    expected = list(source.get_classical_domain())
    assert actual == expected
    assert len(actual) == (p**m) ** (degree + 1)
    for a, b in zip(actual, expected, strict=True):
        assert type(a) is type(b) and a.field is b.field is encoding.field.gf_type
        encoding.assert_valid_val(a)
        source.assert_valid_classical_val(b)
        coeffs = encoding.to_gf_coefficients(a)
        np.testing.assert_array_equal(
            coeffs, source._bit_encoding.to_gf_coefficients(b)
        )
        assert type(coeffs) is encoding.field.gf_type
        assert encoding.from_gf_coefficients(coeffs) == b
        same_call(partial(encoding.to_bits, a), partial(source.to_bits, b))
    for bits in product((0, 1), repeat=int(encoding.bitsize)):
        same_call(partial(encoding.from_bits, bits), partial(source.from_bits, bits))


@pytest.mark.parametrize("p,m,degree", CASES)
@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,), (2, 0)])
def test_array_defaults_and_shape_failures(
    p: int, m: int, degree: int, classical: bool, shape: tuple[int, ...]
) -> None:
    encoding, source = pair(p, m, degree, classical)
    polynomial = next(iter(encoding.get_domain()))
    values = np.empty(shape, dtype=object)
    values[...] = polynomial
    same_call(
        lambda: encoding.to_bits_array(values), lambda: source.to_bits_array(values)
    )
    bits = np.zeros(shape + (int(encoding.bitsize),), dtype=np.uint8)
    same_call(
        lambda: encoding.from_bits_array(bits), lambda: source.from_bits_array(bits)
    )
    same_call(
        lambda: encoding.assert_valid_val_array(values),
        lambda: source.assert_valid_classical_val_array(values),
    )


@pytest.mark.parametrize("classical", [False, True])
def test_validation_order_and_unchecked_coefficients(classical: bool) -> None:
    import galois

    encoding, source = pair(2, 2, 1, classical)
    cases = [
        0,
        None,
        galois.Poly([1], field=galois.GF(3)),
        galois.Poly([1, 1, 1], field=encoding.field.gf_type),
    ]
    for value in cases:
        same_call(
            partial(encoding.assert_valid_val, value, "sentinel"),
            partial(source.assert_valid_classical_val, value, "sentinel"),
        )
        same_call(partial(encoding.to_bits, value), partial(source.to_bits, value))
    for coefficients in [[], [1], [1, 0, 1], [5], encoding.field.gf_type([1, 2])]:
        same_call(
            partial(encoding.from_gf_coefficients, coefficients),
            partial(source._bit_encoding.from_gf_coefficients, coefficients),
        )
    for bits in [[], [0], [0] * 3, [0] * 5, [0, 1, 2, 0]]:
        same_call(partial(encoding.from_bits, bits), partial(source.from_bits, bits))


@pytest.mark.parametrize("degree", [-1, -2, 0])
def test_degree_edges_are_not_normalized(degree: int) -> None:
    encoding, source = pair(2, 1, degree, False)
    assert encoding.bitsize == source.num_bits
    same_call(
        lambda: list(encoding.get_domain()), lambda: list(source.get_classical_domain())
    )
    same_call(lambda: encoding.from_bits([]), lambda: source.from_bits([]))


def test_symbolic_degree_keeps_metadata_and_original_failures() -> None:
    import sympy

    degree = sympy.Symbol("n", integer=True, positive=True)
    encoding, source = pair(2, 2, degree, False)
    assert encoding.bitsize == source.num_bits == 2 * (degree + 1)
    same_call(
        lambda: list(encoding.get_domain()), lambda: list(source.get_classical_domain())
    )
    same_call(lambda: encoding.from_bits([0, 0]), lambda: source.from_bits([0, 0]))


def test_existing_polynomial_add_consumer_uses_native_coefficients() -> None:
    from qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add import GF2PolyAdd
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.circuit import Circuit
    from flagquantum.compiler.registers.finite_field.field_polynomial_addition import (
        gf2_polynomial_add_ir,
    )
    from flagquantum.compiler.registers.finite_field.field_polynomial_layout import (
        FieldPolynomialLayout,
    )
    from tests.unit.test_field_polynomial_addition import (
        test_polynomial_add_gate_stream_and_source_resources,
    )

    # Existing unchanged source decomposition comparison establishes the physical stream.
    test_polynomial_add_gate_stream_and_source_resources(1, 2)
    encoding, source = pair(2, 2, 1, False)
    layout = FieldPolynomialLayout(1, 2)
    n = layout.logical_width
    mapping = []
    for word in range(1 << n):
        bits = list(map(int, f"{word:0{n}b}"))
        polynomial = encoding.from_bits(bits)
        packed = layout.pack_basis(
            tuple(map(int, encoding.to_gf_coefficients(polynomial)))
        )
        assert packed == word
        mapping.append(polynomial)
    matrix = torch.zeros((256, 256), dtype=torch.complex128)
    for left, right in product(range(16), repeat=2):
        result = mapping[left] + mapping[right]
        out = layout.pack_basis(tuple(map(int, encoding.to_gf_coefficients(result))))
        assert out == left ^ right
        matrix[(left << n) | out, (left << n) | right] = 1
    values = torch.arange(1, 257, dtype=torch.float64)
    live = torch.complex(values, values.flip(0))
    live /= live.norm()
    state = torch.zeros(512, dtype=torch.complex128)
    state[::2] = live
    gates = gf2_polynomial_add_ir(tuple(range(n)), tuple(range(n, 2 * n)), layout)
    operation = FieldOperation(9, gates)
    circuit = Circuit(9, inputs=state, dtype=torch.complex128).append(operation)
    expected = torch.zeros_like(state)
    expected[::2] = matrix @ live
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=1e-12)
    assert torch.count_nonzero(circuit.state().flatten()[1::2]) == 0
    torch.testing.assert_close(
        circuit.append(operation).state().flatten(), state, rtol=0, atol=1e-12
    )
    assert Counter(g.name for g in gates) == {"cx": 4}
    model = get_cost_value(GF2PolyAdd(source), QECGatesCost())
    assert model.clifford == 4 and model.total_t_count() == 0 and model.measurement == 0
