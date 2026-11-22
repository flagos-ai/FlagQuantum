"""Compare complete binary field construction and algebra with frozen C++ output."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.finite_field.binary_linear_map import (
    binary_linear_map_ir,
    field_constant_scale_ir,
    field_frobenius_ir,
)
from flagquantum.core._binary_field import BinaryField
from flagquantum.core._binary_field_modulus import (
    PolynomialModulus,
    default_polynomial,
    is_irreducible,
    mersenne_prime_factors,
)
from flagquantum.core._binary_matrix import BinaryMatrix
from flagquantum.core._binary_polynomial import multiply, parse

pytestmark = pytest.mark.unit
_FIXTURE = Path(__file__).with_name("fixtures") / "binary_field_values.json"


def _hex(value: int) -> str:
    return f"0x{value:X}"


def _packed(matrix: BinaryMatrix) -> str:
    return f"{matrix.rows},{matrix.columns}:" + "".join(
        f"{row:X}," for row in matrix.packed_rows
    )


def _observe(request: list[str]) -> str:
    kind = request[0]
    if kind == "factors":
        return "".join(
            _hex(prime) + "," for prime in mersenne_prime_factors(int(request[1]))
        )
    if kind == "default":
        return _hex(default_polynomial(int(request[1])))
    if kind == "irreducible":
        return str(int(is_irreducible(parse(bytes.fromhex(request[1]).decode()))))
    mode, degree = int(request[1]), int(request[2])
    modulus = bytes.fromhex(request[3]).decode()
    primitive = bytes.fromhex(request[4]).decode() if request[4] != "-" else None
    if mode == 0:
        field = BinaryField(degree)
    elif mode == 2:
        field = BinaryField(modulus=modulus, primitive_element=primitive)
    else:
        field = BinaryField(degree, modulus, primitive)
    left, right, exponent, count = (int(value, 0) for value in request[5:])
    parts = [
        str(field.degree),
        _hex(field.modulus),
        _hex(field.primitive_element),
        "0x0",
        "0x1",
        _hex(field.mod(2)),
        str(int(field.is_element(left))),
        str(int(field.is_primitive(left))),
        _hex(field.mod(left)),
        _hex(field.add(left, right)),
        _hex(field.multiply(left, right)),
        _hex(field.square(left)),
        _hex(field.frobenius(left, count)),
    ]
    try:
        parts.append(_hex(field.power(left, exponent)))
    except ValueError:
        parts.append("ERROR")
    parts += [
        _hex(field.inverse(left)),
        _hex(field.divide(left, right)),
        _hex(field.reduce(multiply(field.mod(left), field.mod(right)))),
        _packed(field.frobenius_matrix(count)),
    ]
    try:
        parts.append(_packed(field.constant_matrix(right)))
    except ValueError:
        parts.append("ERROR")
    parts += [
        _packed(
            field.linear_map_matrix(lambda value: field.add(field.square(value), value))
        ),
        "1",
        str(int(field != BinaryField(1))),
    ]
    return "|".join(parts)


def test_complete_original_field_values_and_construction_refusals() -> None:
    fixture = json.loads(_FIXTURE.read_text())
    assert {source["parent"] for source in fixture["sources"]} >= {
        "K-0302",
        "K-0303",
        "K-0304",
        "K-0305",
    }
    for case in fixture["cases"]:
        try:
            result = _observe(case["request"])
        except ValueError:
            result = "ERROR"
        assert hashlib.sha256(result.encode()).hexdigest() == case["digest"], case[
            "request"
        ]


def test_all_factor_orders_and_bounded_default_cache() -> None:
    for degree in range(1, 513):
        factors = mersenne_prime_factors(degree)
        assert len(factors) == len(set(factors))
        remaining = (1 << degree) - 1
        for prime in factors:
            assert prime > 1
            while remaining % prime == 0:
                remaining //= prime
        assert remaining == 1
    assert default_polynomial(8) == 0x11D
    hits = default_polynomial.cache_info().hits
    assert default_polynomial(8) == 0x11D
    assert default_polynomial.cache_info().hits == hits + 1
    assert default_polynomial.cache_info().maxsize == 512


def test_explicit_modulus_generator_identity_and_unreduced_operands() -> None:
    default = BinaryField(8)
    aes = BinaryField(8, 0x11B)
    assert aes == BinaryField(modulus="x^8+x^4+x^3+x+1", primitive_element=3)
    assert aes != BinaryField(8, 0x11B, 5)
    assert aes != default
    assert BinaryField(8, 0x1B) == aes
    assert aes.multiply(0x57, 0x83) == 0xC1
    assert aes.inverse(0x53) == 0xCA
    assert aes.inverse(0) == aes.divide(17, 0) == 0
    assert aes.add(1 << 100, 1) == (1 << 100) | 1
    assert aes.square(1 << 100) == aes.multiply(1 << 100, 1 << 100)


def test_internal_sparse_dense_square_and_irreducibility_order() -> None:
    assert is_irreducible(2) and is_irreducible(3)
    assert not is_irreducible(0) and not is_irreducible(1)
    assert not is_irreducible(1 << 513)
    with pytest.raises(ValueError):
        is_irreducible((1 << 513) | 1)
    sparse = PolynomialModulus(8, 0x11B)
    dense = PolynomialModulus(32, 0x1F2C32B6D)
    assert sparse.sparse_reduction and not dense.sparse_reduction
    for modulus in (sparse, dense):
        value = (1 << modulus.degree) - 1
        assert modulus.square(value) == modulus.multiply(value, value)
    assert PolynomialModulus(1, 3).is_primitive(1)
    assert not PolynomialModulus(8, 0x11B).is_primitive(1)


@pytest.mark.parametrize("degree", [0, 513, True])
def test_invalid_degrees_are_refused(degree: int) -> None:
    with pytest.raises(ValueError):
        BinaryField(degree)


def test_invalid_operands_and_counts_are_refused() -> None:
    with pytest.raises(ValueError):
        BinaryField()
    for modulus in (-1, 1 << 1024):
        with pytest.raises(ValueError):
            BinaryField(8, modulus)
    field = BinaryField(8)
    for value in (-1, 1 << 1024):
        with pytest.raises(ValueError):
            field.mod(value)
        with pytest.raises(ValueError):
            is_irreducible(value)
    with pytest.raises(ValueError):
        field.power(2, -1)
    with pytest.raises(ValueError):
        field.frobenius(2, -1)
    with pytest.raises(ValueError):
        field.frobenius_matrix(-1)
    with pytest.raises(ValueError):
        field.constant_matrix(0)


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
def test_append_uses_field_matrices_and_preserves_complex_spectator(
    degree: int,
) -> None:
    field = BinaryField(degree)
    target = tuple(range(degree))
    constant = 1 if degree == 1 else 3
    indices = torch.arange(1 << (degree + 1), dtype=torch.float64)
    state = torch.complex(indices + 1, 7 - 3 * indices)
    state /= state.norm()
    for matrix, reference in (
        (
            field.constant_matrix(constant),
            field_constant_scale_ir(target, constant, modulus=field.modulus),
        ),
        (
            field.frobenius_matrix(3),
            field_frobenius_ir(target, 3, modulus=field.modulus),
        ),
    ):
        gates = binary_linear_map_ir(target, matrix.transposed().packed_rows)
        # An identity field wrapper may omit its logical mark, but primitive
        # calls, operands and order agree at the physical instruction boundary.
        assert gates == reference
        expected = torch.empty_like(state)
        for index in range(state.numel()):
            expected[(matrix.apply(index >> 1) << 1) | (index & 1)] = state[index]
        circuit = Circuit(degree + 1, inputs=state, dtype=torch.complex128)
        transformed = circuit.append(FieldOperation(degree + 1, gates))
        torch.testing.assert_close(
            transformed.state().flatten(), expected, rtol=0, atol=1e-12
        )
        inverse = binary_linear_map_ir(
            target, matrix.transposed().packed_rows, adjoint=True
        )
        restored = (
            transformed.append(FieldOperation(degree + 1, inverse)).state().flatten()
        )
        torch.testing.assert_close(restored, state, rtol=0, atol=1e-12)
