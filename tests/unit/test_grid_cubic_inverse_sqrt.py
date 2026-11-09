"""Verify table interpolation, normalized arguments and corrected Newton cleanup."""

from cmath import exp
from decimal import Decimal, localcontext
from math import sqrt

import pytest

from flagquantum.algorithms.primitives.grid_cubic_inverse_sqrt import (
    grid_cubic_inverse_sqrt_requirements,
    with_grid_cubic_inverse_sqrt_ir,
)
from flagquantum.compiler.registers.bits.bitwise import xor_ir
from flagquantum.core.ir import CircuitIR
from tests.unit.sparse_register_execution import execute_sparse_register

pytestmark = pytest.mark.unit


def cubic_word(distance, precision):
    if not distance:
        return 0
    with localcontext() as context:
        context.prec = 60
        k = distance.bit_length() - 1
        normalized = Decimal(distance) / (1 << k)
        high = normalized >= Decimal("1.5")
        constants = (
            (
                "0.81648515205385221995",
                "0.27136515484240234115",
                "0.12756148214815175348",
                "0.044753028579153842218",
            )
            if high
            else (
                "0.99994132489119882162",
                "0.49609891915903542303",
                "0.33261112772430493331",
                "0.14876762006038398086",
            )
        )
        u = normalized - Decimal("1.5" if high else "1")
        unit = 1 << precision
        coefficients = [
            int(Decimal(c) * unit / Decimal(1 << k).sqrt()) for c in constants
        ]
        word = coefficients.pop()
        while coefficients:
            word = coefficients.pop() - int(u * word)
        delta = Decimal(
            "3.6279794522852781448e-10" if high else "5.1642030908180720584e-9"
        )
        factor = 3 * unit**2 + int(delta * unit**2) - distance * word**2
        return word * factor // (2 * unit**2)


@pytest.mark.parametrize(
    "width,precision,values", [(4, 5, tuple(range(16))), (3, 16, (0, 1, 3, 6))]
)
def test_table_cubic_inverse_sqrt_clean_words(width, precision, values):
    work, slots = grid_cubic_inverse_sqrt_requirements(width, precision)
    inputs = tuple(range(width))
    out = tuple(range(width, width + precision + 1))
    scratch = tuple(range(width + precision + 1, width + precision + 1 + work))
    gates = with_grid_cubic_inverse_sqrt_ir(
        inputs,
        scratch,
        tuple(range(slots)),
        precision=precision,
        consume=lambda estimate: xor_ir(estimate, out),
    )
    total = width + precision + 1 + work
    incoming, expected = {}, {}
    for value in values:
        key = value << (total - width)
        word = cubic_word(value, precision)
        amp = exp(0.31j * value) / sqrt(len(values))
        incoming[key] = amp
        expected[key | (word << work)] = amp
    for seed in (3, 29):
        observed = execute_sparse_register(CircuitIR(total, gates), incoming, seed=seed)
        assert all(
            abs(observed.get(i, 0) - expected.get(i, 0)) < 3e-12
            for i in observed.keys() | expected.keys()
        )
