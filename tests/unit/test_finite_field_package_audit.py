"""Verify the frozen GF(2^m) package entrances against native executable owners."""

from hashlib import sha1
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.finite_field.coefficients import (
    gf2_add_constant_ir,
    gf2_add_ir,
)
from flagquantum.compiler.registers.finite_field.field_inverse import (
    gf2_inverse_ir,
    gf2_inverse_workspace_size,
)
from flagquantum.compiler.registers.finite_field.field_linear import (
    gf2_constant_multiplication,
    gf2_frobenius,
    gf2_monomial_multiplication,
)
from flagquantum.compiler.registers.finite_field.field_multiplication import (
    bind_field_outputs,
    gf2_multiplication,
)
from flagquantum.compiler.registers.finite_field.polynomial import (
    binary_polynomial_product_ir,
)
from flagquantum.core.ir import Instruction

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _basis_result(width: int, before: int, gates: tuple[Instruction, ...]) -> int:
    ket = torch.zeros(1 << width, dtype=torch.complex128)
    ket[before] = 1
    circuit = Circuit(width, inputs=ket, dtype=ket.dtype).append(
        FieldOperation(width, gates)
    )
    state = circuit.state().flatten()
    position = int(state.abs().argmax())
    expected = torch.zeros_like(state)
    expected[position] = 1
    torch.testing.assert_close(state, expected, atol=0, rtol=0)
    return position


def test_frozen_field_package_exposes_every_native_action() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs import gf_arithmetic as package
    from qualtran.bloqs.gf_arithmetic import (
        gf2_add_k,
        gf2_addition,
        gf2_inverse,
        gf2_square,
    )
    from qualtran.bloqs.gf_arithmetic import gf2_multiplication as mul

    raw = (Path(package.__file__).parent / "__init__.py").read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "ae37c9ece21290a12b26e9c0a62c7a4d0367cd85"
    )
    exported = {
        name: value
        for name, value in vars(package).items()
        if isinstance(value, type) and value.__module__.startswith(package.__name__)
    }
    modules = {
        "GF2AddK": gf2_add_k,
        "GF2Addition": gf2_addition,
        "GF2Inverse": gf2_inverse,
        "BinaryPolynomialMultiplication": mul,
        "GF2MulK": mul,
        "GF2Multiplication": mul,
        "GF2MulViaKaratsuba": mul,
        "GF2ShiftLeft": mul,
        "GF2ShiftRight": mul,
        "MultiplyPolyByOnePlusXk": mul,
        "GF2Square": gf2_square,
    }
    assert exported == {name: getattr(module, name) for name, module in modules.items()}

    modulus = 0b111
    a, b, target = (0, 1), (2, 3), (4, 5)
    mul_q = gf2_multiplication(a, b, target, (), modulus=modulus, algorithm="quadratic")
    mul_k = gf2_multiplication(a, b, target, (), modulus=modulus, algorithm="karatsuba")
    bindings = {
        "GF2AddK": (2, 2, 1, gf2_add_constant_ir(a, 3)),
        "GF2Addition": (4, 11, 9, gf2_add_ir(a, b)),
        "GF2Multiplication": (6, 44, 45, bind_field_outputs(mul_q, (a, b, target))),
        "GF2MulViaKaratsuba": (6, 44, 45, bind_field_outputs(mul_k, (a, b, target))),
    }
    linear = {
        "GF2MulK": (1, gf2_constant_multiplication(a, 3, modulus=modulus)),
        "GF2ShiftLeft": (1, gf2_monomial_multiplication(a, -1, modulus=modulus)),
        "GF2ShiftRight": (3, gf2_monomial_multiplication(a, 1, modulus=modulus)),
        "GF2Square": (3, gf2_frobenius(a, modulus=modulus)),
    }
    for name, (expected, lowered) in linear.items():
        bindings[name] = (2, 2, expected, bind_field_outputs(lowered, (a,)))

    width = gf2_inverse_workspace_size(2, modulus=modulus)
    work = tuple(range(4, 4 + width))
    bindings["GF2Inverse"] = (
        4 + width,
        8 << width,
        11 << width,
        gf2_inverse_ir(a, b, work, modulus=modulus),
    )
    poly = binary_polynomial_product_ir(a, b, (4, 5, 6), ())
    shifted = binary_polynomial_product_ir(a, b, (4, 5, 6, 7), (), shift=1)
    bindings["BinaryPolynomialMultiplication"] = (7, 11 << 3, (11 << 3) | 6, poly)
    bindings["MultiplyPolyByOnePlusXk"] = (8, 11 << 4, (11 << 4) | 10, shifted)

    assert bindings.keys() == exported.keys()
    for name, (wires, before, expected, gates) in bindings.items():
        assert _basis_result(wires, before, gates) == expected, name
