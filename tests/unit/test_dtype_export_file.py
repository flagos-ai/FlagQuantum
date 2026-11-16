"""Verify the complete locked dtype export surface against owned descriptors."""

import ast
import hashlib
from collections.abc import Callable
from importlib import import_module
from pathlib import Path

import pytest
from sympy import Symbol

from flagquantum.core.ir.program import register_types as types
from flagquantum.core.ir.program.registers import RegisterType
from flagquantum.core.value_math import SymbolicInt as RegisterWidth
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_polynomial_encoding import (
    FiniteFieldPolynomialEncoding,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
SOURCE = (
    Path(__file__).resolve().parents[3]
    / "ftqc-reference-env/qualtran/dtype/__init__.py"
)
CATEGORIES = ("QCDType", "CDType", "QDType")
WIDTH = Symbol("n", positive=True, integer=True)


def _field(width: RegisterWidth, *, classical: bool) -> RegisterType:
    return FiniteFieldEncoding(2, width).register_type(classical=classical)


def _polynomial(width: RegisterWidth, *, classical: bool) -> RegisterType:
    return FiniteFieldPolynomialEncoding(
        width, FiniteFieldEncoding(2, 2)
    ).register_type(classical=classical)


CASES: dict[
    str,
    tuple[
        Callable[[RegisterWidth], RegisterType],
        Callable[[RegisterWidth], tuple[object, ...]],
    ],
] = {
    "QAny": (types.opaque_register_type, lambda width: (width,)),
    "QBit": (lambda _: types.quantum_bit_type(), lambda _: ()),
    "CBit": (lambda _: types.classical_bit_type(), lambda _: ()),
    "QInt": (types.signed_register_type, lambda width: (width,)),
    "CInt": (types.classical_signed_register_type, lambda width: (width,)),
    "QIntOnesComp": (types.ones_complement_register_type, lambda width: (width,)),
    "CIntOnesComp": (
        types.classical_ones_complement_register_type,
        lambda width: (width,),
    ),
    "QUInt": (types.unsigned_register_type, lambda width: (width,)),
    "CUInt": (types.classical_unsigned_register_type, lambda width: (width,)),
    "BQUInt": (types.bounded_unsigned_register_type, lambda width: (width,)),
    "BCUInt": (types.classical_bounded_unsigned_register_type, lambda width: (width,)),
    "QFxp": (
        lambda width: types.fixed_point_register_type(width, 1),
        lambda width: (width, 1),
    ),
    "CFxp": (
        lambda width: types.classical_fixed_point_register_type(width, 1),
        lambda width: (width, 1),
    ),
    "QMontgomeryUInt": (types.montgomery_register_type, lambda width: (width,)),
    "CMontgomeryUInt": (
        types.classical_montgomery_register_type,
        lambda width: (width,),
    ),
    "QGF": (lambda width: _field(width, classical=False), lambda width: (2, width)),
    "CGF": (lambda width: _field(width, classical=True), lambda width: (2, width)),
    "QGFPoly": (
        lambda width: _polynomial(width, classical=False),
        lambda width: (width, import_module("qualtran.dtype").QGF(2, 2)),
    ),
    "CGFPoly": (
        lambda width: _polynomial(width, classical=True),
        lambda width: (width, import_module("qualtran.dtype").QGF(2, 2)),
    ),
}


def test_export_order_and_identity() -> None:
    data = SOURCE.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        == "bb233e268a3baee166641123f9d927ac64f1adde"
    )
    package = import_module("qualtran.dtype")
    definitions = import_module("qualtran._infra.data_types")
    exports = next(
        statement
        for statement in ast.parse(data).body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in statement.targets
        )
    )
    assert tuple(ast.literal_eval(exports.value)) == (*CATEGORIES, *CASES)
    assert tuple(package.__all__) == (*CATEGORIES, *CASES)
    for name in package.__all__:
        assert getattr(package, name) is getattr(definitions, name)


@pytest.mark.parametrize("name", tuple(CASES))
@pytest.mark.parametrize("width", [3, WIDTH])
def test_concrete_and_symbolic_storage_and_width(
    name: str, width: RegisterWidth
) -> None:
    source = import_module("qualtran.dtype")
    native_factory, arguments = CASES[name]
    expected = getattr(source, name)(*arguments(width))
    actual = native_factory(width)
    assert isinstance(actual, RegisterType)
    assert isinstance(expected, source.QCDType)
    assert isinstance(
        expected, source.CDType if name.startswith(("C", "BC")) else source.QDType
    )
    assert (actual.num_bits, actual.num_qubits, actual.num_cbits) == (
        expected.num_bits,
        expected.num_qubits,
        expected.num_cbits,
    )
    assert actual.is_symbolic() == expected.is_symbolic()


@pytest.mark.parametrize("name", ["QInt", "CInt", "QUInt", "CUInt"])
def test_zero_width_keeps_storage_identity(name: str) -> None:
    source = import_module("qualtran.dtype")
    native_factory, arguments = CASES[name]
    expected = getattr(source, name)(*arguments(0))
    actual = native_factory(0)
    assert (actual.num_bits, actual.num_qubits, actual.num_cbits) == (0, 0, 0)
    assert expected.num_bits == 0
    assert (("storage", "classical") in actual.parameters) == isinstance(
        expected, source.CDType
    )
