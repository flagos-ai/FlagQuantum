"""Doubling and constant-product graphs preserve the literal source branches."""

from importlib import import_module
from typing import Literal

from tests._reference_dependencies import unavailable_reference

# Typed-register owners require NumPy before their modules can be collected.
try:
    import_module("numpy")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

import pytest
import sympy

from flagquantum.compiler.formats.product_modular_library import (
    product_modular_component_library,
)
from flagquantum.core.ir.program.register_types import (
    montgomery_register_type,
    unsigned_register_type,
)
from flagquantum.core.value_math import SymbolicInt
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_symbolic_modular_library import _canonical

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_N, _P, _K = sympy.symbols("n p k", integer=True, positive=True)


@pytest.mark.parametrize(
    "width,modulus,constant",
    ((3, _P, 2), (4, _P, 2), (_N, 5, 2), (4, 5, _K), (3, 5, 2)),
)
@pytest.mark.parametrize("montgomery", (False, True))
def test_literal_constant_product_all_calls_and_positive_symbolic_reciprocal(
    width: SymbolicInt,
    modulus: SymbolicInt,
    constant: SymbolicInt,
    montgomery: bool,
) -> None:
    try:
        from qualtran._infra.data_types import QMontgomeryUInt, QUInt
        from qualtran.bloqs.mod_arithmetic.mod_multiplication import CModMulK
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    dtype = (
        montgomery_register_type(width) if montgomery else unsigned_register_type(width)
    )
    source = CModMulK(
        QMontgomeryUInt(width) if montgomery else QUInt(width), k=constant, mod=modulus
    )
    native = product_modular_component_library(
        "CModMulK", dtype, modulus, constant=constant
    )
    assert _canonical(native) == _canonical(_library(source, 1))
    scales = [
        row
        for row in native.entries
        if row.declaration.name.endswith(".CtrlScaleModAdd")
    ]
    second = next(
        arg.value for arg in scales[1].declaration.arguments if arg.name == "k"
    )
    expected = (
        sympy.Mod(sympy.Pow(constant, -1), modulus)
        if isinstance(constant, sympy.Expr) or isinstance(modulus, sympy.Expr)
        else -pow(constant, -1, modulus)
    )
    assert second == expected


@pytest.mark.parametrize("width", (1, 2, 3, 4))
@pytest.mark.parametrize("montgomery", (False, True))
def test_doubling_symbolic_modulus_keeps_the_exact_signed_correction_and_bit_order(
    width: int, montgomery: bool
) -> None:
    try:
        from qualtran._infra.data_types import QMontgomeryUInt, QUInt
        from qualtran.bloqs.mod_arithmetic.mod_multiplication import ModDbl
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    dtype = (
        montgomery_register_type(width) if montgomery else unsigned_register_type(width)
    )
    source = ModDbl(QMontgomeryUInt(width) if montgomery else QUInt(width), _P)
    native = product_modular_component_library("ModDbl", dtype, _P)
    assert _canonical(native) == _canonical(_library(source, 1))


@pytest.mark.parametrize("kind", ("ModDbl", "CModMulK"))
@pytest.mark.parametrize("montgomery", (False, True))
def test_owned_libraries_use_existing_physical_registry_and_refuse_unbound_parameters(
    kind: Literal["ModDbl", "CModMulK"],
    montgomery: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.compiler.formats.legacy_modular_multiplication_factories import (
        modular_multiplication_factories,
    )
    from flagquantum.compiler.registers.modular.modular_doubling_component import (
        modular_doubling_program,
    )
    from flagquantum.compiler.registers.modular.modular_multiplication_component import (
        modular_multiplication_program,
    )
    from tests.unit.register_library_checks import bind_quantum_library

    dtype = montgomery_register_type(2) if montgomery else unsigned_register_type(2)
    direct = (
        modular_doubling_program(dtype, 3)
        if kind == "ModDbl"
        else modular_multiplication_program(dtype, 3, 2)
    )
    inputs = ((0, 1),) if kind == "ModDbl" else ((0,), (1, 2))
    total = 2 if kind == "ModDbl" else 3
    expected = bind_quantum_library(direct, inputs, total)
    for name in ("qualtran", "cirq", "kickmix"):
        monkeypatch.setitem(sys.modules, name, None)
    native = product_modular_component_library(kind, dtype, 3, constant=2)
    actual = bind_quantum_library(
        resolve_component_program(native, 0, modular_multiplication_factories()),
        inputs,
        total,
    )
    assert actual.circuit == expected.circuit and actual.outputs == expected.outputs
    assert tuple(
        (e.kind, e.wires, e.instruction_offset) for e in actual.events
    ) == tuple((e.kind, e.wires, e.instruction_offset) for e in expected.events)
    symbolic = product_modular_component_library(kind, dtype, _P, constant=2)
    with pytest.raises(ValueError, match="component bodies"):
        bind_quantum_library(
            resolve_component_program(symbolic, 0, modular_multiplication_factories()),
            inputs,
            total,
        )
