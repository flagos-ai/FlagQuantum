"""Constant SymPy objects retain source classification and available parents."""

from functools import cache
from importlib import import_module
from typing import Any, cast
from unittest.mock import patch

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    for dependency in ("numpy", "cirq", "kickmix"):
        import_module(dependency)
    import sympy
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.cryptography.ecc.ec_add import ECAdd
    from qualtran.bloqs.cryptography.ecc.ec_add_r import ECWindowAddR
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint
except ModuleNotFoundError as error:
    unavailable_reference(error)

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_constant_control_factories import (
    unresolved_constant_addition_program,
)
from flagquantum.compiler.formats.legacy_modular_control_factories import (
    modular_control_factories,
)
from flagquantum.compiler.formats.legacy_modular_multiplication_factories import (
    modular_multiplication_factories,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.modular.additional_control_component import (
    additional_modular_addition_program,
    additional_modular_subtraction_program,
)
from flagquantum.compiler.registers.modular.controlled_doubling_component import (
    controlled_modular_doubling_program,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    montgomery_register_type,
    unsigned_register_type,
)
from flagquantum.ecosystem._elliptic_curve_control_library import (
    controlled_curve_addition,
    controlled_curve_arithmetic_stage,
    elliptic_curve_control_factories,
)
from flagquantum.ecosystem._elliptic_curve_window_library import (
    elliptic_curve_window_factories,
)
from flagquantum.errors import SerializationError
from tests.unit import test_symbolic_additional_modular_controls as modular
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_ecc_addition_model import frozen_source as frozen_source
from tests.unit.test_ecc_controlled_stages import available_layers
from tests.unit.test_ecc_window_controlled import (
    frozen_window_source as frozen_window_source,
)
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_symbolic_modular_addition_calls import (
    frozen_modular_sources as frozen_modular_sources,
)
from tests.unit.test_symbolic_modular_doubling_calls import (
    frozen_doubling_source as frozen_doubling_source,
)
from tests.unit.test_symbolic_modular_doubling_calls import (
    source_body_program,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_KINDS = ("ModAdd", "CModAdd0", "CModAdd1", "ModSub", "CModSub0", "CModSub1", "ModDbl")
_P = sympy.Integer(3)


@cache
def constant_modular(kind: str, inverse: bool) -> ProgramRecord:
    with patch.object(modular, "_P", _P):
        return modular.native(kind, 3, inverse)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("inverse", (False, True))
def test_constant_sympy_modulus_available_layers_and_actual_symbolic_refusals(
    kind: str, inverse: bool
) -> None:
    with patch.object(modular, "_P", _P):
        source = modular.original(kind, 3, inverse).controlled()
    pairs = available_layers(constant_modular(kind, inverse), source)
    witnessed = 0
    for child in pairs.values():
        while type(child).__name__ in {"Controlled", "Adjoint"}:
            child = child.subbloq
        if type(child).__name__ == "AddK":
            assert isinstance(child.k, sympy.Integer) and not child.k.free_symbols
            with pytest.raises(DecomposeTypeError, match="symbolic AddK"):
                child.decompose_bloq()
            witnessed += 1
    assert witnessed >= 2


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("depth", (0, 1))
def test_constant_symbolic_constructor_transport(kind: str, depth: int) -> None:
    with patch.object(modular, "_P", _P):
        source = modular.original(kind, 3, True).controlled()
    factories = (
        modular_multiplication_factories()
        if kind == "ModDbl"
        else modular_control_factories()
    )
    codec = import_module("qualtran.serialization.bloq")
    proto = codec.bloqs_to_proto(source, max_depth=depth)
    if "Sub" in kind:
        # The frozen source resolver omits ModSub/CModSub entirely. Preserve
        # that failure and compare its standalone argument decoder instead.
        with pytest.raises(ValueError, match="Unable to find a Bloq.*mod_subtraction"):
            codec.bloqs_from_proto(proto)
        class_name = kind[:-1] if kind.startswith("C") else kind
        entry = next(
            item for item in proto.table if item.bloq.name.endswith("." + class_name)
        )
        argument = next(arg for arg in entry.bloq.args if arg.name == "mod")
        decoded_modulus = codec.arg_from_proto(argument)["mod"]
        assert type(decoded_modulus) is int
        with patch.object(modular, "_P", decoded_modulus):
            decoded = modular.original(kind, 3, True).controlled()
    else:
        decoded = codec.bloqs_from_proto(proto)[0]
    child = decoded
    while type(child).__name__ in {"Controlled", "Adjoint"}:
        child = child.subbloq
    # The original wire decoder itself changes SymPy Integer into Python int.
    # Compare the actual decoded constructor, not the pre-serialization object.
    assert type(child.mod) is int and child.mod == 3
    record = resolve_component_program(_library(source, depth), 0, factories)
    record = source_body_program(record, source)
    dtype = montgomery_register_type(3, 3 if kind == "ModAdd" else None)
    if kind == "ModDbl":
        expected = controlled_modular_doubling_program(dtype, 3, adjoint=True)
    else:
        callback = (
            additional_modular_addition_program
            if "Add" in kind
            else additional_modular_subtraction_program
        )
        expected = callback(
            dtype,
            3,
            control_value=int(kind[-1]) if kind.startswith("C") else None,
            adjoint=True,
        )
    assert matches_component_definition(record, record.functions[0].name, expected)


@pytest.mark.parametrize("kind", ("coordinates", "products"))
def test_controlled_ecc_constant_symbolic_modulus_retains_real_stage_bodies(
    kind: Any,
) -> None:
    from qualtran.bloqs.cryptography.ecc import ec_add

    native = controlled_curve_arithmetic_stage(kind, 3, _P)
    original = (
        ec_add._ECAddStepThree if kind == "coordinates" else ec_add._ECAddStepFour
    )(3, _P).controlled()
    available_layers(native, original)


def test_controlled_ecc_six_call_parent_and_constructor_binding() -> None:
    source = ECAdd(3, _P).controlled()
    direct = controlled_curve_addition(3, _P)
    direct_calls = direct.functions[0].blocks[0].operations
    assert len(direct_calls) == 6
    compare_layers(
        direct, source, {op.callee for op in direct_calls if op.callee is not None}
    )
    record = resolve_component_program(
        _library(source, 0), 0, elliptic_curve_control_factories()
    )
    codec = import_module("qualtran.serialization.bloq")
    decoded = codec.bloqs_from_proto(codec.bloqs_to_proto(source, max_depth=0))[0]
    assert type(decoded.subbloq.mod) is int
    assert matches_component_definition(
        record,
        record.functions[0].name,
        controlled_curve_addition(3, decoded.subbloq.mod),
    )
    calls = record.functions[0].blocks[0].operations
    assert len(calls) == 6
    compare_layers(
        record, decoded, {op.callee for op in calls if op.callee is not None}
    )


@cache
def window_record(
    symbolic_coordinate: bool, controlled: bool
) -> tuple[Any, ProgramRecord]:
    x = sympy.Integer(1) if symbolic_coordinate else 1
    source = ECWindowAddR(3, ECPoint(x, 1, 3, 1), 1)
    if controlled:
        source = source.controlled()
    factories = (
        elliptic_curve_control_factories()
        if controlled
        else elliptic_curve_window_factories()
    )
    return source, resolve_component_program(_library(source, 0), 0, factories)


@pytest.mark.parametrize("controlled", (False, True))
def test_integer_sympy_point_coordinate_keeps_complete_ordinary_and_controlled_window(
    controlled: bool,
) -> None:
    source, record = window_record(True, controlled)
    _, finite = window_record(False, controlled)
    calls = record.functions[0].blocks[0].operations
    assert len(calls) == 5
    assert matches_component_definition(record, record.functions[0].name, finite)
    compare_layers(record, source, {op.callee for op in calls if op.callee is not None})
    original = ECWindowAddR(3, ECPoint(sympy.Integer(1), 1, 3, 1), 1)
    numeric = ECWindowAddR(3, ECPoint(1, 1, 3, 1), 1)
    assert original.qrom == numeric.qrom


def test_actual_point_modulus_symbolic_assertion_is_not_converted_to_integer() -> None:
    source = ECWindowAddR(4, ECPoint(2, 2, sympy.Integer(7), 3), 2).controlled()
    with pytest.raises(AssertionError):
        source.decompose_bloq()
    with pytest.raises(AssertionError):
        resolve_component_program(
            _library(source, 0), 0, elliptic_curve_control_factories()
        )


def test_python_integer_does_not_enter_symbolic_constant_declaration_path() -> None:
    with pytest.raises(SerializationError, match="SymPy expression"):
        unresolved_constant_addition_program(
            unsigned_register_type(3), cast(Any, 3), name="constant"
        )
