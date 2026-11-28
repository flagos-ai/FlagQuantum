"""Symbolic modular controls retain source roles, directions and real refusals."""

import sys
from importlib import import_module
from typing import Any
from unittest.mock import patch

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    for dependency in ("numpy", "cirq", "kickmix"):
        import_module(dependency)
    import sympy
    from qualtran import CtrlSpec, QMontgomeryUInt, QUInt
    from qualtran.bloqs.mod_arithmetic.mod_addition import CModAdd, ModAdd
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import ModDbl
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import CModSub, ModSub
except ModuleNotFoundError as error:
    unavailable_reference(error)

from flagquantum.compiler.formats import legacy_symbolic_modular_controls as owners
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
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
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    montgomery_register_type,
    unsigned_register_type,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_ecc_addition_model import frozen_source as frozen_source
from tests.unit.test_ecc_controlled_stages import available_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_symbolic_modular_addition_calls import (
    frozen_modular_sources as frozen_modular_sources,
)
from tests.unit.test_symbolic_modular_doubling_calls import (
    assert_unbound,
    source_body_program,
)
from tests.unit.test_symbolic_modular_doubling_calls import (
    frozen_doubling_source as frozen_doubling_source,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_P = sympy.Symbol("p")
_CASES = ("ModAdd", "ModSub", "ModDbl", "CModAdd0", "CModAdd1", "CModSub0", "CModSub1")


def original(kind: str, width: int, inverse: bool = False) -> Any:
    word = QMontgomeryUInt(width)
    base = (
        ModAdd(width, _P)
        if kind == "ModAdd"
        else (
            ModSub(word, _P)
            if kind == "ModSub"
            else (
                ModDbl(word, _P)
                if kind == "ModDbl"
                else (CModAdd if kind.startswith("CModAdd") else CModSub)(
                    word, _P, int(kind[-1])
                )
            )
        )
    )
    return base.adjoint() if inverse else base


def native(kind: str, width: int, inverse: bool) -> ProgramRecord:
    dtype = montgomery_register_type(width, _P if kind == "ModAdd" else None)
    if kind == "ModDbl":
        return owners.symbolic_controlled_modular_doubling_program(
            dtype, _P, adjoint=inverse
        )
    callback = (
        owners.symbolic_additional_modular_addition_program
        if "Add" in kind
        else owners.symbolic_additional_modular_subtraction_program
    )
    if kind.startswith("C"):
        return callback(dtype, _P, control_value=int(kind[-1]), adjoint=inverse)
    return callback(dtype, _P, adjoint=inverse)


@pytest.mark.parametrize("kind", _CASES)
@pytest.mark.parametrize("width", (2, 3))
@pytest.mark.parametrize("inverse", (False, True))
def test_every_available_layer_operands_and_actual_leaf_refusals(
    kind: str, width: int, inverse: bool
) -> None:
    record = native(kind, width, inverse)
    available_layers(record, original(kind, width, inverse).controlled())
    assert len(record.functions[0].blocks[0].operations) == (
        15
        if kind == "ModDbl"
        else 17 if kind == "ModAdd" else 14 if kind.startswith("CModAdd") else 5
    )
    assert_unbound(record)


@pytest.mark.parametrize("kind", _CASES)
@pytest.mark.parametrize("depth", (0, 1))
def test_constructor_roundtrip_exact_inverse_and_sdk_free_resolution(
    kind: str, depth: int
) -> None:
    source = original(kind, 3, True).controlled()
    library = _library(source, depth)
    encoded = serialize_component_library(library)
    library = deserialize_component_library(encoded)
    assert serialize_component_library(library) == encoded
    factories = (
        modular_multiplication_factories()
        if kind == "ModDbl"
        else modular_control_factories()
    )
    denied = {
        name: None
        for name in sys.modules
        if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}
    }
    with patch.dict(sys.modules, denied):
        record = resolve_component_program(library, 0, factories)
    record = source_body_program(record, source)
    assert matches_component_definition(
        record, record.functions[0].name, native(kind, 3, True)
    )


@pytest.mark.parametrize("kind", _CASES)
def test_negative_outer_wrapper_keeps_original_three_calls(kind: str) -> None:
    source = original(kind, 3).controlled(CtrlSpec(cvs=0))
    factories = (
        modular_multiplication_factories()
        if kind == "ModDbl"
        else modular_control_factories()
    )
    record = resolve_component_program(_library(source, 0), 0, factories)
    record = source_body_program(record, source)
    calls = record.functions[0].blocks[0].operations
    assert len(calls) == 3
    compare_layers(record, source, {op.callee for op in calls if op.callee is not None})


def test_explicit_load_and_unload_roles_preserve_expression_and_direction() -> None:
    with patch.object(
        owners,
        "unresolved_constant_addition_program",
        wraps=unresolved_constant_addition_program,
    ) as observer:
        owners.symbolic_additional_modular_subtraction_program(
            montgomery_register_type(3), _P, control_value=0, adjoint=True
        )
    calls = observer.call_args_list
    loads = [
        call
        for call in calls
        if call.kwargs["name"] in {"load_complement", "unload_complement"}
    ]
    assert {call.args[1] for call in loads} == {_P + 1}
    assert all(
        bool(call.kwargs.get("adjoint", False))
        == (call.kwargs["name"] == "unload_complement")
        for call in loads
    )
    assert {
        call.args[1] for call in calls if call.kwargs["name"] == "subtract_modulus"
    } == {-_P}
    assert {
        call.args[1] for call in calls if call.kwargs["name"] == "restore_modulus"
    } == {_P}
    for inverse in (False, True):
        record = native("CModSub0", 3, inverse)
        pairs = available_layers(record, original("CModSub0", 3, inverse).controlled())
        witnessed = set()
        for name, child in pairs.items():
            reversed_constant = False
            while type(child).__name__ in {"Controlled", "Adjoint"}:
                reversed_constant ^= type(child).__name__ == "Adjoint"
                child = child.subbloq
            if type(child).__name__ != "AddK":
                continue
            if "unload_complement" in name:
                role, value, direction = "unload", _P + 1, True
            elif "load_complement" in name:
                role, value, direction = "load", _P + 1, False
            elif "subtract_modulus" in name:
                role, value, direction = "subtract", -_P, inverse
            elif "restore_modulus" in name:
                role, value, direction = "restore", _P, inverse
            else:
                raise AssertionError(
                    f"Unbound constant lacks its explicit role: {name}"
                )
            assert child.k == value and reversed_constant == direction
            witnessed.add(role)
        assert witnessed == {"load", "unload", "subtract", "restore"}


@pytest.mark.parametrize("cv", (0, 1))
def test_unsigned_controlled_addition_uses_the_same_shared_specialization(
    cv: int,
) -> None:
    record = owners.symbolic_additional_modular_addition_program(
        unsigned_register_type(3), _P, control_value=cv
    )
    available_layers(record, CModAdd(QUInt(3), _P, cv).controlled())
