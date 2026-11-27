"""Unbound modular graphs preserve exact source constructors and typed edges."""

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import pytest
import sympy

from flagquantum.compiler.formats.legacy_component_arguments import (
    ComponentArgument,
    ComponentReference,
    ControlSpecification,
    serialize_component_argument,
)
from flagquantum.compiler.formats.legacy_component_library import (
    ComponentLibrary,
    ComponentOccurrence,
    LibraryBoundary,
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.symbolic_modular_library import (
    ModularGraphKind,
    modular_component_library,
)
from flagquantum.core.ir.program.register_types import (
    montgomery_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.value_math import SymbolicInt
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CASES: tuple[tuple[ModularGraphKind, int], ...] = (
    ("ModAdd", 1),
    ("CModAdd", 0),
    ("CModAdd", 1),
    ("CModAddK", 1),
    ("CtrlScaleModAdd", 1),
    ("ModNeg", 1),
    ("CModNeg", 0),
    ("CModNeg", 1),
    ("ModSub", 1),
    ("CModSub", 0),
    ("CModSub", 1),
)


def _canonical(library: ComponentLibrary) -> tuple[object, ...]:
    # Compare the two actual wire codecs at the same boundary. The original
    # Montgomery codec omits modulus metadata; raw native data is checked below.
    def dtype(value: RegisterType) -> RegisterType:
        return (
            replace(value, parameters=()) if value.encoding == "montgomery" else value
        )

    def register(value: Register) -> Register:
        return Register(
            value.name, dtype(value.dtype), value.shape_symbolic, value.side
        )

    def argument(value: ComponentArgument) -> ComponentArgument:
        if isinstance(value.value, RegisterType):
            return replace(value, value=dtype(value.value))
        if isinstance(value.value, ControlSpecification):
            return replace(
                value,
                value=ControlSpecification(
                    tuple((dtype(t), cvs) for t, cvs in value.value.clauses)
                ),
            )
        return value

    # Explicitly model the original lossy Montgomery wire for this comparison;
    # the actual owned library retains its modulus and is not changed.
    projected = replace(
        library,
        entries=tuple(
            replace(
                row,
                declaration=replace(
                    row.declaration,
                    registers=tuple(
                        register(r) for r in row.declaration.registers or ()
                    ),
                    arguments=tuple(argument(a) for a in row.declaration.arguments),
                ),
                connections=tuple(
                    replace(
                        edge,
                        source=replace(
                            edge.source, register=register(edge.source.register)
                        ),
                        destination=replace(
                            edge.destination,
                            register=register(edge.destination.register),
                        ),
                    )
                    for edge in row.connections
                ),
            )
            for row in library.entries
        ),
    )
    library = deserialize_component_library(serialize_component_library(projected))
    rows = {row.identifier: row for row in library.entries}

    def declaration(identifier: int) -> tuple[object, ...]:
        item = rows[identifier].declaration
        values: list[tuple[str, object]] = []
        for arg in item.arguments:
            value = (
                declaration(arg.value.identifier)
                if isinstance(arg.value, ComponentReference)
                else serialize_component_argument(arg)
            )
            values.append((arg.name, value))
        return item.name, tuple(sorted(values)), item.registers

    def owner(value: ComponentOccurrence | LibraryBoundary) -> object:
        if isinstance(value, LibraryBoundary):
            return value.value
        return value.instance_id, declaration(value.definition_id)

    edges: Iterable[object] = (
        (
            owner(edge.source.owner),
            edge.source.register,
            edge.source.index,
            owner(edge.destination.owner),
            edge.destination.register,
            edge.destination.index,
        )
        for edge in rows[0].connections
    )
    return declaration(0), tuple(sorted(edges, key=repr))


@pytest.mark.parametrize("kind,cv", CASES)
@pytest.mark.parametrize("width", (1, 2, 3, 4))
def test_complete_symbolic_source_roots_and_all_operand_connections(
    kind: ModularGraphKind,
    cv: int,
    width: SymbolicInt,
    montgomery: bool = False,
) -> None:
    from importlib import import_module

    from qualtran._infra.data_types import QMontgomeryUInt, QUInt

    p, k = sympy.symbols("p k", positive=True, integer=True)
    source_module = import_module(
        "qualtran.bloqs.mod_arithmetic."
        + (
            "mod_addition"
            if kind in ("ModAdd", "CModAdd", "CModAddK", "CtrlScaleModAdd")
            else "mod_subtraction"
        )
    )
    arguments: dict[str, Any] = {"mod": p}
    if kind in ("ModAdd", "CModAddK", "CtrlScaleModAdd"):
        arguments["bitsize"] = width
    else:
        arguments["dtype"] = QMontgomeryUInt(width) if montgomery else QUInt(width)
    if kind in ("CModAddK", "CtrlScaleModAdd"):
        arguments["k"] = 1 if kind == "CModAddK" else k
    if kind in ("CModAdd", "CModNeg", "CModSub"):
        arguments["cv"] = cv
    source = getattr(source_module, kind)(**arguments)
    native = modular_component_library(
        kind,
        (
            montgomery_register_type(width)
            if montgomery
            else unsigned_register_type(width)
        ),
        p,
        constant=1 if kind == "CModAddK" else k,
        control_value=cv,
    )
    assert _canonical(native) == _canonical(_library(source, 1))
    params = {arg.name: arg.value for arg in native.entries[0].declaration.arguments}
    assert params["mod"] == p
    raw_modulus = params["mod"]
    assert isinstance(raw_modulus, sympy.Symbol)
    assert raw_modulus.assumptions0 == p.assumptions0


@pytest.mark.parametrize(
    "kind,cv", (("CModAddK", 1), ("ModSub", 1), ("CModSub", 0), ("CModSub", 1))
)
def test_source_defined_symbolic_width_shells(kind: ModularGraphKind, cv: int) -> None:
    test_complete_symbolic_source_roots_and_all_operand_connections(
        kind, cv, sympy.Symbol("n", positive=True, integer=True)
    )


@pytest.mark.parametrize("kind,cv", CASES)
def test_native_library_consumes_existing_physical_owners_without_sdk(
    kind: ModularGraphKind,
    cv: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.compiler.formats.legacy_modular_addition_factories import (
        modular_addition_factories,
    )
    from flagquantum.compiler.formats.legacy_modular_subtraction_factories import (
        modular_subtraction_factories,
    )
    from tests.unit.test_modular_addition_component import bind_case
    from tests.unit.test_modular_addition_component import fixture as addition
    from tests.unit.test_modular_subtraction_component import fixture as subtraction

    add_kinds = {
        "ModAdd": "mod",
        "CModAdd": "unsigned",
        "CModAddK": "constant",
        "CtrlScaleModAdd": "scaled",
    }
    sub_kinds = {"ModNeg": "neg", "CModNeg": "cneg", "ModSub": "sub", "CModSub": "csub"}
    expected, _ = (
        addition(add_kinds[kind], 2, 2, 1, cv)
        if kind in add_kinds
        else subtraction(sub_kinds[kind], 2, 2, False, cv)
    )
    expected_bound, _, _ = bind_case(expected)
    for module in ("qualtran", "cirq", "kickmix"):
        monkeypatch.setitem(sys.modules, module, None)
    native = modular_component_library(
        kind, unsigned_register_type(2), 2, constant=1, control_value=cv
    )
    factories = modular_addition_factories() | modular_subtraction_factories()
    actual = resolve_component_program(native, 0, factories)
    bound, _, _ = bind_case(actual)
    assert bound.circuit == expected_bound.circuit
    assert bound.outputs == expected_bound.outputs
    assert bound.events == expected_bound.events
    symbolic = modular_component_library(
        kind, unsigned_register_type(2), sympy.Symbol("p"), constant=1, control_value=cv
    )
    declared = resolve_component_program(symbolic, 0, factories)
    with pytest.raises(ValueError, match="component bodies"):
        bind_case(declared)


@pytest.mark.parametrize("constant", (-1, 4))
def test_original_constant_loader_refusal_is_not_converted_to_a_graph(
    constant: int,
) -> None:
    from qualtran.bloqs.mod_arithmetic.mod_addition import CModAddK

    with pytest.raises(ValueError):
        CModAddK(k=constant, bitsize=2, mod=sympy.Symbol("p")).decompose_bloq()
    with pytest.raises(ValueError, match="fit"):
        modular_component_library(
            "CModAddK", unsigned_register_type(2), sympy.Symbol("p"), constant=constant
        )


@pytest.mark.parametrize(
    "kind,cv",
    tuple(
        (kind, cv)
        for kind, cv in CASES
        if kind in {"CModAdd", "ModNeg", "CModNeg", "ModSub", "CModSub"}
    ),
)
def test_original_montgomery_word_recipes(kind: ModularGraphKind, cv: int) -> None:
    test_complete_symbolic_source_roots_and_all_operand_connections(kind, cv, 3, True)


def test_live_montgomery_metadata_is_preserved_and_not_silently_written_lossily() -> (
    None
):
    from flagquantum.errors import SerializationError

    p = sympy.Symbol("p", integer=True, positive=True)
    library = modular_component_library("ModAdd", unsigned_register_type(3), p)
    assert library.entries[0].declaration.registers is not None
    assert (
        dict(library.entries[0].declaration.registers[0].dtype.parameters)["modulus"]
        == p
    )
    assert not library.entries[0].counts
    with pytest.raises(SerializationError, match="cannot retain a Montgomery modulus"):
        serialize_component_library(library)
