from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from flagquantum._compiler.ir.modules import Block, QuantumModule, Region
from flagquantum._compiler.ir.operations import (
    FrozenAttributes,
    Operation,
    SourceLocation,
)
from flagquantum._compiler.ir.types import QUBIT
from flagquantum._compiler.ir.values import ValueId, ValueRef

pytestmark = pytest.mark.unit


def _module(*, angle: float = 0.5, location: SourceLocation | None = None):
    before = ValueRef(ValueId(0), QUBIT)
    after = ValueRef(ValueId(1), QUBIT)
    operation = Operation(
        "quantum.rx",
        operands=(before,),
        results=(after,),
        attributes={"theta": angle, "tags": ["semantic"]},
        location=location,
    )
    return QuantumModule(Region((Block(arguments=(before,), operations=(operation,)),)))


def test_module_and_nested_attributes_are_immutable() -> None:
    module = _module()
    attributes = module.body.blocks[0].operations[0].attributes

    assert isinstance(attributes, FrozenAttributes)
    assert attributes["tags"] == ("semantic",)
    with pytest.raises(TypeError):
        attributes["theta"] = 1.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        module.revision = 2  # type: ignore[misc]


def test_identity_is_deterministic_and_excludes_location_and_revision() -> None:
    first = _module(location=SourceLocation("a.py", 1, 2))
    moved = _module(location=SourceLocation("b.py", 50, 4))
    revised = QuantumModule(first.body, revision=9)

    assert first.program_identity == moved.program_identity
    assert first.program_identity == revised.program_identity
    assert len(first.program_identity) == 64


def test_semantic_change_changes_program_identity() -> None:
    assert _module(angle=0.5).program_identity != _module(angle=0.75).program_identity


def test_canonical_scalar_encoding_distinguishes_bool_int_float_and_complex() -> None:
    values = Operation(
        "test.constants",
        attributes={"bool": True, "int": 1, "float": 1.0, "complex": 1 + 0j},
    ).canonical()["attributes"]

    assert values["bool"]["kind"] == "bool"
    assert values["int"]["kind"] == "int"
    assert values["float"]["kind"] == "float"
    assert values["complex"]["kind"] == "complex"


def test_invalid_container_and_operation_names_fail_closed() -> None:
    with pytest.raises(ValueError, match="qualified"):
        Operation("rx")
    with pytest.raises(TypeError, match="unsupported immutable"):
        Operation("test.bad", attributes={"object": object()})
    with pytest.raises(TypeError, match="unordered set"):
        Operation("test.bad", attributes={"items": {1, 2}})
    with pytest.raises(ValueError, match="must be finite"):
        Operation("test.bad", attributes={"value": float("nan")})
    with pytest.raises(ValueError, match="at least one block"):
        Region(())
