from __future__ import annotations

import pytest

from flagquantum._compiler.ir.schemas import (
    AttributeSpec,
    OperationSchema,
    OperationSchemaRegistry,
    SchemaVersionError,
    UnknownOperationError,
    circuit_ir_v1_schema_registry,
)
from flagquantum._compiler.ir.types import QUBIT
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS

pytestmark = pytest.mark.unit


def test_static_profile_has_all_canonical_opcodes_and_custom_unitary() -> None:
    registry = circuit_ir_v1_schema_registry()

    assert len(registry) == len(OPERATOR_SCHEMAS) + 1
    assert tuple(registry) == tuple(sorted(registry))
    for opcode, public_schema in OPERATOR_SCHEMAS.items():
        schema = registry[f"quantum.{opcode}"]
        assert schema.operands == (QUBIT,) * public_schema.arity
        assert schema.results == (QUBIT,) * public_schema.arity
        assert schema.reversible_to_circuit_ir_v1 is True
        assert "consume_linear" in schema.effects

    custom = registry["quantum.custom_unitary"]
    assert custom.variadic_qubit_arity is True


def test_registry_fails_closed_for_unknown_operation_and_version() -> None:
    registry = circuit_ir_v1_schema_registry()

    with pytest.raises(UnknownOperationError, match="unknown internal operation"):
        registry["quantum.not_registered"]
    with pytest.raises(SchemaVersionError, match="unsupported registry version"):
        OperationSchemaRegistry((), version="2.0")


def test_schema_declaration_rejects_unknown_version_and_duplicate_attributes() -> None:
    with pytest.raises(SchemaVersionError, match="unsupported schema version"):
        OperationSchema("test.op", (), (), version="2.0")
    with pytest.raises(ValueError, match="unique"):
        OperationSchema(
            "test.op",
            (),
            (),
            attributes=(AttributeSpec("x", ("int",)), AttributeSpec("x", ("float",))),
        )


def test_registry_is_immutable_by_construction() -> None:
    registry = circuit_ir_v1_schema_registry()

    with pytest.raises(TypeError):
        registry["quantum.x"] = registry["quantum.h"]  # type: ignore[index]
