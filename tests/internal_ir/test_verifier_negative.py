from __future__ import annotations

import ast
import inspect
from collections.abc import Iterable

import pytest

import flagquantum._compiler.ir.verifier as verifier_module
from flagquantum._compiler.diagnostics import DiagnosticCode
from flagquantum._compiler.ir.modules import Block, QuantumModule, Region
from flagquantum._compiler.ir.operations import Operation, SourceLocation
from flagquantum._compiler.ir.schemas import (
    OperationSchema,
    OperationSchemaRegistry,
    circuit_ir_v1_schema_registry,
)
from flagquantum._compiler.ir.types import BIT, QUBIT
from flagquantum._compiler.ir.values import ValueId, ValueRef
from flagquantum._compiler.ir.verifier import verify_module

pytestmark = pytest.mark.unit


def _q(index: int) -> ValueRef:
    return ValueRef(ValueId(index), QUBIT)


def _bit(index: int) -> ValueRef:
    return ValueRef(ValueId(index), BIT)


def _registry(*extra: OperationSchema) -> OperationSchemaRegistry:
    base = circuit_ir_v1_schema_registry()
    support = (
        OperationSchema(
            "quantum.measure",
            (QUBIT,),
            (BIT,),
            effects=("consume_linear", "measurement"),
        ),
        OperationSchema(
            "quantum.release",
            (QUBIT,),
            (),
            effects=("consume_linear", "release"),
        ),
        OperationSchema("control.return", (), (), effects=("terminator",)),
    )
    return OperationSchemaRegistry(tuple(base.values()) + support + extra)


def _module(
    operations: Iterable[Operation], *, arguments: tuple[ValueRef, ...] = (_q(0),)
) -> QuantumModule:
    return QuantumModule(Region((Block(arguments, tuple(operations)),)))


def _codes(module: QuantumModule, registry: OperationSchemaRegistry | None = None):
    result = verify_module(module, registry or _registry())
    return result, tuple(item.code for item in result.diagnostics)


def test_valid_linear_gate_chain_passes_without_output(capsys) -> None:
    module = _module(
        (
            Operation("quantum.h", (_q(0),), (_q(1),)),
            Operation("quantum.rx", (_q(1),), (_q(2),), {"theta": 0.5}),
            Operation("quantum.measure", (_q(2),), (_bit(3),)),
        )
    )

    result = verify_module(module, _registry())

    assert result.ok is True
    assert result.diagnostics == ()
    assert capsys.readouterr() == ("", "")
    result.require_valid()


def test_unknown_operation_fails_closed_with_source_location() -> None:
    location = SourceLocation("input.fq", 8, 3)
    result, codes = _codes(_module((Operation("test.unknown", location=location),)))

    assert codes == (DiagnosticCode.UNKNOWN_OPERATION,)
    assert result.diagnostics[0].location.to_dict() == {
        "source": "input.fq",
        "line": 8,
        "column": 3,
    }


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        ({}, DiagnosticCode.MISSING_ATTRIBUTE),
        ({"theta": 0.5, "secret": 1}, DiagnosticCode.UNKNOWN_ATTRIBUTE),
        ({"theta": "0.5"}, DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH),
    ],
)
def test_attribute_contract_fails_closed(attributes, expected) -> None:
    _, codes = _codes(
        _module((Operation("quantum.rx", (_q(0),), (_q(1),), attributes),))
    )

    assert expected in codes


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (Operation("quantum.x", (), (_q(1),)), DiagnosticCode.OPERAND_ARITY_MISMATCH),
        (Operation("quantum.x", (_q(0),), ()), DiagnosticCode.RESULT_ARITY_MISMATCH),
        (
            Operation(
                "quantum.custom_unitary",
                (_q(0),),
                (_q(1), _q(2)),
                {"symbolic_name": "u", "matrix": (1, 0, 0, 1), "arity": 1},
            ),
            DiagnosticCode.INVALID_VARIADIC_ARITY,
        ),
    ],
)
def test_arity_contract_is_verified(operation, expected) -> None:
    _, codes = _codes(_module((operation,)))

    assert expected in codes


def test_use_before_definition_is_rejected() -> None:
    _, codes = _codes(
        _module((Operation("quantum.x", (_q(99),), (_q(1),)),), arguments=())
    )

    assert DiagnosticCode.USE_BEFORE_DEFINITION in codes


def test_old_linear_value_and_duplicate_consumption_are_rejected() -> None:
    module = _module(
        (
            Operation("quantum.h", (_q(0),), (_q(1),)),
            Operation("quantum.x", (_q(0),), (_q(2),)),
        )
    )
    _, codes = _codes(module)

    assert DiagnosticCode.LINEAR_VALUE_REUSED in codes


def test_same_linear_value_cannot_fill_two_operands_of_one_gate() -> None:
    _, codes = _codes(
        _module((Operation("quantum.cx", (_q(0), _q(0)), (_q(1), _q(2))),))
    )

    assert DiagnosticCode.LINEAR_VALUE_REUSED in codes


def test_release_after_use_has_a_specific_diagnostic() -> None:
    module = _module(
        (
            Operation("quantum.release", (_q(0),)),
            Operation("quantum.x", (_q(0),), (_q(1),)),
        )
    )
    _, codes = _codes(module)

    assert DiagnosticCode.USE_AFTER_RELEASE in codes


def test_duplicate_definition_is_rejected() -> None:
    module = _module(
        (
            Operation("quantum.x", (_q(0),), (_q(1),)),
            Operation("quantum.h", (_q(1),), (_q(1),)),
        )
    )
    _, codes = _codes(module)

    assert DiagnosticCode.DUPLICATE_DEFINITION in codes


def test_duplicate_block_argument_is_rejected() -> None:
    _, codes = _codes(_module((), arguments=(_q(0), _q(0))))

    assert DiagnosticCode.DUPLICATE_DEFINITION in codes


def test_measurement_result_type_is_checked() -> None:
    _, codes = _codes(_module((Operation("quantum.measure", (_q(0),), (_q(1),)),)))

    assert DiagnosticCode.VALUE_TYPE_MISMATCH in codes


def test_operation_after_terminator_is_rejected() -> None:
    module = _module(
        (
            Operation("control.return"),
            Operation("quantum.x", (_q(0),), (_q(1),)),
        )
    )
    _, codes = _codes(module)

    assert DiagnosticCode.OPERATION_AFTER_TERMINATOR in codes


def test_region_count_is_checked() -> None:
    schema = OperationSchema("test.with_region", (), (), region_count=1)
    _, codes = _codes(_module((Operation("test.with_region"),)), _registry(schema))

    assert DiagnosticCode.REGION_COUNT_MISMATCH in codes


def test_require_valid_raises_only_after_structured_verification() -> None:
    result = verify_module(_module((Operation("test.unknown"),)), _registry())

    with pytest.raises(ValueError, match="IRV001"):
        result.require_valid()


def test_verifier_source_contains_no_print_calls() -> None:
    tree = ast.parse(inspect.getsource(verifier_module))

    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
        for node in ast.walk(tree)
    )
