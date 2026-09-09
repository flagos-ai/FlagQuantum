from __future__ import annotations

from dataclasses import replace

import pytest

from flagquantum.compiler._hybrid import (
    BOOL,
    INDEX,
    QUANTUM_EFFECT,
    Block,
    HybridProgram,
    Operation,
    Region,
    SourceLocation,
    Value,
    ValueId,
    scalar_type,
    tensor_type,
    verify_program,
)

pytestmark = pytest.mark.unit

F64 = scalar_type("float64")


class Values:
    def __init__(self) -> None:
        self._next: dict[str, int] = {}

    def new(self, scope: str, value_type) -> Value:
        index = self._next.get(scope, 0)
        self._next[scope] = index + 1
        return Value(ValueId(scope, index), value_type)


def _yield(value: Value) -> Operation:
    return Operation("scf.yield", operands=(value,))


def _branch_rx(values: Values, parameter: Value, wire: Value) -> Region:
    effect = values.new("rx", QUANTUM_EFFECT)
    result = values.new("rx", QUANTUM_EFFECT)
    return Region(
        (
            Block(
                arguments=(effect,),
                operations=(
                    Operation(
                        "quantum.rx",
                        operands=(parameter, wire, effect),
                        results=(result,),
                    ),
                    _yield(result),
                ),
            ),
        )
    )


def _branch_ry_or_noop(
    values: Values,
    predicate: Value,
    parameter: Value,
    wire: Value,
) -> Region:
    outer_effect = values.new("rx_else", QUANTUM_EFFECT)
    ry_effect = values.new("ry", QUANTUM_EFFECT)
    ry_result = values.new("ry", QUANTUM_EFFECT)
    noop_effect = values.new("noop", QUANTUM_EFFECT)
    nested_result = values.new("rx_else", QUANTUM_EFFECT)
    nested = Operation(
        "scf.if",
        operands=(predicate, outer_effect),
        results=(nested_result,),
        regions=(
            Region(
                (
                    Block(
                        arguments=(ry_effect,),
                        operations=(
                            Operation(
                                "quantum.ry",
                                operands=(parameter, wire, ry_effect),
                                results=(ry_result,),
                            ),
                            _yield(ry_result),
                        ),
                    ),
                )
            ),
            Region(
                (Block(arguments=(noop_effect,), operations=(_yield(noop_effect),)),)
            ),
        ),
    )
    return Region(
        (
            Block(
                arguments=(outer_effect,),
                operations=(nested, _yield(nested_result)),
            ),
        )
    )


def build_nested_control_program() -> HybridProgram:
    """Build the bounded nested-loop and data-dependent-gate golden scenario."""

    values = Values()
    weights = values.new("entry", tensor_type("float64", (None, 4)))
    data = values.new("entry", tensor_type("float64", (4,)))
    entry_effect = values.new("entry", QUANTUM_EFFECT)
    zero_index = values.new("entry", INDEX)
    one_index = values.new("entry", INDEX)
    four_index = values.new("entry", INDEX)
    layer_count = values.new("entry", INDEX)
    zero_scalar = values.new("entry", F64)
    embedded_effect = values.new("entry", QUANTUM_EFFECT)

    layer_index = values.new("layer", INDEX)
    layer_effect = values.new("layer", QUANTUM_EFFECT)
    gate_index = values.new("gate", INDEX)
    gate_effect = values.new("gate", QUANTUM_EFFECT)
    parameter = values.new("gate", F64)
    positive = values.new("gate", BOOL)
    negative = values.new("gate", BOOL)
    selected_effect = values.new("gate", QUANTUM_EFFECT)
    gate_loop_effect = values.new("layer", QUANTUM_EFFECT)

    gate_body = Block(
        arguments=(gate_index, gate_effect),
        operations=(
            Operation(
                "tensor.extract",
                operands=(weights, layer_index, gate_index),
                results=(parameter,),
            ),
            Operation(
                "arith.cmp",
                operands=(parameter, zero_scalar),
                results=(positive,),
                attributes={"predicate": "gt"},
            ),
            Operation(
                "arith.cmp",
                operands=(parameter, zero_scalar),
                results=(negative,),
                attributes={"predicate": "lt"},
            ),
            Operation(
                "scf.if",
                operands=(positive, gate_effect),
                results=(selected_effect,),
                regions=(
                    _branch_rx(values, parameter, gate_index),
                    _branch_ry_or_noop(values, negative, parameter, gate_index),
                ),
            ),
            _yield(selected_effect),
        ),
    )
    gate_loop = Operation(
        "scf.for",
        operands=(zero_index, four_index, one_index, layer_effect),
        results=(gate_loop_effect,),
        regions=(Region((gate_body,)),),
    )

    cx_index = values.new("cx", INDEX)
    cx_effect = values.new("cx", QUANTUM_EFFECT)
    next_index = values.new("cx", INDEX)
    target_index = values.new("cx", INDEX)
    cx_result = values.new("cx", QUANTUM_EFFECT)
    cx_loop_effect = values.new("layer", QUANTUM_EFFECT)
    cx_body = Block(
        arguments=(cx_index, cx_effect),
        operations=(
            Operation(
                "arith.add",
                operands=(cx_index, one_index),
                results=(next_index,),
            ),
            Operation(
                "arith.rem",
                operands=(next_index, four_index),
                results=(target_index,),
            ),
            Operation(
                "quantum.cx",
                operands=(cx_index, target_index, cx_effect),
                results=(cx_result,),
            ),
            _yield(cx_result),
        ),
    )
    cx_loop = Operation(
        "scf.for",
        operands=(zero_index, four_index, one_index, gate_loop_effect),
        results=(cx_loop_effect,),
        regions=(Region((cx_body,)),),
    )

    completed_effect = values.new("entry", QUANTUM_EFFECT)
    layer_body = Block(
        arguments=(layer_index, layer_effect),
        operations=(gate_loop, cx_loop, _yield(cx_loop_effect)),
    )
    layer_loop = Operation(
        "scf.for",
        operands=(zero_index, layer_count, one_index, embedded_effect),
        results=(completed_effect,),
        regions=(Region((layer_body,)),),
    )
    expectation = values.new("entry", F64)
    entry = Block(
        arguments=(weights, data, entry_effect),
        operations=(
            Operation("arith.constant", results=(zero_index,), attributes={"value": 0}),
            Operation("arith.constant", results=(one_index,), attributes={"value": 1}),
            Operation("arith.constant", results=(four_index,), attributes={"value": 4}),
            Operation(
                "arith.constant", results=(zero_scalar,), attributes={"value": 0.0}
            ),
            Operation(
                "tensor.dim",
                operands=(weights, zero_index),
                results=(layer_count,),
            ),
            Operation(
                "quantum.angle_embedding",
                operands=(data, entry_effect),
                results=(embedded_effect,),
                attributes={"wires": (0, 1, 2, 3)},
            ),
            layer_loop,
            Operation(
                "quantum.expectation",
                operands=(completed_effect,),
                results=(expectation,),
                attributes={"terms": (("z", 0), ("z", 3))},
            ),
            Operation("program.return", operands=(expectation,)),
        ),
    )
    return HybridProgram("cost", Region((entry,)))


def test_nested_control_program_is_representable_and_verified() -> None:
    program = build_nested_control_program()

    assert verify_program(program) is program
    assert len(program.semantic_identity) == 64


def test_semantic_identity_is_deterministic_and_ignores_source_location() -> None:
    first = build_nested_control_program()
    second = build_nested_control_program()
    entry = second.body.blocks[0]
    operations = list(entry.operations)
    operations[0] = replace(operations[0], location=SourceLocation("moved.py", 200, 7))
    relocated = replace(
        second,
        body=Region((replace(entry, operations=tuple(operations)),)),
        location=SourceLocation("moved.py", 100, 3),
    )

    assert first.semantic_identity == second.semantic_identity
    assert first.semantic_identity == relocated.semantic_identity


def test_attributes_are_immutable() -> None:
    operation = build_nested_control_program().body.blocks[0].operations[0]

    with pytest.raises(TypeError):
        operation.attributes["value"] = 7  # type: ignore[index]
