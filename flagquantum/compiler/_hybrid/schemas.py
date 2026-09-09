"""Closed operation vocabulary for the first private hybrid IR profile."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperationSchema:
    """Structural limits validated before operation-specific semantics."""

    min_operands: int
    max_operands: int | None
    min_results: int
    max_results: int | None
    regions: int
    required_attributes: frozenset[str] = frozenset()
    optional_attributes: frozenset[str] = frozenset()
    terminator: bool = False


OPERATION_SCHEMAS = {
    "program.return": OperationSchema(1, 2, 0, 0, 0, terminator=True),
    "tensor.dim": OperationSchema(2, 2, 1, 1, 0),
    "tensor.extract": OperationSchema(2, None, 1, 1, 0),
    "arith.constant": OperationSchema(
        0, 0, 1, 1, 0, required_attributes=frozenset({"value"})
    ),
    "arith.add": OperationSchema(2, 2, 1, 1, 0),
    "arith.rem": OperationSchema(2, 2, 1, 1, 0),
    "arith.not": OperationSchema(1, 1, 1, 1, 0),
    "arith.and": OperationSchema(2, 2, 1, 1, 0),
    "arith.cmp": OperationSchema(
        2, 2, 1, 1, 0, required_attributes=frozenset({"predicate"})
    ),
    "scf.for": OperationSchema(4, None, 1, None, 1),
    "scf.if": OperationSchema(2, None, 1, None, 2),
    "scf.yield": OperationSchema(1, None, 0, 0, 0, terminator=True),
    "quantum.angle_embedding": OperationSchema(
        2, 2, 1, 1, 0, required_attributes=frozenset({"wires"})
    ),
    "quantum.rx": OperationSchema(3, 3, 1, 1, 0),
    "quantum.ry": OperationSchema(3, 3, 1, 1, 0),
    "quantum.cx": OperationSchema(3, 3, 1, 1, 0),
    "quantum.h": OperationSchema(2, 2, 1, 1, 0),
    "quantum.x": OperationSchema(2, 2, 1, 1, 0),
    "quantum.measure": OperationSchema(2, 2, 2, 2, 0),
    "quantum.expectation": OperationSchema(
        1, 1, 1, 1, 0, required_attributes=frozenset({"terms"})
    ),
}
