"""Fail-closed verifier for the first private hybrid program profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .model import (
    BOOL,
    INDEX,
    QUANTUM_EFFECT,
    Block,
    HybridProgram,
    IRType,
    Operation,
    Value,
    ValueId,
)
from .schemas import OPERATION_SCHEMAS, OperationSchema


@dataclass(frozen=True)
class Diagnostic:
    """One deterministic verification failure."""

    code: str
    message: str


class HybridVerificationError(ValueError):
    """Raised when a private hybrid program violates its semantic profile."""

    def __init__(self, diagnostics: tuple[Diagnostic, ...]) -> None:
        self.diagnostics = diagnostics
        summary = "; ".join(f"{item.code}: {item.message}" for item in diagnostics)
        super().__init__(summary)


def _arity_ok(count: int, minimum: int, maximum: int | None) -> bool:
    return count >= minimum and (maximum is None or count <= maximum)


def _is_scalar(value_type: IRType) -> bool:
    return value_type.kind in {"scalar", "index"}


class _Verifier:
    def __init__(self) -> None:
        self.diagnostics: list[Diagnostic] = []
        self.definitions: set[ValueId] = set()

    def error(self, code: str, message: str) -> None:
        self.diagnostics.append(Diagnostic(code, message))

    def verify(self, program: HybridProgram) -> None:
        if len(program.body.blocks) != 1:
            self.error("program.block_count", "first profile requires one entry block")
            return
        self._verify_block(program.body.blocks[0], {}, "program.return")

    def _define(self, value: Value, environment: dict[ValueId, IRType]) -> None:
        if value.id in self.definitions:
            self.error("ssa.duplicate_definition", f"duplicate definition {value.id}")
            return
        self.definitions.add(value.id)
        environment[value.id] = value.type

    def _verify_block(
        self,
        block: Block,
        inherited: Mapping[ValueId, IRType],
        expected_terminator: str,
    ) -> None:
        environment = dict(inherited)
        live_linear: set[ValueId] = set()
        for argument in block.arguments:
            self._define(argument, environment)
            if argument.type.linear:
                live_linear.add(argument.id)

        if not block.operations or block.operations[-1].name != expected_terminator:
            self.error(
                "block.terminator",
                f"block must end with {expected_terminator!r}",
            )

        for index, operation in enumerate(block.operations):
            schema = OPERATION_SCHEMAS.get(operation.name)
            if schema is None:
                self.error(
                    "operation.unknown", f"unsupported operation {operation.name!r}"
                )
                continue
            if schema.terminator and index != len(block.operations) - 1:
                self.error(
                    "operation.terminator_position",
                    f"terminator {operation.name!r} must be the final operation",
                )
            self._verify_structure(operation, schema)
            for operand in operation.operands:
                defined_type = environment.get(operand.id)
                if defined_type is None:
                    self.error(
                        "ssa.use_before_definition", f"undefined value {operand.id}"
                    )
                    continue
                if operand.type != defined_type:
                    self.error("ssa.use_type", f"type mismatch for value {operand.id}")
                if defined_type.linear:
                    if operand.id not in live_linear:
                        self.error(
                            "effect.reuse",
                            f"linear value {operand.id} is unavailable or already consumed",
                        )
                    else:
                        live_linear.remove(operand.id)

            self._verify_operation_semantics(operation)
            visible_non_linear = {
                value_id: value_type
                for value_id, value_type in environment.items()
                if not value_type.linear
            }
            for region in operation.regions:
                if len(region.blocks) != 1:
                    self.error(
                        "region.block_count",
                        f"{operation.name!r} region requires exactly one block",
                    )
                    continue
                self._verify_block(region.blocks[0], visible_non_linear, "scf.yield")

            for result in operation.results:
                self._define(result, environment)
                if result.type.linear:
                    live_linear.add(result.id)

        for value_id in sorted(live_linear):
            self.error("effect.dropped", f"linear value {value_id} was not consumed")

    def _verify_structure(self, operation: Operation, schema: OperationSchema) -> None:
        if not _arity_ok(
            len(operation.operands), schema.min_operands, schema.max_operands
        ):
            self.error(
                "operation.operand_arity",
                f"{operation.name!r} has {len(operation.operands)} operand(s)",
            )
        if not _arity_ok(
            len(operation.results), schema.min_results, schema.max_results
        ):
            self.error(
                "operation.result_arity",
                f"{operation.name!r} has {len(operation.results)} result(s)",
            )
        if len(operation.regions) != schema.regions:
            self.error(
                "operation.region_arity",
                f"{operation.name!r} requires {schema.regions} region(s)",
            )
        keys = set(operation.attributes)
        missing = schema.required_attributes - keys
        unknown = keys - schema.required_attributes - schema.optional_attributes
        if missing:
            self.error(
                "operation.attribute_missing",
                f"{operation.name!r} is missing {sorted(missing)}",
            )
        if unknown:
            self.error(
                "operation.attribute_unknown",
                f"{operation.name!r} has unsupported attributes {sorted(unknown)}",
            )

    def _verify_operation_semantics(self, operation: Operation) -> None:
        operands = operation.operands
        results = operation.results
        name = operation.name

        if name == "program.return":
            if operands and (
                not _is_scalar(operands[0].type) or operands[0].type.linear
            ):
                self.error(
                    "program.return_type", "program must return one scalar value"
                )
        elif name == "arith.constant" and len(results) == 1:
            self._verify_constant(operation)
        elif name == "tensor.dim" and len(operands) == 2 and len(results) == 1:
            if (
                operands[0].type.kind != "tensor"
                or operands[1].type != INDEX
                or results[0].type != INDEX
            ):
                self.error(
                    "tensor.dim_type", "tensor.dim requires tensor, index -> index"
                )
        elif name == "tensor.extract" and len(operands) >= 2 and len(results) == 1:
            self._verify_tensor_extract(operation)
        elif (
            name in {"arith.add", "arith.rem"}
            and len(operands) == 2
            and len(results) == 1
        ):
            if (
                operands[0].type != operands[1].type
                or results[0].type != operands[0].type
                or not _is_scalar(operands[0].type)
            ):
                self.error(
                    "arith.type", f"{name} requires equal scalar operand/result types"
                )
        elif name == "arith.cmp" and len(operands) == 2 and len(results) == 1:
            predicates = {"eq", "ne", "lt", "le", "gt", "ge"}
            if operation.attributes.get("predicate") not in predicates:
                self.error("arith.predicate", "arith.cmp predicate is unsupported")
            if (
                operands[0].type != operands[1].type
                or not _is_scalar(operands[0].type)
                or results[0].type != BOOL
            ):
                self.error(
                    "arith.cmp_type",
                    "arith.cmp requires equal scalars and returns bool",
                )
        elif name == "scf.if":
            self._verify_if(operation)
        elif name == "scf.for":
            self._verify_for(operation)
        elif name == "scf.yield":
            return
        elif name == "quantum.angle_embedding":
            self._verify_angle_embedding(operation)
        elif (
            name in {"quantum.rx", "quantum.ry"}
            and len(operands) == 3
            and len(results) == 1
        ):
            if (
                not _is_scalar(operands[0].type)
                or operands[1].type != INDEX
                or operands[2].type != QUANTUM_EFFECT
                or results[0].type != QUANTUM_EFFECT
            ):
                self.error(
                    "quantum.rotation_type",
                    f"{name} requires scalar, index, effect -> effect",
                )
        elif name == "quantum.cx" and len(operands) == 3 and len(results) == 1:
            if (
                operands[0].type != INDEX
                or operands[1].type != INDEX
                or operands[2].type != QUANTUM_EFFECT
                or results[0].type != QUANTUM_EFFECT
            ):
                self.error(
                    "quantum.cx_type",
                    "quantum.cx requires index, index, effect -> effect",
                )
        elif name == "quantum.expectation" and len(operands) == 1 and len(results) == 1:
            if operands[0].type != QUANTUM_EFFECT or not _is_scalar(results[0].type):
                self.error(
                    "quantum.expectation_type", "expectation requires effect -> scalar"
                )
            terms = operation.attributes.get("terms")
            if (
                not isinstance(terms, tuple)
                or not terms
                or any(
                    not isinstance(term, tuple)
                    or len(term) != 2
                    or not isinstance(term[0], str)
                    or not isinstance(term[1], int)
                    or term[1] < 0
                    for term in terms
                )
            ):
                self.error(
                    "quantum.observable",
                    "expectation terms must contain observable names and non-negative wires",
                )

    def _verify_constant(self, operation: Operation) -> None:
        result_type = operation.results[0].type
        constant = operation.attributes.get("value")
        if result_type == BOOL and not isinstance(constant, bool):
            self.error("arith.constant_type", "bool constant requires a bool value")
        elif result_type == INDEX and (
            not isinstance(constant, int) or isinstance(constant, bool)
        ):
            self.error("arith.constant_type", "index constant requires an int value")
        elif result_type.kind == "scalar" and (
            not isinstance(constant, (int, float)) or isinstance(constant, bool)
        ):
            self.error(
                "arith.constant_type", "scalar constant requires a numeric value"
            )
        elif result_type not in {BOOL, INDEX} and result_type.kind != "scalar":
            self.error(
                "arith.constant_type", "first profile constants must be bool or scalar"
            )

    def _verify_tensor_extract(self, operation: Operation) -> None:
        tensor = operation.operands[0]
        indices = operation.operands[1:]
        result = operation.results[0]
        if tensor.type.kind != "tensor" or any(
            value.type != INDEX for value in indices
        ):
            self.error(
                "tensor.extract_type",
                "tensor.extract requires a tensor and index operands",
            )
            return
        dtype, shape = tensor.type.parameters
        if len(indices) != len(shape) or result.type != IRType("scalar", (dtype,)):
            self.error(
                "tensor.extract_result", "tensor.extract rank or scalar dtype mismatch"
            )

    def _verify_if(self, operation: Operation) -> None:
        if len(operation.operands) < 2 or not operation.results:
            return
        if operation.operands[0].type != BOOL:
            self.error("scf.if_condition", "scf.if condition must be bool")
        carried = tuple(value.type for value in operation.operands[1:])
        result_types = tuple(value.type for value in operation.results)
        if carried != result_types:
            self.error("scf.if_result", "scf.if carried and result types must match")
        for region in operation.regions:
            if len(region.blocks) != 1:
                continue
            block = region.blocks[0]
            if tuple(value.type for value in block.arguments) != carried:
                self.error(
                    "scf.if_arguments",
                    "scf.if branch arguments must match carried types",
                )
            self._verify_yield_signature(block, result_types, "scf.if")

    def _verify_for(self, operation: Operation) -> None:
        if len(operation.operands) < 4 or not operation.results:
            return
        if any(value.type != INDEX for value in operation.operands[:3]):
            self.error("scf.for_bounds", "scf.for lower, upper, and step must be index")
        carried = tuple(value.type for value in operation.operands[3:])
        result_types = tuple(value.type for value in operation.results)
        if carried != result_types:
            self.error("scf.for_result", "scf.for carried and result types must match")
        if len(operation.regions) != 1 or len(operation.regions[0].blocks) != 1:
            return
        block = operation.regions[0].blocks[0]
        expected_arguments = (INDEX, *carried)
        if tuple(value.type for value in block.arguments) != expected_arguments:
            self.error(
                "scf.for_arguments",
                "scf.for body arguments must be index plus carried types",
            )
        self._verify_yield_signature(block, result_types, "scf.for")

    def _verify_yield_signature(
        self, block: Block, expected: tuple[IRType, ...], owner: str
    ) -> None:
        if not block.operations or block.operations[-1].name != "scf.yield":
            return
        actual = tuple(value.type for value in block.operations[-1].operands)
        if actual != expected:
            self.error(
                "scf.yield_signature", f"{owner} yield types must match result types"
            )

    def _verify_angle_embedding(self, operation: Operation) -> None:
        if len(operation.operands) != 2 or len(operation.results) != 1:
            return
        if (
            operation.operands[0].type.kind != "tensor"
            or operation.operands[1].type != QUANTUM_EFFECT
            or operation.results[0].type != QUANTUM_EFFECT
        ):
            self.error(
                "quantum.embedding_type",
                "angle_embedding requires tensor, effect -> effect",
            )
        wires = operation.attributes.get("wires")
        if (
            not isinstance(wires, tuple)
            or not wires
            or any(not isinstance(wire, int) or wire < 0 for wire in wires)
            or len(set(wires)) != len(wires)
        ):
            self.error(
                "quantum.wires", "angle_embedding requires unique non-negative wires"
            )


def verify_program(program: HybridProgram) -> HybridProgram:
    """Validate a private hybrid program and return it unchanged."""

    verifier = _Verifier()
    verifier.verify(program)
    if verifier.diagnostics:
        raise HybridVerificationError(tuple(verifier.diagnostics))
    return program
