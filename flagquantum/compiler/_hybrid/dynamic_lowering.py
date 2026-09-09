"""Lower the bounded measurement-feedback profile to Core CircuitIR."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, NoReturn, Sequence

from ...core.ir import CircuitIR, Instruction
from .model import Block, HybridProgram, Operation, Region, Value, ValueId
from .specialize import SpecializationError
from .verifier import verify_program


@dataclass(frozen=True)
class LoweredDynamicProgram:
    """Core handoff for one private measurement-feedback program."""

    circuit: CircuitIR
    program_identity: str
    input_signature_identity: str
    measurement_count: int
    return_classical_bit: int
    stochastic_gradient_policy: str = "unsupported_fail_closed"


@dataclass(frozen=True)
class _MeasurementRef:
    classical_bit: int


_EFFECT = object()


class _DynamicLowerer:
    def __init__(self) -> None:
        self.instructions: list[Instruction] = []
        self.measurement_count = 0
        self.return_classical_bit: int | None = None

    def fail(self, operation: Operation, code: str, message: str) -> NoReturn:
        raise SpecializationError(code, message, operation.location)

    def resolve(self, value: Value, environment: Mapping[ValueId, Any]) -> Any:
        try:
            return environment[value.id]
        except KeyError as exc:
            raise SpecializationError(
                "ssa.runtime_value", f"runtime value is missing for {value.id}"
            ) from exc

    def execute_block(
        self,
        block: Block,
        outer: Mapping[ValueId, Any],
        arguments: Sequence[Any],
        *,
        conditions: tuple[tuple[int, int], ...] = (),
    ) -> tuple[Any, ...]:
        if len(block.arguments) != len(arguments):
            raise SpecializationError(
                "region.arguments", "runtime region argument arity mismatch"
            )
        environment = dict(outer)
        environment.update(
            (reference.id, runtime)
            for reference, runtime in zip(block.arguments, arguments)
        )
        for operation in block.operations:
            operands = tuple(
                self.resolve(reference, environment) for reference in operation.operands
            )
            results = self.execute_operation(
                operation, operands, environment, conditions=conditions
            )
            if operation.name in {"scf.yield", "program.return"}:
                return results
            if len(results) != len(operation.results):
                self.fail(
                    operation,
                    "operation.runtime_results",
                    "runtime result arity differs from verified IR",
                )
            environment.update(
                (reference.id, runtime)
                for reference, runtime in zip(operation.results, results)
            )
        raise SpecializationError(
            "block.terminator", "verified block did not terminate"
        )

    def execute_operation(
        self,
        operation: Operation,
        operands: tuple[Any, ...],
        environment: Mapping[ValueId, Any],
        *,
        conditions: tuple[tuple[int, int], ...],
    ) -> tuple[Any, ...]:
        name = operation.name
        if name == "arith.constant":
            return (operation.attributes["value"],)
        if name in {"quantum.h", "quantum.x"}:
            self.require_effect(operation, operands[1])
            wire = self.as_index(operation, operands[0])
            self.append_gate(name.removeprefix("quantum."), (wire,), conditions)
            return (_EFFECT,)
        if name == "quantum.cx":
            self.require_effect(operation, operands[2])
            wires = tuple(self.as_index(operation, value) for value in operands[:2])
            if wires[0] == wires[1]:
                self.fail(operation, "quantum.wires", "CX wires must be distinct")
            self.append_gate("cx", wires, conditions)
            return (_EFFECT,)
        if name == "quantum.measure":
            self.require_effect(operation, operands[1])
            if conditions:
                self.fail(
                    operation,
                    "dynamic.conditional_measurement",
                    "the first dynamic profile forbids measurement inside a branch",
                )
            wire = self.as_index(operation, operands[0])
            classical_bit = self.measurement_count
            self.measurement_count += 1
            self.instructions.append(
                Instruction(
                    "measure",
                    (wire,),
                    metadata={
                        "is_dynamic": True,
                        "classical_bit": classical_bit,
                    },
                )
            )
            return (_MeasurementRef(classical_bit), _EFFECT)
        if name == "scf.if":
            predicate = operands[0]
            if not isinstance(predicate, _MeasurementRef):
                self.fail(
                    operation,
                    "dynamic.condition",
                    "dynamic lowering requires a direct measurement bool condition",
                )
            if any(bit == predicate.classical_bit for bit, _ in conditions):
                self.fail(
                    operation,
                    "dynamic.condition",
                    "retesting the same measurement in a nested branch is unsupported",
                )
            then_result = self.execute_region(
                operation.regions[0],
                environment,
                operands[1:],
                conditions=(*conditions, (predicate.classical_bit, 1)),
            )
            else_result = self.execute_region(
                operation.regions[1],
                environment,
                operands[1:],
                conditions=(*conditions, (predicate.classical_bit, 0)),
            )
            if then_result != else_result or then_result != (_EFFECT,):
                self.fail(
                    operation,
                    "dynamic.branch_effect",
                    "dynamic branches must yield only the quantum effect",
                )
            return (_EFFECT,)
        if name == "scf.yield":
            return operands
        if name == "program.return":
            if len(operands) != 2 or not isinstance(operands[0], _MeasurementRef):
                self.fail(
                    operation,
                    "dynamic.return",
                    "dynamic program must return one measurement bool",
                )
            self.require_effect(operation, operands[1])
            self.return_classical_bit = operands[0].classical_bit
            return operands
        self.fail(
            operation,
            "dynamic.operation_unsupported",
            f"dynamic profile cannot lower {name!r}",
        )

    def execute_region(
        self,
        region: Region,
        outer: Mapping[ValueId, Any],
        arguments: Sequence[Any],
        *,
        conditions: tuple[tuple[int, int], ...],
    ) -> tuple[Any, ...]:
        if len(region.blocks) != 1:
            raise SpecializationError(
                "region.block_count", "dynamic profile requires one region block"
            )
        return self.execute_block(
            region.blocks[0], outer, arguments, conditions=conditions
        )

    def append_gate(
        self,
        name: str,
        wires: tuple[int, ...],
        conditions: tuple[tuple[int, int], ...],
    ) -> None:
        metadata = {} if not conditions else {"conditions": tuple(sorted(conditions))}
        self.instructions.append(Instruction(name, wires, metadata=metadata))

    def as_index(self, operation: Operation, value: Any) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            self.fail(
                operation,
                "index.value",
                "dynamic wires must resolve to non-negative integer constants",
            )
        return value

    def require_effect(self, operation: Operation, value: Any) -> None:
        if value is not _EFFECT:
            self.fail(operation, "quantum.effect", "runtime quantum effect is invalid")


def lower_dynamic_program(
    program: HybridProgram,
    inputs: Sequence[Any] = (),
    *,
    circuit_dtype: str = "complex64",
) -> LoweredDynamicProgram:
    """Lower a no-input measurement-feedback program without executing it."""

    verify_program(program)
    if len(inputs) != 0:
        raise SpecializationError(
            "dynamic.input_profile",
            "the first dynamic profile accepts no runtime inputs",
        )
    entry = program.body.blocks[0]
    if len(entry.arguments) != 1:
        raise SpecializationError(
            "dynamic.input_profile",
            "the first dynamic profile accepts no declared inputs",
        )
    if circuit_dtype not in {"complex64", "complex128"}:
        raise ValueError("dynamic circuit_dtype must be complex64 or complex128")
    lowerer = _DynamicLowerer()
    lowerer.execute_block(entry, {}, (_EFFECT,))
    if not lowerer.instructions or lowerer.measurement_count == 0:
        raise SpecializationError(
            "dynamic.measurement", "dynamic program produced no measurement"
        )
    if lowerer.return_classical_bit is None:
        raise SpecializationError(
            "dynamic.return", "dynamic program did not return a measurement bool"
        )
    referenced_wires = [
        wire for instruction in lowerer.instructions for wire in instruction.wires
    ]
    circuit = CircuitIR(
        n_wires=max(referenced_wires) + 1,
        instructions=tuple(lowerer.instructions),
        dtype=circuit_dtype,
        metadata={
            "hybrid_dynamic_session": True,
            "hybrid_dynamic_return_bit": lowerer.return_classical_bit,
            "hybrid_dynamic_measurement_count": lowerer.measurement_count,
            "hybrid_stochastic_gradient_policy": "unsupported_fail_closed",
        },
    )
    input_identity = hashlib.sha256(b"[]").hexdigest()
    return LoweredDynamicProgram(
        circuit=circuit,
        program_identity=program.semantic_identity,
        input_signature_identity=input_identity,
        measurement_count=lowerer.measurement_count,
        return_classical_bit=lowerer.return_classical_bit,
    )


__all__ = ("LoweredDynamicProgram", "lower_dynamic_program")
