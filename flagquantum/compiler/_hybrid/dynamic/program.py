"""Lower the bounded measurement-feedback profile to Core CircuitIR."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, NoReturn

from ....core.ir import CircuitIR, Instruction
from ....core.parameters import Parameter, bind_parameter_value
from ..diagnostics import SpecializationError
from ..model import Block, HybridProgram, Operation, Region, Value, ValueId
from ..passes import PassRecord, run_pass_pipeline
from ..runtime_inputs import _input_signature
from . import expressions
from .expressions import (
    _FALSE_CLAUSES,
    _TRUE_CLAUSES,
    _ClassicalPredicate,
    _ConditionalValue,
    _MeasurementRef,
)


@dataclass(frozen=True)
class LoweredDynamicProgram:
    """Core handoff for one private measurement-feedback program."""

    circuit_template: CircuitIR
    bindings: Mapping[str, Any] = field(compare=False, repr=False)
    program_identity: str
    optimized_program_identity: str
    optimization_records: tuple[PassRecord, ...]
    input_signature_identity: str
    measurement_count: int
    return_classical_bit: int
    stochastic_gradient_policy: str = "unsupported_fail_closed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", MappingProxyType(dict(self.bindings)))

    @property
    def ordered_bindings(self) -> tuple[tuple[str, Any], ...]:
        names = tuple(self.circuit_template.metadata["hybrid_parameter_order"])
        return tuple((name, self.bindings[name]) for name in names)

    def bind(self) -> CircuitIR:
        instructions = tuple(
            replace(
                instruction,
                params=bind_parameter_value(instruction.params, self.bindings),
            )
            for instruction in self.circuit_template.instructions
        )
        return replace(self.circuit_template, instructions=instructions)

    @property
    def circuit(self) -> CircuitIR:
        """Return a bound Core artifact accepted by the private Runtime handoff."""

        return self.bind()


_EFFECT = object()


class _DynamicLowerer:
    def __init__(
        self,
        *,
        max_unrolled_iterations: int,
        max_condition_clauses: int,
        max_dynamic_measurements: int,
    ) -> None:
        if int(max_unrolled_iterations) <= 0:
            raise ValueError("max_unrolled_iterations must be positive")
        self.max_unrolled_iterations = int(max_unrolled_iterations)
        if int(max_condition_clauses) <= 0:
            raise ValueError("max_condition_clauses must be positive")
        self.max_condition_clauses = int(max_condition_clauses)
        if int(max_dynamic_measurements) <= 0:
            raise ValueError("max_dynamic_measurements must be positive")
        self.max_dynamic_measurements = int(max_dynamic_measurements)
        self.unrolled_iterations = 0
        self.instructions: list[Instruction] = []
        self.bindings: dict[str, Any] = {}
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
        conditions: tuple[tuple[tuple[int, int], ...], ...] = _TRUE_CLAUSES,
    ) -> tuple[Any, ...]:
        if len(block.arguments) != len(arguments):
            raise SpecializationError(
                "region.arguments", "runtime region argument arity mismatch"
            )
        environment = dict(outer)
        environment.update(
            (reference.id, runtime)
            for reference, runtime in zip(block.arguments, arguments, strict=True)
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
                for reference, runtime in zip(operation.results, results, strict=True)
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
        conditions: tuple[tuple[tuple[int, int], ...], ...],
    ) -> tuple[Any, ...]:
        name = operation.name
        if name == "arith.constant":
            return (operation.attributes["value"],)
        if name == "arith.add":
            return (
                expressions.map_conditional_values(
                    self, operation, operands, lambda left, right: left + right
                ),
            )
        if name == "arith.rem":
            try:
                return (
                    expressions.map_conditional_values(
                        self, operation, operands, lambda left, right: left % right
                    ),
                )
            except (RuntimeError, TypeError, ValueError, ZeroDivisionError) as exc:
                self.fail(operation, "arith.remainder", f"remainder failed: {exc}")
        if name == "arith.not":
            return (expressions.negate_predicate(self, operation, operands[0]),)
        if name == "arith.and":
            return (expressions.conjoin_predicates(self, operation, operands),)
        if name == "arith.or":
            return (expressions.disjoin_predicates(self, operation, operands),)
        if name == "arith.cmp":
            predicate = operation.attributes["predicate"]
            if any(
                isinstance(
                    value,
                    (_MeasurementRef, _ClassicalPredicate, _ConditionalValue),
                )
                for value in operands
            ):
                return (
                    expressions.compare_predicate(self, operation, predicate, operands),
                )
            return (expressions.compare_values(predicate, *operands),)
        if name in {"quantum.h", "quantum.x"}:
            self.require_effect(operation, operands[1])
            for case_conditions, values in expressions.conditional_product(
                self, operation, (operands[0],)
            ):
                wire = self.as_index(operation, values[0])
                self.append_gate(
                    name.removeprefix("quantum."),
                    (wire,),
                    expressions.and_clauses(
                        self, operation, conditions, case_conditions
                    ),
                )
            return (_EFFECT,)
        if name in {"quantum.rx", "quantum.ry"}:
            self.require_effect(operation, operands[2])
            for case_conditions, values in expressions.conditional_product(
                self, operation, operands[:2]
            ):
                parameter, raw_wire = values
                wire = self.as_index(operation, raw_wire)
                self.append_gate(
                    name.removeprefix("quantum."),
                    (wire,),
                    expressions.and_clauses(
                        self, operation, conditions, case_conditions
                    ),
                    parameter=parameter,
                )
            return (_EFFECT,)
        if name == "quantum.cx":
            self.require_effect(operation, operands[2])
            for case_conditions, values in expressions.conditional_product(
                self, operation, operands[:2]
            ):
                wires = tuple(self.as_index(operation, value) for value in values)
                if wires[0] == wires[1]:
                    self.fail(operation, "quantum.wires", "CX wires must be distinct")
                self.append_gate(
                    "cx",
                    wires,
                    expressions.and_clauses(
                        self, operation, conditions, case_conditions
                    ),
                )
            return (_EFFECT,)
        if name == "quantum.measure":
            self.require_effect(operation, operands[1])
            if conditions != _TRUE_CLAUSES:
                self.fail(
                    operation,
                    "dynamic.conditional_measurement",
                    "the first dynamic profile forbids measurement inside a branch",
                )
            if isinstance(operands[0], _ConditionalValue):
                self.fail(
                    operation,
                    "dynamic.conditional_measurement_target",
                    "measurement-dependent wires are unsupported",
                )
            wire = self.as_index(operation, operands[0])
            classical_bit = self.measurement_count
            if classical_bit >= self.max_dynamic_measurements:
                self.fail(
                    operation,
                    "dynamic.measurement_limit",
                    f"program exceeds {self.max_dynamic_measurements} measurements",
                )
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
        if name == "quantum.reset":
            self.require_effect(operation, operands[1])
            if conditions != _TRUE_CLAUSES:
                self.fail(
                    operation,
                    "dynamic.conditional_reset",
                    "conditional reset is outside the fixed-round profile",
                )
            if isinstance(operands[0], _ConditionalValue):
                self.fail(
                    operation,
                    "dynamic.conditional_reset_target",
                    "measurement-dependent reset wires are unsupported",
                )
            self.instructions.append(
                Instruction(
                    "reset",
                    (self.as_index(operation, operands[0]),),
                    metadata={"is_dynamic": True},
                )
            )
            return (_EFFECT,)
        if name == "scf.if":
            predicate = operands[0]
            if not isinstance(
                predicate,
                (_MeasurementRef, _ClassicalPredicate, _ConditionalValue),
            ):
                selected = expressions.as_predicate(self, operation, predicate)
                return self.execute_region(
                    operation.regions[0 if selected else 1],
                    environment,
                    operands[1:],
                    conditions=conditions,
                )
            predicate_clauses = expressions.predicate_clauses(
                self, operation, predicate
            )
            then_conditions = expressions.and_clauses(
                self, operation, conditions, predicate_clauses
            )
            else_conditions = expressions.and_clauses(
                self,
                operation,
                conditions,
                expressions.not_clauses(self, operation, predicate_clauses),
            )
            then_result = self.execute_region(
                operation.regions[0],
                environment,
                operands[1:],
                conditions=then_conditions,
            )
            else_result = self.execute_region(
                operation.regions[1],
                environment,
                operands[1:],
                conditions=else_conditions,
            )
            if (
                not then_result
                or not else_result
                or (then_result[-1] is not _EFFECT or else_result[-1] is not _EFFECT)
            ):
                self.fail(
                    operation,
                    "dynamic.branch_effect",
                    "dynamic branches must yield the quantum effect",
                )
            if len(then_result) != len(else_result):
                self.fail(
                    operation,
                    "dynamic.branch_results",
                    "dynamic branches returned different result counts",
                )
            merged = tuple(
                expressions.merge_conditional_values(
                    self,
                    operation,
                    ((then_conditions, then_value), (else_conditions, else_value)),
                )
                for then_value, else_value in zip(
                    then_result[:-1], else_result[:-1], strict=False
                )
            )
            return (*merged, _EFFECT)
        if name == "scf.for":
            if any(isinstance(value, _ConditionalValue) for value in operands[:3]):
                self.fail(
                    operation,
                    "dynamic.loop_bounds",
                    "measurement-dependent loop bounds are unsupported",
                )
            lower, upper, step = (
                self.as_index(operation, value) for value in operands[:3]
            )
            if step == 0:
                self.fail(operation, "control.range_step", "loop step cannot be zero")
            iterations = range(lower, upper, step)
            self.unrolled_iterations += len(iterations)
            if self.unrolled_iterations > self.max_unrolled_iterations:
                self.fail(
                    operation,
                    "control.unroll_limit",
                    f"path exceeds {self.max_unrolled_iterations} loop iterations",
                )
            carried = operands[3:]
            for iteration in iterations:
                carried = self.execute_region(
                    operation.regions[0],
                    environment,
                    (iteration, *carried),
                    conditions=conditions,
                )
            return carried
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
        conditions: tuple[tuple[tuple[int, int], ...], ...],
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
        conditions: tuple[tuple[tuple[int, int], ...], ...],
        *,
        parameter: Any | None = None,
    ) -> None:
        if conditions == _FALSE_CLAUSES:
            return
        metadata: dict[str, object]
        if conditions == _TRUE_CLAUSES:
            metadata = {}
        elif len(conditions) == 1:
            metadata = {"conditions": conditions[0]}
        else:
            metadata = {"condition_clauses": conditions}
        params: dict[str, Any] = {}
        if parameter is not None:
            slot = f"hybrid_dynamic_p{len(self.bindings)}"
            params["theta"] = Parameter(slot)
            self.bindings[slot] = parameter
        self.instructions.append(
            Instruction(name, wires, params=params, metadata=metadata)
        )

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
    max_unrolled_iterations: int = 10_000,
    max_condition_clauses: int = 64,
    max_dynamic_measurements: int = 4_096,
    optimize: bool = True,
) -> LoweredDynamicProgram:
    """Lower a bounded measurement-feedback program without executing numerics."""

    if type(optimize) is not bool:
        raise TypeError("optimize must be a bool")
    optimization = (
        run_pass_pipeline(program) if optimize else run_pass_pipeline(program, ())
    )
    optimized_program = optimization.program
    entry = optimized_program.body.blocks[0]
    declared_inputs = entry.arguments[:-1]
    if len(inputs) != len(declared_inputs):
        raise SpecializationError(
            "input.arity",
            f"expected {len(declared_inputs)} runtime input(s), got {len(inputs)}",
        )
    if any(
        value.type.kind not in {"scalar", "index", "bool"} for value in declared_inputs
    ):
        raise SpecializationError(
            "dynamic.input_profile",
            "bounded dynamic inputs must be scalar, index, or bool values",
        )
    if any(bool(getattr(value, "requires_grad", False)) for value in inputs):
        raise SpecializationError(
            "dynamic.stochastic_gradient",
            "trainable inputs are unsupported for finite-shot dynamic sessions",
        )
    declared_types = tuple(value.type for value in declared_inputs)
    input_identity = _input_signature(tuple(inputs), declared_types)
    if circuit_dtype not in {"complex64", "complex128"}:
        raise ValueError("dynamic circuit_dtype must be complex64 or complex128")
    real_dtype = "float64" if circuit_dtype == "complex128" else "float32"
    input_dtypes = {
        value.type.parameters[0]
        for value in declared_inputs
        if value.type.kind == "scalar"
    }
    if input_dtypes - {real_dtype}:
        raise SpecializationError(
            "dynamic.dtype",
            f"{circuit_dtype} requires scalar inputs declared as {real_dtype}",
        )
    lowerer = _DynamicLowerer(
        max_unrolled_iterations=max_unrolled_iterations,
        max_condition_clauses=max_condition_clauses,
        max_dynamic_measurements=max_dynamic_measurements,
    )
    if optimization.preexpanded_iterations > lowerer.max_unrolled_iterations:
        raise SpecializationError(
            "control.unroll_limit",
            "compile-time-expanded loops exceed "
            f"{lowerer.max_unrolled_iterations} loop iterations",
            program.location,
        )
    lowerer.unrolled_iterations = optimization.preexpanded_iterations
    lowerer.execute_block(entry, {}, (*inputs, _EFFECT))
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
    circuit_template = CircuitIR(
        n_wires=max(referenced_wires) + 1,
        instructions=tuple(lowerer.instructions),
        dtype=circuit_dtype,
        metadata={
            "hybrid_dynamic_session": True,
            "hybrid_dynamic_return_bit": lowerer.return_classical_bit,
            "hybrid_dynamic_measurement_count": lowerer.measurement_count,
            "hybrid_stochastic_gradient_policy": "unsupported_fail_closed",
            "hybrid_parameter_order": tuple(lowerer.bindings),
            "hybrid_dynamic_unrolled_iterations": lowerer.unrolled_iterations,
        },
    )
    return LoweredDynamicProgram(
        circuit_template=circuit_template,
        bindings=lowerer.bindings,
        program_identity=optimization.source_identity,
        optimized_program_identity=optimization.optimized_identity,
        optimization_records=optimization.records,
        input_signature_identity=input_identity,
        measurement_count=lowerer.measurement_count,
        return_classical_bit=lowerer.return_classical_bit,
    )


__all__ = ("LoweredDynamicProgram", "lower_dynamic_program")
