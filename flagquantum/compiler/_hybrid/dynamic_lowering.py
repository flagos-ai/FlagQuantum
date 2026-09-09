"""Lower the bounded measurement-feedback profile to Core CircuitIR."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping, NoReturn, Sequence

from ...core.ir import CircuitIR, Instruction
from ...core.parameters import Parameter, bind_parameter_value
from .model import Block, HybridProgram, Operation, Region, Value, ValueId
from .specialize import SpecializationError, _input_signature, _runtime_shape
from .verifier import verify_program


@dataclass(frozen=True)
class LoweredDynamicProgram:
    """Core handoff for one private measurement-feedback program."""

    circuit_template: CircuitIR
    bindings: Mapping[str, Any] = field(compare=False, repr=False)
    program_identity: str
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


@dataclass(frozen=True)
class _MeasurementRef:
    classical_bit: int


@dataclass(frozen=True)
class _ClassicalPredicate:
    clauses: tuple[tuple[tuple[int, int], ...], ...]


_EFFECT = object()


_TRUE_CLAUSES: tuple[tuple[tuple[int, int], ...], ...] = ((),)
_FALSE_CLAUSES: tuple[tuple[tuple[int, int], ...], ...] = ()


class _DynamicLowerer:
    def __init__(
        self, *, max_unrolled_iterations: int, max_condition_clauses: int
    ) -> None:
        if int(max_unrolled_iterations) <= 0:
            raise ValueError("max_unrolled_iterations must be positive")
        self.max_unrolled_iterations = int(max_unrolled_iterations)
        if int(max_condition_clauses) <= 0:
            raise ValueError("max_condition_clauses must be positive")
        self.max_condition_clauses = int(max_condition_clauses)
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
        conditions: tuple[tuple[tuple[int, int], ...], ...],
    ) -> tuple[Any, ...]:
        name = operation.name
        if name == "arith.constant":
            return (operation.attributes["value"],)
        if name == "arith.add":
            return (operands[0] + operands[1],)
        if name == "arith.rem":
            try:
                return (operands[0] % operands[1],)
            except (RuntimeError, TypeError, ValueError, ZeroDivisionError) as exc:
                self.fail(operation, "arith.remainder", f"remainder failed: {exc}")
        if name == "arith.not":
            return (self.negate_predicate(operation, operands[0]),)
        if name == "arith.and":
            return (self.conjoin_predicates(operation, operands),)
        if name == "arith.or":
            return (self.disjoin_predicates(operation, operands),)
        if name == "arith.cmp":
            predicate = operation.attributes["predicate"]
            if any(
                isinstance(value, (_MeasurementRef, _ClassicalPredicate))
                for value in operands
            ):
                return (self.compare_predicate(operation, predicate, operands),)
            comparisons = {
                "eq": lambda: operands[0] == operands[1],
                "ne": lambda: operands[0] != operands[1],
                "lt": lambda: operands[0] < operands[1],
                "le": lambda: operands[0] <= operands[1],
                "gt": lambda: operands[0] > operands[1],
                "ge": lambda: operands[0] >= operands[1],
            }
            return (comparisons[predicate](),)
        if name in {"quantum.h", "quantum.x"}:
            self.require_effect(operation, operands[1])
            wire = self.as_index(operation, operands[0])
            self.append_gate(name.removeprefix("quantum."), (wire,), conditions)
            return (_EFFECT,)
        if name in {"quantum.rx", "quantum.ry"}:
            self.require_effect(operation, operands[2])
            wire = self.as_index(operation, operands[1])
            self.append_gate(
                name.removeprefix("quantum."),
                (wire,),
                conditions,
                parameter=operands[0],
            )
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
            if conditions != _TRUE_CLAUSES:
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
            if not isinstance(predicate, (_MeasurementRef, _ClassicalPredicate)):
                selected = self.as_predicate(operation, predicate)
                return self.execute_region(
                    operation.regions[0 if selected else 1],
                    environment,
                    operands[1:],
                    conditions=conditions,
                )
            predicate_clauses = self.predicate_clauses(predicate)
            if len(operands) != 2:
                self.fail(
                    operation,
                    "dynamic.measurement_branch_carry",
                    "measurement-dependent branches cannot carry classical values",
                )
            then_conditions = self.and_clauses(operation, conditions, predicate_clauses)
            else_conditions = self.and_clauses(
                operation, conditions, self.not_clauses(operation, predicate_clauses)
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
            if then_result != else_result or then_result != (_EFFECT,):
                self.fail(
                    operation,
                    "dynamic.branch_effect",
                    "dynamic branches must yield only the quantum effect",
                )
            return (_EFFECT,)
        if name == "scf.for":
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
            measurements_before = self.measurement_count
            for iteration in iterations:
                carried = self.execute_region(
                    operation.regions[0],
                    environment,
                    (iteration, *carried),
                    conditions=conditions,
                )
            if self.measurement_count != measurements_before:
                self.fail(
                    operation,
                    "dynamic.loop_measurement",
                    "measurement inside a loop is outside the bounded profile",
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

    def predicate_clauses(
        self, value: _MeasurementRef | _ClassicalPredicate
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        if isinstance(value, _MeasurementRef):
            return (((value.classical_bit, 1),),)
        return value.clauses

    def canonicalize_clauses(
        self,
        operation: Operation,
        clauses: Sequence[Sequence[tuple[int, int]]],
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        canonical: set[tuple[tuple[int, int], ...]] = set()
        for clause in clauses:
            terms: dict[int, int] = {}
            contradictory = False
            for bit, expected in clause:
                previous = terms.get(bit)
                if previous is not None and previous != expected:
                    contradictory = True
                    break
                terms[bit] = expected
            if not contradictory:
                normalized = tuple(sorted(terms.items()))
                if not normalized:
                    return _TRUE_CLAUSES
                canonical.add(normalized)
        reduced = tuple(
            clause
            for clause in sorted(canonical, key=lambda item: (len(item), item))
            if not any(
                set(other).issubset(clause) for other in canonical if other != clause
            )
        )
        if len(reduced) > self.max_condition_clauses:
            self.fail(
                operation,
                "dynamic.condition_clause_limit",
                f"predicate exceeds {self.max_condition_clauses} canonical clauses",
            )
        return reduced

    def and_clauses(self, operation: Operation, left: Any, right: Any) -> Any:
        if not left or not right:
            return _FALSE_CLAUSES
        return self.canonicalize_clauses(
            operation, tuple((*a, *b) for a in left for b in right)
        )

    def or_clauses(self, operation: Operation, left: Any, right: Any) -> Any:
        return self.canonicalize_clauses(operation, (*left, *right))

    def not_clauses(self, operation: Operation, clauses: Any) -> Any:
        if clauses == _FALSE_CLAUSES:
            return _TRUE_CLAUSES
        if clauses == _TRUE_CLAUSES:
            return _FALSE_CLAUSES
        result = _TRUE_CLAUSES
        for clause in clauses:
            complement = tuple(((bit, 1 - expected),) for bit, expected in clause)
            result = self.and_clauses(operation, result, complement)
        return result

    def negate_predicate(self, operation: Operation, value: Any) -> Any:
        if isinstance(value, _MeasurementRef):
            return _ClassicalPredicate((((value.classical_bit, 0),),))
        if isinstance(value, _ClassicalPredicate):
            return _ClassicalPredicate(self.not_clauses(operation, value.clauses))
        return not self.as_predicate(operation, value)

    def conjoin_predicates(self, operation: Operation, values: tuple[Any, ...]) -> Any:
        result = _TRUE_CLAUSES
        for value in values:
            if not isinstance(value, (_MeasurementRef, _ClassicalPredicate)):
                if not self.as_predicate(operation, value):
                    return False
                continue
            result = self.and_clauses(operation, result, self.predicate_clauses(value))
        if result == _FALSE_CLAUSES:
            return False
        if result == _TRUE_CLAUSES:
            return True
        return _ClassicalPredicate(result)

    def disjoin_predicates(self, operation: Operation, values: tuple[Any, ...]) -> Any:
        result = _FALSE_CLAUSES
        for value in values:
            if not isinstance(value, (_MeasurementRef, _ClassicalPredicate)):
                if self.as_predicate(operation, value):
                    return True
                continue
            result = self.or_clauses(operation, result, self.predicate_clauses(value))
        if result == _FALSE_CLAUSES:
            return False
        if result == _TRUE_CLAUSES:
            return True
        return _ClassicalPredicate(result)

    def compare_predicate(
        self,
        operation: Operation,
        predicate: str,
        values: tuple[Any, ...],
    ) -> Any:
        if predicate not in {"eq", "ne"}:
            self.fail(
                operation,
                "dynamic.classical_expression",
                "measurement predicates support only == or !=",
            )
        left, right = values
        left_symbolic = isinstance(left, (_MeasurementRef, _ClassicalPredicate))
        right_symbolic = isinstance(right, (_MeasurementRef, _ClassicalPredicate))
        if left_symbolic and right_symbolic:
            equal = self.disjoin_predicates(
                operation,
                (
                    self.conjoin_predicates(operation, (left, right)),
                    self.conjoin_predicates(
                        operation,
                        (
                            self.negate_predicate(operation, left),
                            self.negate_predicate(operation, right),
                        ),
                    ),
                ),
            )
            return (
                equal if predicate == "eq" else self.negate_predicate(operation, equal)
            )
        symbolic = left if left_symbolic else right
        constant = right if left_symbolic else left
        if not (left_symbolic or right_symbolic) or not isinstance(constant, bool):
            self.fail(
                operation,
                "dynamic.classical_expression",
                "measurement comparison requires a measurement predicate or bool constant",
            )
        should_negate = (predicate == "eq") != constant
        return self.negate_predicate(operation, symbolic) if should_negate else symbolic

    def as_predicate(self, operation: Operation, value: Any) -> bool:
        shape = _runtime_shape(value)
        if shape not in {None, ()}:
            self.fail(
                operation,
                "control.predicate_shape",
                f"runtime predicate must be scalar, got shape {shape}",
            )
        try:
            return bool(value)
        except (RuntimeError, TypeError, ValueError) as exc:
            self.fail(operation, "control.predicate", f"predicate is invalid: {exc}")

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
) -> LoweredDynamicProgram:
    """Lower a bounded measurement-feedback program without executing numerics."""

    verify_program(program)
    entry = program.body.blocks[0]
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
    )
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
        program_identity=program.semantic_identity,
        input_signature_identity=input_identity,
        measurement_count=lowerer.measurement_count,
        return_classical_bit=lowerer.return_classical_bit,
    )


__all__ = ("LoweredDynamicProgram", "lower_dynamic_program")
