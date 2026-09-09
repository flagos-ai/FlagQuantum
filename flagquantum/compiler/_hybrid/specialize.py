"""Runtime path specialization for verified private hybrid programs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, NoReturn, Sequence

from .model import (
    Block,
    HybridProgram,
    IRType,
    Operation,
    Region,
    SourceLocation,
    Value,
    ValueId,
)
from .verifier import verify_program


class SpecializationError(ValueError):
    """Raised when runtime values cannot specialize the accepted IR profile."""

    def __init__(
        self, code: str, message: str, location: SourceLocation | None = None
    ) -> None:
        self.code = code
        self.location = location
        prefix = "" if location is None else f"{location.filename}:{location.line}: "
        super().__init__(f"{prefix}{code}: {message}")


@dataclass(frozen=True)
class TraceGate:
    """One selected gate; its parameter is deliberately non-structural."""

    name: str
    wires: tuple[int, ...]
    parameter: Any | None = field(default=None, compare=False, repr=False)

    def structure(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "wires": list(self.wires),
            "parameterized": self.parameter is not None,
        }


@dataclass(frozen=True)
class SpecializedTrace:
    """Ephemeral selected quantum trace, not a second persistent quantum IR."""

    program_identity: str
    input_signature_identity: str
    gates: tuple[TraceGate, ...]
    observables: tuple[tuple[str, int], ...]
    control_decisions: tuple[str, ...]

    @property
    def structure_identity(self) -> str:
        payload = {
            "gates": [gate.structure() for gate in self.gates],
            "observables": [list(term) for term in self.observables],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


_EFFECT = object()
_EXPECTATION = object()


def _normalized_dtype(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return str(dtype).removeprefix("torch.")


def _runtime_shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(size) for size in shape)
    except (TypeError, ValueError):
        return None


def _runtime_device_class(value: Any) -> str:
    device = getattr(value, "device", None)
    if device is None:
        return "python"
    return str(getattr(device, "type", device))


def _input_descriptor(value: Any, declared: IRType) -> dict[str, Any]:
    shape = _runtime_shape(value)
    dtype = _normalized_dtype(value)
    if declared.kind == "tensor":
        expected_dtype, expected_shape = declared.parameters
        if shape is None:
            raise SpecializationError(
                "input.tensor", "tensor input must expose a concrete shape"
            )
        if len(shape) != len(expected_shape) or any(
            expected is not None and actual != expected
            for actual, expected in zip(shape, expected_shape)
        ):
            raise SpecializationError(
                "input.shape",
                f"runtime shape {shape} does not match declared {expected_shape}",
            )
        if dtype is not None and dtype != expected_dtype:
            raise SpecializationError(
                "input.dtype",
                f"runtime dtype {dtype!r} does not match declared {expected_dtype!r}",
            )
    elif declared.kind == "scalar":
        if shape not in {None, ()}:
            raise SpecializationError(
                "input.scalar",
                f"scalar input must have shape (), got {shape}",
            )
        expected_dtype = declared.parameters[0]
        if dtype is not None and dtype != expected_dtype:
            raise SpecializationError(
                "input.dtype",
                f"runtime dtype {dtype!r} does not match declared {expected_dtype!r}",
            )
    elif declared.kind == "index":
        if shape not in {None, ()}:
            raise SpecializationError(
                "input.scalar", f"index input must have shape (), got {shape}"
            )
        if dtype is None:
            if not isinstance(value, int) or isinstance(value, bool):
                raise SpecializationError(
                    "input.index", "index input must be an integer"
                )
        elif not dtype.startswith(("int", "uint")):
            raise SpecializationError(
                "input.index", f"index tensor must have integer dtype, got {dtype!r}"
            )
    elif declared.kind == "bool":
        if shape not in {None, ()}:
            raise SpecializationError(
                "input.scalar", f"bool input must have shape (), got {shape}"
            )
        if dtype is None and not isinstance(value, bool):
            raise SpecializationError("input.bool", "bool input must be boolean")
        if dtype is not None and dtype != "bool":
            raise SpecializationError(
                "input.bool", f"bool tensor must have bool dtype, got {dtype!r}"
            )
    else:
        raise SpecializationError(
            "input.type", f"unsupported runtime input type {declared.kind!r}"
        )
    return {
        "declared": declared.to_data(),
        "runtime_shape": None if shape is None else list(shape),
        "runtime_dtype": dtype or type(value).__name__,
        "device_class": _runtime_device_class(value),
    }


def _input_signature(values: Sequence[Any], types: Sequence[IRType]) -> str:
    descriptors = [
        _input_descriptor(value, declared) for value, declared in zip(values, types)
    ]
    encoded = json.dumps(descriptors, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class _Specializer:
    def __init__(self, *, max_unrolled_iterations: int) -> None:
        if int(max_unrolled_iterations) <= 0:
            raise ValueError("max_unrolled_iterations must be positive")
        self.max_unrolled_iterations = int(max_unrolled_iterations)
        self.unrolled_iterations = 0
        self.gates: list[TraceGate] = []
        self.observables: tuple[tuple[str, int], ...] = ()
        self.control_decisions: list[str] = []

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
            results = self.execute_operation(operation, operands, environment)
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
    ) -> tuple[Any, ...]:
        name = operation.name
        if name == "arith.constant":
            return (operation.attributes["value"],)
        if name == "tensor.dim":
            shape = _runtime_shape(operands[0])
            if shape is None:
                self.fail(operation, "tensor.shape", "tensor shape is unavailable")
            dimension = self.as_index(operation, operands[1])
            if dimension >= len(shape):
                self.fail(
                    operation, "tensor.dimension", "tensor dimension is out of range"
                )
            return (shape[dimension],)
        if name == "tensor.extract":
            indices = tuple(self.as_index(operation, value) for value in operands[1:])
            try:
                return (operands[0][indices],)
            except (IndexError, KeyError, TypeError) as exc:
                self.fail(
                    operation, "tensor.extract", f"tensor extraction failed: {exc}"
                )
        if name == "arith.add":
            return (operands[0] + operands[1],)
        if name == "arith.rem":
            try:
                return (operands[0] % operands[1],)
            except (RuntimeError, TypeError, ValueError, ZeroDivisionError) as exc:
                self.fail(operation, "arith.remainder", f"remainder failed: {exc}")
        if name == "arith.cmp":
            comparisons = {
                "eq": lambda: operands[0] == operands[1],
                "ne": lambda: operands[0] != operands[1],
                "lt": lambda: operands[0] < operands[1],
                "le": lambda: operands[0] <= operands[1],
                "gt": lambda: operands[0] > operands[1],
                "ge": lambda: operands[0] >= operands[1],
            }
            return (comparisons[operation.attributes["predicate"]](),)
        if name == "scf.if":
            selected = self.as_predicate(operation, operands[0])
            self.control_decisions.append(f"if:{str(selected).lower()}")
            region = operation.regions[0 if selected else 1]
            return self.execute_region(region, environment, operands[1:])
        if name == "scf.for":
            lower, upper, step = (
                self.as_index(operation, value) for value in operands[:3]
            )
            if step == 0:
                self.fail(operation, "control.range_step", "loop step cannot be zero")
            iterations = range(lower, upper, step)
            count = len(iterations)
            self.unrolled_iterations += count
            if self.unrolled_iterations > self.max_unrolled_iterations:
                self.fail(
                    operation,
                    "control.unroll_limit",
                    f"path exceeds {self.max_unrolled_iterations} loop iterations",
                )
            self.control_decisions.append(f"for:{count}")
            carried = operands[3:]
            for iteration in iterations:
                carried = self.execute_region(
                    operation.regions[0], environment, (iteration, *carried)
                )
            return carried
        if name == "quantum.angle_embedding":
            self.require_effect(operation, operands[1])
            wires = tuple(int(wire) for wire in operation.attributes["wires"])
            shape = _runtime_shape(operands[0])
            if shape is None or len(shape) != 1 or shape[0] != len(wires):
                self.fail(
                    operation,
                    "quantum.embedding_shape",
                    f"angle embedding requires shape ({len(wires)},), got {shape}",
                )
            for index, wire in enumerate(wires):
                self.gates.append(TraceGate("rx", (wire,), operands[0][index]))
            return (_EFFECT,)
        if name in {"quantum.rx", "quantum.ry"}:
            self.require_effect(operation, operands[2])
            wire = self.as_index(operation, operands[1])
            self.gates.append(
                TraceGate(name.removeprefix("quantum."), (wire,), operands[0])
            )
            return (_EFFECT,)
        if name == "quantum.cx":
            self.require_effect(operation, operands[2])
            wires = tuple(self.as_index(operation, value) for value in operands[:2])
            if wires[0] == wires[1]:
                self.fail(operation, "quantum.wires", "CX wires must be distinct")
            self.gates.append(TraceGate("cx", wires))
            return (_EFFECT,)
        if name == "quantum.expectation":
            self.require_effect(operation, operands[0])
            self.observables = tuple(
                (str(pauli), int(wire)) for pauli, wire in operation.attributes["terms"]
            )
            return (_EXPECTATION,)
        if name == "scf.yield":
            return operands
        if name == "program.return":
            if operands != (_EXPECTATION,):
                self.fail(
                    operation,
                    "program.return",
                    "program must return the selected expectation",
                )
            return operands
        self.fail(operation, "operation.unsupported", f"cannot specialize {name!r}")

    def execute_region(
        self,
        region: Region,
        outer: Mapping[ValueId, Any],
        arguments: Sequence[Any],
    ) -> tuple[Any, ...]:
        if len(region.blocks) != 1:
            raise SpecializationError(
                "region.block_count", "first profile requires one region block"
            )
        return self.execute_block(region.blocks[0], outer, arguments)

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
        shape = _runtime_shape(value)
        if shape not in {None, ()}:
            self.fail(
                operation,
                "index.shape",
                f"runtime index must be scalar, got shape {shape}",
            )
        try:
            index = int(value)
        except (RuntimeError, TypeError, ValueError) as exc:
            self.fail(operation, "index.value", f"index conversion failed: {exc}")
        if index < 0:
            self.fail(
                operation, "index.value", "wire and tensor indices must be non-negative"
            )
        return index

    def require_effect(self, operation: Operation, value: Any) -> None:
        if value is not _EFFECT:
            self.fail(operation, "quantum.effect", "runtime quantum effect is invalid")


def specialize_program(
    program: HybridProgram,
    inputs: Sequence[Any],
    *,
    max_unrolled_iterations: int = 10_000,
) -> SpecializedTrace:
    """Select one bounded runtime path without executing numerical kernels."""

    verify_program(program)
    entry = program.body.blocks[0]
    declared_inputs = entry.arguments[:-1]
    if len(inputs) != len(declared_inputs):
        raise SpecializationError(
            "input.arity",
            f"expected {len(declared_inputs)} runtime input(s), got {len(inputs)}",
        )
    declared_types = tuple(value.type for value in declared_inputs)
    input_identity = _input_signature(tuple(inputs), declared_types)
    specializer = _Specializer(max_unrolled_iterations=max_unrolled_iterations)
    returned = specializer.execute_block(entry, {}, (*inputs, _EFFECT))
    if returned != (_EXPECTATION,):
        raise SpecializationError(
            "program.return", "specialized program did not return one expectation"
        )
    if not specializer.observables:
        raise SpecializationError(
            "quantum.observable", "specialized program produced no observable"
        )
    return SpecializedTrace(
        program_identity=program.semantic_identity,
        input_signature_identity=input_identity,
        gates=tuple(specializer.gates),
        observables=specializer.observables,
        control_decisions=tuple(specializer.control_decisions),
    )
