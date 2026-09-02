"""Phase 2 Batch A static canonicalization on private QuantumIR."""

from __future__ import annotations

from dataclasses import replace
from math import isclose
from typing import Any

import torch

from ..bindings import RuntimeBindingRef, SymbolicExpression, SymbolicParameter
from ..diagnostics import Diagnostic, DiagnosticCode
from ..ir.modules import Block, QuantumModule, Region
from ..ir.operations import FrozenAttributes, Operation
from ..ir.schemas import circuit_ir_v1_schema_registry
from ..ir.values import ValueId, ValueRef
from ..ir.verifier import verify_module
from .base import CompilerPass, PassDescriptor, PassResult

_SELF_INVERSE = {
    "quantum.x",
    "quantum.y",
    "quantum.z",
    "quantum.h",
    "quantum.cx",
    "quantum.cy",
    "quantum.cz",
    "quantum.swap",
    "quantum.ccx",
    "quantum.cswap",
}
_ROTATION_PARAM = {
    "quantum.rx": "theta",
    "quantum.ry": "theta",
    "quantum.rz": "theta",
    "quantum.phase": "theta",
    "quantum.u1": "theta",
    "quantum.rxx": "theta",
    "quantum.ryy": "theta",
    "quantum.rzz": "theta",
    "quantum.crx": "theta",
    "quantum.cry": "theta",
    "quantum.crz": "theta",
    "quantum.cphase": "theta",
}


def _failure(message: str) -> Diagnostic:
    return Diagnostic(
        DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
        message,
        notes=("Phase 2 canonicalization never repairs unsupported IR",),
    )


def _single_block(module: QuantumModule) -> Block | None:
    if len(module.body.blocks) != 1:
        return None
    return module.body.blocks[0]


def _resolve(value: ValueRef, aliases: dict[ValueId, ValueRef]) -> ValueRef:
    seen: set[ValueId] = set()
    while value.id in aliases:
        if value.id in seen:
            raise ValueError("canonicalization value alias cycle")
        seen.add(value.id)
        value = aliases[value.id]
    return value


def _rewire(operation: Operation, aliases: dict[ValueId, ValueRef]) -> Operation:
    operands = tuple(_resolve(value, aliases) for value in operation.operands)
    return (
        operation
        if operands == operation.operands
        else replace(operation, operands=operands)
    )


def _wire_state(block: Block) -> dict[ValueId, int]:
    return {value.id: wire for wire, value in enumerate(block.arguments)}


def _operation_wires(
    operation: Operation, wire_by_value: dict[ValueId, int]
) -> tuple[int, ...]:
    return tuple(wire_by_value[value.id] for value in operation.operands)


def _record_results(
    operation: Operation,
    wires: tuple[int, ...],
    wire_by_value: dict[ValueId, int],
) -> None:
    for result, wire in zip(operation.results, wires, strict=True):
        wire_by_value[result.id] = wire


def _last_touching(
    history_by_wire: dict[int, list[int]],
    wires: tuple[int, ...],
) -> int | None:
    candidates = (
        history_by_wire[wire][-1] for wire in wires if history_by_wire.get(wire)
    )
    return max(candidates, default=None)


def _record_touch(
    index: int,
    wires: tuple[int, ...],
    history_by_wire: dict[int, list[int]],
) -> None:
    for wire in wires:
        history_by_wire.setdefault(wire, []).append(index)


def _remove_touch(
    index: int,
    wires: tuple[int, ...],
    history_by_wire: dict[int, list[int]],
) -> None:
    for wire in wires:
        history = history_by_wire[wire]
        if not history or history[-1] != index:
            raise ValueError("canonicalization wire history is inconsistent")
        history.pop()


def _static_tensor(value: object) -> torch.Tensor | None:
    if not isinstance(value, FrozenAttributes) or value.get("kind") != "tensor":
        return None
    dtype = getattr(torch, str(value.get("dtype")), None)
    shape = tuple(value.get("shape", ()))
    if dtype is None:
        return None
    return torch.tensor(value.get("data"), dtype=dtype).reshape(shape)


def _freeze_tensor(value: torch.Tensor) -> FrozenAttributes:
    tensor = value.detach().cpu()
    data = tensor.item() if tensor.ndim == 0 else tensor.tolist()
    return FrozenAttributes(
        {
            "kind": "tensor",
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": tuple(tensor.shape),
            "data": data,
        }
    )


def _is_zero(value: object, atol: float = 1e-12) -> bool:
    if isinstance(
        value,
        (RuntimeBindingRef, SymbolicParameter, SymbolicExpression),
    ):
        return False
    tensor = _static_tensor(value)
    if tensor is not None:
        if tensor.numel() != 1:
            return False
        value = tensor.item()
    try:
        return isclose(float(value), 0.0, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def _add_values(left: object, right: object) -> object:
    left_tensor = _static_tensor(left)
    right_tensor = _static_tensor(right)
    symbolic = (
        RuntimeBindingRef,
        SymbolicParameter,
        SymbolicExpression,
    )
    if isinstance(left, symbolic) or isinstance(right, symbolic):
        return SymbolicExpression("add", (left, right))
    if left_tensor is not None or right_tensor is not None:
        left_value: Any = left_tensor if left_tensor is not None else left
        right_value: Any = right_tensor if right_tensor is not None else right
        try:
            result = left_value + right_value
        except (TypeError, RuntimeError) as exc:
            raise ValueError("static rotation parameters cannot be added") from exc
        if not isinstance(result, torch.Tensor):
            result = torch.as_tensor(result)
        return _freeze_tensor(result)
    try:
        return left + right  # type: ignore[operator]
    except TypeError as exc:
        raise ValueError("rotation parameters cannot be added") from exc


def _result(
    before: QuantumModule,
    operations: list[Operation],
    *,
    changed: bool,
    statistics: dict[str, int],
    verify_output: bool = True,
) -> PassResult:
    if not changed:
        return PassResult(
            before,
            changed=False,
            preserved_analyses=frozenset({"def_use", "qubit_lifetime"}),
            statistics=statistics,
        )
    block = before.body.blocks[0]
    candidate = QuantumModule(
        Region((Block(block.arguments, tuple(operations)),)),
        revision=before.revision + 1,
    )
    if verify_output:
        verification = verify_module(candidate, circuit_ir_v1_schema_registry())
        if not verification.ok:
            return PassResult(
                candidate,
                changed=True,
                diagnostics=verification.diagnostics,
                statistics=statistics,
            )
    return PassResult(candidate, changed=True, statistics=statistics)


class RemoveIdentityOperationsPass:
    descriptor = PassDescriptor(
        "remove_identity_operations",
        "2.0",
        program_identity_policy="transform",
    )

    def __init__(self, *, verify_output: bool = True) -> None:
        self._verify_output = bool(verify_output)

    def run(self, module: QuantumModule) -> PassResult:
        block = _single_block(module)
        if block is None:
            return PassResult(module, diagnostics=(_failure("one block is required"),))
        aliases: dict[ValueId, ValueRef] = {}
        operations: list[Operation] = []
        removed = 0
        for original in block.operations:
            operation = _rewire(original, aliases)
            parameter = _ROTATION_PARAM.get(operation.name)
            remove = operation.name in {"quantum.i", "quantum.id"} or (
                parameter is not None and _is_zero(operation.attributes.get(parameter))
            )
            if remove:
                for result, operand in zip(
                    operation.results, operation.operands, strict=True
                ):
                    aliases[result.id] = operand
                removed += 1
            else:
                operations.append(operation)
        return _result(
            module,
            operations,
            changed=bool(removed),
            statistics={"operations_removed": removed},
            verify_output=self._verify_output,
        )


class CancelSelfInverseOperationsPass:
    descriptor = PassDescriptor(
        "cancel_self_inverse_operations",
        "2.0",
        program_identity_policy="transform",
    )

    def __init__(self, *, verify_output: bool = True) -> None:
        self._verify_output = bool(verify_output)

    def run(self, module: QuantumModule) -> PassResult:
        block = _single_block(module)
        if block is None:
            return PassResult(module, diagnostics=(_failure("one block is required"),))
        aliases: dict[ValueId, ValueRef] = {}
        wire_by_value = _wire_state(block)
        operations: list[Operation | None] = []
        history_by_wire: dict[int, list[int]] = {}
        cancelled_pairs = 0
        for original in block.operations:
            operation = _rewire(original, aliases)
            wires = _operation_wires(operation, wire_by_value)
            previous_index = _last_touching(history_by_wire, wires)
            previous = (
                operations[previous_index] if previous_index is not None else None
            )
            if (
                operation.name in _SELF_INVERSE
                and previous is not None
                and previous.name == operation.name
                and _operation_wires(previous, wire_by_value) == wires
                and not operation.attributes
                and not previous.attributes
            ):
                previous_wires = _operation_wires(previous, wire_by_value)
                operations[previous_index] = None
                _remove_touch(previous_index, previous_wires, history_by_wire)
                for result, operand in zip(
                    operation.results, previous.operands, strict=True
                ):
                    aliases[result.id] = operand
                cancelled_pairs += 1
                continue
            operations.append(operation)
            _record_touch(len(operations) - 1, wires, history_by_wire)
            _record_results(operation, wires, wire_by_value)
        return _result(
            module,
            [operation for operation in operations if operation is not None],
            changed=bool(cancelled_pairs),
            statistics={"pairs_cancelled": cancelled_pairs},
            verify_output=self._verify_output,
        )


class MergeAdjacentRotationsPass:
    descriptor = PassDescriptor(
        "merge_adjacent_rotations",
        "2.0",
        program_identity_policy="transform",
    )

    def __init__(self, *, verify_output: bool = True) -> None:
        self._verify_output = bool(verify_output)

    def run(self, module: QuantumModule) -> PassResult:
        block = _single_block(module)
        if block is None:
            return PassResult(module, diagnostics=(_failure("one block is required"),))
        aliases: dict[ValueId, ValueRef] = {}
        wire_by_value = _wire_state(block)
        operations: list[Operation | None] = []
        history_by_wire: dict[int, list[int]] = {}
        merged_count = 0
        removed_zero_count = 0
        for original in block.operations:
            operation = _rewire(original, aliases)
            wires = _operation_wires(operation, wire_by_value)
            parameter = _ROTATION_PARAM.get(operation.name)
            previous_index = _last_touching(history_by_wire, wires)
            previous = (
                operations[previous_index] if previous_index is not None else None
            )
            if (
                parameter is not None
                and previous is not None
                and previous.name == operation.name
                and _operation_wires(previous, wire_by_value) == wires
                and parameter in previous.attributes
                and parameter in operation.attributes
            ):
                try:
                    merged_value = _add_values(
                        previous.attributes[parameter],
                        operation.attributes[parameter],
                    )
                except ValueError as exc:
                    return PassResult(
                        module,
                        diagnostics=(_failure(str(exc)),),
                        statistics={"rotations_merged": merged_count},
                    )
                merged_count += 1
                if _is_zero(merged_value):
                    previous_wires = _operation_wires(previous, wire_by_value)
                    operations[previous_index] = None
                    _remove_touch(previous_index, previous_wires, history_by_wire)
                    for result, operand in zip(
                        operation.results, previous.operands, strict=True
                    ):
                        aliases[result.id] = operand
                    removed_zero_count += 1
                    continue
                attributes = dict(previous.attributes)
                attributes[parameter] = merged_value
                merged = replace(
                    previous,
                    results=operation.results,
                    attributes=FrozenAttributes(attributes),
                )
                operations[previous_index] = merged
                _record_results(merged, wires, wire_by_value)
                continue
            operations.append(operation)
            _record_touch(len(operations) - 1, wires, history_by_wire)
            _record_results(operation, wires, wire_by_value)
        return _result(
            module,
            [operation for operation in operations if operation is not None],
            changed=bool(merged_count),
            statistics={
                "rotations_merged": merged_count,
                "zero_sums_removed": removed_zero_count,
            },
            verify_output=self._verify_output,
        )


class StaticCanonicalizationPass:
    """Fused Batch A path with one entry and one exit verification."""

    descriptor = PassDescriptor(
        "static_canonicalization",
        "2.0",
        options={
            "components": (
                "remove_identity_operations",
                "cancel_self_inverse_operations",
                "merge_adjacent_rotations",
                "remove_identity_operations",
            ),
            "verification": "entry_exit",
        },
        program_identity_policy="transform",
    )

    def run(self, module: QuantumModule) -> PassResult:
        entry_verification = verify_module(module, circuit_ir_v1_schema_registry())
        if not entry_verification.ok:
            return PassResult(module, diagnostics=entry_verification.diagnostics)
        current = module
        changed = False
        operations_removed = 0
        pairs_cancelled = 0
        rotations_merged = 0
        zero_sums_removed = 0
        for compiler_pass in _independent_passes(verify_output=False):
            result = compiler_pass.run(current)
            if result.diagnostics:
                return PassResult(module, diagnostics=result.diagnostics)
            changed |= result.changed
            operations_removed += int(result.statistics.get("operations_removed", 0))
            pairs_cancelled += int(result.statistics.get("pairs_cancelled", 0))
            rotations_merged += int(result.statistics.get("rotations_merged", 0))
            zero_sums_removed += int(result.statistics.get("zero_sums_removed", 0))
            current = result.module
        if not changed:
            return PassResult(
                module,
                changed=False,
                preserved_analyses=frozenset({"def_use", "qubit_lifetime"}),
                statistics={"component_passes": 4},
            )
        exit_verification = verify_module(current, circuit_ir_v1_schema_registry())
        if not exit_verification.ok:
            return PassResult(
                current,
                changed=True,
                diagnostics=exit_verification.diagnostics,
            )
        return PassResult(
            current,
            changed=True,
            statistics={
                "component_passes": 4,
                "operations_removed": operations_removed,
                "pairs_cancelled": pairs_cancelled,
                "rotations_merged": rotations_merged,
                "zero_sums_removed": zero_sums_removed,
            },
        )


def _independent_passes(*, verify_output: bool) -> tuple[CompilerPass, ...]:
    return (
        RemoveIdentityOperationsPass(verify_output=verify_output),
        CancelSelfInverseOperationsPass(verify_output=verify_output),
        MergeAdjacentRotationsPass(verify_output=verify_output),
        RemoveIdentityOperationsPass(verify_output=verify_output),
    )


def phase2_batch_a_passes(*, fused: bool = False) -> tuple[CompilerPass, ...]:
    """Return the approved deterministic Batch A pass sequence."""

    if fused:
        return (StaticCanonicalizationPass(),)
    return _independent_passes(verify_output=True)


__all__ = [
    "CancelSelfInverseOperationsPass",
    "MergeAdjacentRotationsPass",
    "RemoveIdentityOperationsPass",
    "StaticCanonicalizationPass",
    "phase2_batch_a_passes",
]
