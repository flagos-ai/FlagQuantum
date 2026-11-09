"""Finite gate-call structure over the existing Program SSA authority.

Proposal 103 prerequisite 1.1 admits calls and explicit returns. Register words,
workspace allocation and numerical execution remain separate responsibilities.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

from ..diagnostics import LevelDiagnostic, _diagnostic
from .model import FunctionRecord, OpcodeSignature, Operation, ProgramRecord
from .physical import is_physical_record


def has_call_profile(record: ProgramRecord) -> bool:
    """Whether any field requests finite-call or explicit-return semantics."""
    return any(
        function.registers is not None
        or any(
            block.return_values is not None
            or any(value.kind == "register" for value in block.arguments)
            or any(
                op.opcode
                in {
                    "call",
                    "split_register",
                    "join_register",
                    "allocate_register",
                    "free_register",
                }
                or op.callee is not None
                or op.register_type is not None
                or op.workspace_dirty is not None
                or any(value.kind == "register" for value in op.results)
                for op in block.operations
            )
            for block in function.blocks
        )
        for function in record.functions
    )


def call_signature(
    operation: Operation, functions: Mapping[str, FunctionRecord]
) -> OpcodeSignature | None:
    """Resolve types from the callee's actual entry and explicit return values."""
    callee = functions.get(operation.callee or "")
    if callee is None or len(callee.blocks) != 1:
        return None
    block = callee.blocks[0]
    if block.return_values is None:
        return None
    return OpcodeSignature(
        "call",
        operand_kinds=tuple(value.kind for value in block.arguments),
        result_kinds=tuple(value.kind for value in block.return_values),
    )


def verify_call_structure(record: ProgramRecord) -> list[LevelDiagnostic]:
    """Check all definitions and cycles, including unreachable callees."""
    diagnostics: list[LevelDiagnostic] = []
    functions = {function.name: function for function in record.functions}
    if len(functions) != len(record.functions) or any(not name for name in functions):
        diagnostics.append(
            _diagnostic(
                "program.call_callee", "callee names must be unique and nonempty"
            )
        )
    if is_physical_record(record):
        diagnostics.append(
            _diagnostic(
                "program.call_structure", "physical and finite-call profiles cannot mix"
            )
        )
    dependencies: dict[str, set[str]] = {name: set() for name in functions}
    for function in record.functions:
        if len(function.blocks) != 1:
            diagnostics.append(
                _diagnostic(
                    "program.call_structure", f"{function.name!r} needs one block"
                )
            )
        for block in function.blocks:
            if (
                block.return_values is None
                or block.terminator != "return"
                or block.edges
            ):
                diagnostics.append(
                    _diagnostic(
                        "program.call_structure",
                        f"{function.name!r} needs an explicit return",
                    )
                )
            identities: set[tuple[str, int]] = set()
            for value in (
                *block.arguments,
                *(v for op in block.operations for v in op.results),
            ):
                identity = (value.scope, value.index)
                if not value.scope or type(value.index) is not int or value.index < 0:
                    diagnostics.append(
                        _diagnostic(
                            "value.identity", f"invalid call-profile value {value!r}"
                        )
                    )
                if identity in identities:
                    diagnostics.append(
                        _diagnostic(
                            "program.value_redefined", f"redefined value {identity!r}"
                        )
                    )
                identities.add(identity)
            for operation in block.operations:
                if operation.opcode != "call":
                    if operation.callee is not None:
                        diagnostics.append(
                            _diagnostic(
                                "program.call_structure",
                                "only call operations can name a callee",
                            )
                        )
                    continue
                if not operation.callee or operation.callee not in functions:
                    diagnostics.append(
                        _diagnostic(
                            "program.call_callee",
                            f"unknown callee {operation.callee!r}",
                        )
                    )
                else:
                    dependencies[function.name].add(operation.callee)
                if (
                    operation.parameters
                    or operation.literal is not None
                    or operation.condition_clauses
                    or operation.physical_data is not None
                ):
                    diagnostics.append(
                        _diagnostic(
                            "program.call_structure",
                            "calls cannot carry primitive parameters, conditions or physical data",
                        )
                    )
    if _has_cycle(dependencies):
        diagnostics.append(
            _diagnostic("program.call_structure", "finite calls cannot contain cycles")
        )
    return diagnostics


def _has_cycle(dependencies: Mapping[str, set[str]]) -> bool:
    remaining = {name: len(children) for name, children in dependencies.items()}
    parents: dict[str, list[str]] = {name: [] for name in dependencies}
    for name, children in dependencies.items():
        for child in children:
            parents[child].append(name)
    ready = deque(name for name, count in remaining.items() if count == 0)
    visited = 0
    while ready:
        child = ready.popleft()
        visited += 1
        for parent in parents[child]:
            remaining[parent] -= 1
            if remaining[parent] == 0:
                ready.append(parent)
    return visited != len(dependencies)
