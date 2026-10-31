"""Private fresh-register producer for the shared checked synthesis workflow."""

from collections.abc import Sequence
from typing import Literal

from ..compiler._construction.selection import (
    Constraints,
    ConstructionBudget,
    select_register_program,
)
from ..compiler._construction.target import Target
from ..compiler._hybrid.model import HybridProgram
from ..runtime.options import ExecutionOptions
from .synthesis import Program, _complete

_CONSTRAINTS = Constraints()
_BUDGET = ConstructionBudget()
_TARGET = Target()


def synthesize_registers(
    program: HybridProgram,
    functions: Sequence[HybridProgram] = (),
    *,
    constraints: Constraints = _CONSTRAINTS,
    budget: ConstructionBudget = _BUDGET,
    target: Target = _TARGET,
    objective: Literal["width", "depth"] = "width",
    memory_limit_bytes: int,
    memory_limit_origin: Literal["profile_default", "explicit"] = "explicit",
    source_format: Literal["register", "qmod"] = "register",
    source_name: str | None = None,
    source_version: str | None = None,
) -> Program:
    """Plan verified owned records; require an explicit Runtime memory ceiling.

    Source parsing and source-version authentication do not belong here. The
    caller retains native source context; Compiler identities describe owned
    semantic records only. No text, SDK object, or uploaded state is executed.
    """
    # Reuse Runtime's option validation before construction; booleans and absent
    # limits must not silently turn explicit source admission into an open cap.
    if memory_limit_bytes is None:
        raise ValueError("register admission requires an explicit memory ceiling")
    ExecutionOptions(memory_limit_bytes=memory_limit_bytes)
    selection = select_register_program(
        program,
        functions,
        constraints=constraints,
        budget=budget,
        target=target,
        objective=objective,
    )
    report: dict[str, object] = {
        "function": program.name,
        "source_identities": (
            program.semantic_identity,
            *(f.semantic_identity for f in sorted(functions, key=lambda f: f.name)),
        ),
        "source_locations": tuple(
            dict.fromkeys(
                op.location
                for op in program.body.blocks[0].operations
                if op.location is not None
            )
        ),
        "scope": "actual_cpu_preparation_arithmetic_cleanup",
        "source_format": source_format,
    }
    if source_format == "qmod":
        report.update(source_name=source_name, source_version=source_version)
    return _complete(
        selection,
        report,
        memory_limit_bytes=memory_limit_bytes,
        memory_limit_origin=memory_limit_origin,
    )
