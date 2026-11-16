"""Private fresh-register producer for the shared checked synthesis workflow."""

from collections.abc import Mapping, Sequence
from typing import Literal

from ..compiler._construction.selection import (
    Constraints,
    ConstructionBudget,
    select_register_program,
)
from ..compiler._construction.target import Target
from ..compiler._hybrid.model import HybridProgram
from ..runtime.options import ExecutionOptions
from ..runtime.planner.estimates import (
    exact_basis_mps_bond,
    exact_interval_mps_bond,
)
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
    source_format: Literal["register", "qmod", "python"] = "register",
    source_name: str | None = None,
    source_version: str | None = None,
    fractional_bits: Mapping[str, int] | None = None,
    signed_registers: tuple[str, ...] = (),
    deferred_measurements: bool = False,
    exact_mps: bool = False,
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
        include_boolean=source_format in {"python", "qmod"},
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
        record_layouts = tuple(
            (
                op.attributes["name"],
                op.attributes["record_type"],
                op.attributes["fields"],
            )
            for op in program.body.blocks[0].operations
            if op.name == "qreg.declare" and "record_type" in op.attributes
        )
        if record_layouts:
            report["qmod_records"] = record_layouts
        if fractional_bits:
            report["qmod_fractional_bits"] = fractional_bits
        if deferred_measurements:
            report["qmod_measurement_profile"] = "deferred_private_measurements"
        if signed_registers:
            report["qmod_signed_registers"] = signed_registers
    formats = tuple(
        (op.attributes["name"], *op.attributes["numeric_format"])
        for op in program.body.blocks[0].operations
        if op.name == "qreg.declare" and "numeric_format" in op.attributes
    )
    if formats:
        report["numeric_formats"] = formats
    selected = selection.selected
    if (
        source_format == "python"
        and selected is not None
        and selected.target is not None
        and selected.target.submitted.n_wires > 32
        and not any(
            gate.metadata.get("is_channel")
            for gate in selected.target.submitted.instructions
        )
    ):
        submitted = selected.target.submitted
        if exact_basis_mps_bond(submitted) == 1:
            exact_mps = True
            report.update(
                execution_profile="exact_computational_basis_mps",
                intermediate_mps_bond_bound=1,
            )
        else:
            bond = exact_interval_mps_bond(
                tuple(gate.wires for gate in submitted.instructions)
            )
            if bond is not None:
                exact_mps = True
                report.update(
                    execution_profile="exact_independent_interval_mps",
                    intermediate_mps_bond_bound=bond,
                )
    return _complete(
        selection,
        report,
        memory_limit_bytes=memory_limit_bytes,
        memory_limit_origin=memory_limit_origin,
        exact_mps=exact_mps,
        numeric_formats=formats,
    )
