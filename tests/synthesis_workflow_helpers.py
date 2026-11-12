"""Shared synthesis execution assertions, independent of collected test modules."""

from collections.abc import Mapping
from typing import TypeVar

import torch

from flagquantum.compiler._construction.constraints import CircuitLimits
from flagquantum.compiler._construction.selection import Constraints, ConstructionBudget
from flagquantum.ecosystem._qmod.adapter import SUPPORTED_SOURCE_VERSION, parse_qmod
from flagquantum.runtime.execution_plan import ExecutionPlan
from flagquantum.runtime.result import ExecutionResult
from flagquantum.services._synthesis_registers import synthesize_registers
from flagquantum.services.synthesis import Program

MEMORY = 512 * 1024**2
Values = TypeVar("Values", bound=tuple[int, ...])


def native(
    source: str,
    *,
    constraints: Constraints | None = None,
    budget: ConstructionBudget | None = None,
) -> Program:
    imported = parse_qmod(
        source, filename="workflow.qmod", source_version=SUPPORTED_SOURCE_VERSION
    )
    return synthesize_registers(
        imported.program,
        imported.functions,
        memory_limit_bytes=MEMORY,
        constraints=(
            constraints if constraints is not None else Constraints(CircuitLimits())
        ),
        budget=budget if budget is not None else ConstructionBudget(),
    )


def check_state(
    program: Program,
    result: ExecutionResult,
    expected: Mapping[Values, complex],
) -> None:
    """Compare every complex amplitude, including all mapped clean workspace."""
    assert isinstance(result.plan, ExecutionPlan)
    assert result.plan.identity == program.execution.identity
    state = result.statevector().reshape(-1)
    residual = state.clone()
    row = program.selection.selected
    assert row is not None and row.target is not None
    for values, amplitude in expected.items():
        index = 0
        for value, ports in zip(values, program.registers.values(), strict=True):
            for bit, logical in enumerate(reversed(ports)):
                physical = row.target.layout.physical_of(logical)
                index |= ((value >> bit) & 1) << (
                    program.circuit.n_wires - 1 - physical
                )
        residual[index] -= amplitude
    assert float(torch.linalg.vector_norm(residual)) <= 1e-12
    output_ports = {q for ports in program.registers.values() for q in ports}
    # Reshape/permutation uses actual final physical slots. It avoids a dense
    # integer index vector at the larger original-file widths.
    workspace = tuple(
        row.target.layout.physical_of(q)
        for q in range(program.circuit.n_wires)
        if q not in output_ports
    )
    outputs = tuple(q for q in range(program.circuit.n_wires) if q not in workspace)
    view = (
        state.reshape((2,) * program.circuit.n_wires)
        .permute(*outputs, *workspace)
        .reshape(1 << len(outputs), 1 << len(workspace))
    )
    assert float(torch.linalg.vector_norm(view[:, 1:]).square()) <= 1e-12
