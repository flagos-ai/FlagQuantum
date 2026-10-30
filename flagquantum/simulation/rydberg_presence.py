"""Private deterministic seams for one native Rydberg presence trajectory."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch

from ..core.geometry import same_time
from ..core.ir.program.model import ProgramRecord
from ..core.ir.program.physical import RydbergContext, RydbergPresence
from ..errors import CapabilityError, ExecutionError
from ..operators import SuperOperator
from .lindblad import _evolve_density_matrix
from .lindblad_integrators import DEFAULT_INTEGRATOR
from .lindblad_request import _require_density_matrices
from .rydberg_model import native_model, native_record


def _presence_model(
    record: ProgramRecord,
    times: Any,
    *,
    parameter_values: Any = None,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, Any], list[dict[str, float]] | None, int]:
    """Assemble the unchanged full control schedule; never restart its clock."""
    checked = native_record(record)
    if checked is None or not isinstance(checked.physical_context, RydbergContext):
        raise CapabilityError("presence evolution requires a native Rydberg source")
    function = checked.functions[0]
    block = function.blocks[0]
    observations = [
        op.physical_data
        for op in block.operations
        if isinstance(op.physical_data, RydbergPresence)
    ]
    if len(observations) != 1:
        raise CapabilityError("presence evolution requires exactly one observation")
    controls = replace(
        checked,
        functions=(
            replace(
                function,
                blocks=(
                    replace(
                        block,
                        operations=tuple(
                            op
                            for op in block.operations
                            if not isinstance(op.physical_data, RydbergPresence)
                        ),
                    ),
                ),
            ),
        ),
    )
    numeric, bindings = native_model(
        controls,
        times,
        dtype=torch.complex128,
        device=device,
        parameter_values=parameter_values,
    )
    grid = numeric["times"]
    matches = [
        i
        for i, value in enumerate(grid.tolist())
        if same_time(value, observations[0].at, precision="float64")
    ]
    if len(matches) != 1 or not 0 < matches[0] < len(grid) - 1:
        raise ExecutionError(
            "presence time must match one distinct internal grid point"
        )
    return numeric, bindings, matches[0]


def _presence_operators(
    record: ProgramRecord, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    checked = native_record(record)
    if checked is None or not isinstance(checked.physical_context, RydbergContext):
        raise CapabilityError("presence operators require a native Rydberg source")
    if len(checked.physical_context.atoms) > 2:
        raise CapabilityError("presence dynamics supports one or two actors")
    observations = [
        op.physical_data
        for op in checked.functions[0].blocks[0].operations
        if isinstance(op.physical_data, RydbergPresence)
    ]
    if len(observations) != 1:
        raise CapabilityError("presence operators require exactly one observation")
    observation = observations[0]
    local = (
        torch.diag(
            torch.tensor(
                [1, 1, math.sqrt(1 - observation.efficiency)],
                dtype=torch.complex128,
                device=device,
            )
        ),
        torch.diag(
            torch.tensor(
                [0, 0, math.sqrt(observation.efficiency)],
                dtype=torch.complex128,
                device=device,
            )
        ),
    )
    operators = []
    for operator in local:
        full = torch.ones((1, 1), dtype=torch.complex128, device=device)
        for atom, _ in checked.physical_context.atoms:
            full = torch.kron(
                full,
                (
                    operator
                    if atom == observation.atom
                    else torch.eye(3, dtype=torch.complex128, device=device)
                ),
            )
        operators.append(full)
    return operators[0], operators[1]


def _presence_segment(
    numeric: Mapping[str, Any],
    initial: torch.Tensor,
    start: int,
    stop: int,
    *,
    method: str,
    solve_tolerance: float | None,
) -> torch.Tensor:
    request = dict(numeric)
    # Slice already-bound intervals at their original absolute times. No native
    # source is cropped or rebound, and no boundary is inserted into the grid.
    request["times"] = numeric["times"][start : stop + 1]
    request["bound_hamiltonians"] = numeric["bound_hamiltonians"][start:stop]
    result = _evolve_density_matrix(
        **request,
        initial_state=initial,
        dtype=torch.complex128,
        device=initial.device,
        return_density_matrices=True,
        method=method,
        solve_tolerance=solve_tolerance,
    )
    assert result.density_matrices is not None
    return result.density_matrices


def _rydberg_presence_prefix(
    numeric: Mapping[str, Any],
    initial: torch.Tensor,
    boundary: int,
    *,
    method: str = DEFAULT_INTEGRATOR,
    solve_tolerance: float | None = None,
) -> torch.Tensor:
    """Evolve to the observation without sampling or constructing evidence."""
    if type(boundary) is not int or not 0 < boundary < len(numeric["times"]) - 1:
        raise ExecutionError("presence boundary must be an internal grid index")
    if (
        initial.ndim != 2
        or initial.dtype != torch.complex128
        or initial.device.type not in {"cpu", "flagos"}
        or initial.requires_grad
        or any(matrix.requires_grad for matrix in numeric["bound_hamiltonians"])
    ):
        raise ExecutionError(
            "presence evolution requires a CPU complex128 density without gradients"
        )
    return _presence_segment(
        numeric, initial, 0, boundary, method=method, solve_tolerance=solve_tolerance
    )


def _continue_rydberg_presence(
    record: ProgramRecord,
    numeric: Mapping[str, Any],
    prefix: torch.Tensor,
    selected_index: int,
    *,
    method: str = DEFAULT_INTEGRATOR,
    solve_tolerance: float | None = None,
) -> torch.Tensor:
    """Apply a Runtime-selected branch using its original positive trace."""
    if type(selected_index) is not int or selected_index not in (0, 1):
        raise ExecutionError("selected presence outcome must be index 0 or 1")
    boundary = len(prefix) - 1
    if (
        not 0 < boundary < len(numeric["times"]) - 1
        or prefix.ndim != 3
        or prefix.dtype != torch.complex128
        or prefix.device.type not in {"cpu", "flagos"}
        or prefix.requires_grad
    ):
        raise ExecutionError("presence prefix must be a CPU complex128 trajectory")
    checked = native_record(record)
    assert checked is not None
    observations = [
        op.physical_data
        for op in checked.functions[0].blocks[0].operations
        if isinstance(op.physical_data, RydbergPresence)
    ]
    matches = [
        i
        for i, time in enumerate(numeric["times"].tolist())
        if len(observations) == 1
        and same_time(time, observations[0].at, precision="float64")
    ]
    if matches != [boundary]:
        raise ExecutionError("presence continuation boundary differs from source time")
    _require_density_matrices(prefix, tolerance=1e-10)
    operator = _presence_operators(record, prefix.device)[selected_index]
    branch = SuperOperator.from_kraus((operator,)).apply(prefix[-1])
    probability = torch.trace(branch).real
    if not bool(torch.isfinite(probability)) or probability <= 0:
        raise ExecutionError("selected presence branch must have positive finite trace")
    suffix = _presence_segment(
        numeric,
        branch / probability,
        boundary,
        len(numeric["times"]) - 1,
        method=method,
        solve_tolerance=solve_tolerance,
    )
    return torch.cat((prefix[:-1], suffix))
