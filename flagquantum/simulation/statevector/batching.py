"""Private CPU statevector batch sizing and parameter-window helpers."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Number
from typing import TYPE_CHECKING

import torch

from ...core.ir import Instruction
from .operations import _environment_flag, _gate_parameter_tensor
from .program import _StatevectorFusedGateStep

if TYPE_CHECKING:
    from ...circuit import Circuit


# The Apple-arm64 18-wire/batch-32 corpus measured this budget at 1.05x-1.27x
# over one monolithic batch across all five representative workloads. This is a
# logical state budget, not a process-RSS promise; individual kernels still own
# their temporary storage.
_CPU_STATEVECTOR_BATCH_CHUNK_BUDGET_BYTES = 64 * 1024 * 1024


def _cpu_statevector_batch_chunking_enabled() -> bool:
    """Whether wide CPU parameter batches may execute in bounded windows."""

    return bool(_environment_flag("FQ_CPU_STATEVECTOR_BATCH_CHUNKING", default=True))


def _cpu_statevector_batch_chunk_size(state: torch.Tensor) -> int:
    """Return the rows whose logical state fits the measured working-set budget."""

    batch_size = int(state.shape[0])
    if (
        state.device.type != "cpu"
        or batch_size <= 1
        or not _cpu_statevector_batch_chunking_enabled()
    ):
        return batch_size
    row_bytes = int(state.shape[1]) * int(state.element_size())
    return min(
        batch_size,
        max(1, _CPU_STATEVECTOR_BATCH_CHUNK_BUDGET_BYTES // row_bytes),
    )


def _initial_state(circuit: Circuit) -> torch.Tensor:
    """Resolve or create a Circuit's local statevector input."""

    if circuit._inputs is not None:
        resolved = circuit._inputs.to(device=circuit.device, dtype=circuit.dtype)
        return resolved.reshape(1, -1) if resolved.ndim == 1 else resolved
    workspace = circuit._initial_state_workspace
    if workspace is None:
        workspace = torch.zeros(
            circuit.bsz,
            2**circuit.n_wires,
            dtype=circuit.dtype,
            device=circuit.device,
        )
        workspace[:, 0] = 1
        circuit._initial_state_workspace = workspace
    return workspace


def _gate_parameters(
    circuit: Circuit,
    instruction: Instruction,
    state: torch.Tensor,
    parameter_bindings: tuple[torch.Tensor, ...] | None,
) -> torch.Tensor | None:
    cacheable = parameter_bindings is None and all(
        isinstance(value, Number) for value in instruction.params.values()
    )
    if not cacheable:
        return _gate_parameter_tensor(
            instruction,
            state,
            parameter_bindings=parameter_bindings,
        )
    key = (id(instruction), str(state.device), state.dtype, int(state.shape[0]))
    cached = circuit._statevector_constant_parameters.get(key)
    if cached is None:
        cached = _gate_parameter_tensor(instruction, state, parameter_bindings=None)
        if cached is not None:
            circuit._statevector_constant_parameters[key] = cached
    return cached


def _rotation_region_angles(
    circuit: Circuit,
    steps: Sequence[_StatevectorFusedGateStep],
    state: torch.Tensor,
    parameter_bindings: tuple[torch.Tensor, ...] | None,
) -> torch.Tensor | None:
    """Collect batched angles, or decline fusion when a parameter is unavailable."""

    regions: list[torch.Tensor] = []
    for step in steps:
        parameters = tuple(
            _gate_parameters(circuit, instruction, state, parameter_bindings)
            for instruction in step.instructions
        )
        angles: list[torch.Tensor] = []
        for parameter in parameters:
            if parameter is None:
                return None
            angles.append(parameter[:, 0])
        regions.append(torch.stack(angles, dim=-1))
    return torch.stack(regions, dim=0) if regions else None
