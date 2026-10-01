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


def _cpu_statevector_batch_bounded_initial_state_enabled() -> bool:
    """Whether wide zero-state batches materialize only one execution window."""

    return bool(
        _environment_flag(
            "FQ_CPU_STATEVECTOR_BATCH_BOUNDED_INITIAL_STATE", default=True
        )
    )


def _cpu_statevector_batch_chunk_size_for(
    *, batch_size: int, amplitudes: int, element_size: int, device_type: str
) -> int:
    """Return the rows whose logical state fits the measured working-set budget."""

    if (
        device_type != "cpu"
        or batch_size <= 1
        or not _cpu_statevector_batch_chunking_enabled()
    ):
        return batch_size
    row_bytes = int(amplitudes) * int(element_size)
    return min(
        batch_size,
        max(1, _CPU_STATEVECTOR_BATCH_CHUNK_BUDGET_BYTES // row_bytes),
    )


def _cpu_statevector_batch_chunk_size(state: torch.Tensor) -> int:
    """Return the rows whose logical state fits the measured working-set budget."""

    return _cpu_statevector_batch_chunk_size_for(
        batch_size=int(state.shape[0]),
        amplitudes=int(state.shape[1]),
        element_size=int(state.element_size()),
        device_type=state.device.type,
    )


def _initial_state_batch_window(circuit: Circuit, batch_size: int) -> torch.Tensor:
    """Return a cached zero-state window without materializing the full batch."""

    workspace = circuit._initial_state_batch_window_workspace
    expected_shape = (int(batch_size), 2**circuit.n_wires)
    if workspace is None or tuple(workspace.shape) != expected_shape:
        workspace = torch.zeros(
            expected_shape,
            dtype=circuit.dtype,
            device=circuit.device,
        )
        workspace[:, 0] = 1
        circuit._initial_state_batch_window_workspace = workspace
    return workspace


def _statevector_batch_input(
    circuit: Circuit,
) -> tuple[int, int, bool, torch.Tensor]:
    """Resolve logical batch size and the smallest safe execution input."""

    batch_size = (
        int(circuit._inputs.shape[0])
        if circuit._inputs is not None and circuit._inputs.ndim > 1
        else int(circuit.bsz)
    )
    initial_window_size = _cpu_statevector_batch_chunk_size_for(
        batch_size=batch_size,
        amplitudes=2**circuit.n_wires,
        element_size=torch.empty((), dtype=circuit.dtype).element_size(),
        device_type=torch.device(circuit.device).type,
    )
    uses_bounded_zero_state = (
        _cpu_statevector_batch_bounded_initial_state_enabled()
        and circuit._inputs is None
        and initial_window_size < batch_size
    )
    output = (
        _initial_state_batch_window(circuit, initial_window_size)
        if uses_bounded_zero_state
        else circuit.initial_state()
    )
    return batch_size, initial_window_size, uses_bounded_zero_state, output


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
