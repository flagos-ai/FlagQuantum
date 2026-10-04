"""Private CPU statevector batch sizing and parameter-window helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Number
from typing import TYPE_CHECKING

import torch

from ...core.ir import Instruction
from ..gate_matrix import _parameter_batch_window
from .operations import _environment_flag, _gate_parameter_tensor
from .program import (
    _direct_batch_assembly_beneficial,
    _preallocated_batch_assembly_beneficial,
    _StatevectorControlledPhaseGraphStep,
    _StatevectorCrossWireDiagonalStep,
    _StatevectorCXSequenceStep,
    _StatevectorCZGraphStep,
    _StatevectorFusedGateStep,
    _StatevectorProgramStep,
)

if TYPE_CHECKING:
    from ...circuit import Circuit


# The Apple-arm64 18-wire/batch-32 corpus measured this budget at 1.05x-1.27x
# over one monolithic batch across all five representative workloads. This is a
# logical state budget, not a process-RSS promise; individual kernels still own
# their temporary storage.
_CPU_STATEVECTOR_BATCH_CHUNK_BUDGET_BYTES = 64 * 1024 * 1024

# The 18-wire/batch-32 complex128 memory corpus measured 32 MiB windows as a
# lower-RSS win for preallocated CX-sequence, cross-wire-diagonal, and
# controlled-phase-graph programs. Direct-assembly programs retain the general
# budget because smaller windows regressed their end-to-end timing.
_CPU_STATEVECTOR_BATCH_REDUCED_CHUNK_BUDGET_BYTES = 32 * 1024 * 1024


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


def _cpu_statevector_batch_preallocated_assembly_enabled() -> bool:
    """Whether inference windows write directly into their final result."""

    return bool(
        _environment_flag(
            "FQ_CPU_STATEVECTOR_BATCH_PREALLOCATED_ASSEMBLY", default=True
        )
    )


def _cpu_statevector_batch_direct_assembly_enabled() -> bool:
    """Whether owned final slices may also serve as execution windows."""

    return bool(
        _environment_flag("FQ_CPU_STATEVECTOR_BATCH_DIRECT_ASSEMBLY", default=True)
    )


def _cpu_statevector_batch_adaptive_budget_enabled() -> bool:
    """Whether measured program classes may use the smaller memory budget."""

    return bool(
        _environment_flag("FQ_CPU_STATEVECTOR_BATCH_ADAPTIVE_BUDGET", default=True)
    )


def _cpu_statevector_batch_budget_for_program(
    program: Sequence[_StatevectorProgramStep],
) -> int:
    """Return the measured logical-state budget for one compiled program."""

    reduced_candidate = _preallocated_batch_assembly_beneficial(program) and any(
        isinstance(
            step,
            (
                _StatevectorCXSequenceStep,
                _StatevectorControlledPhaseGraphStep,
                _StatevectorCrossWireDiagonalStep,
                _StatevectorCZGraphStep,
            ),
        )
        for step in program
    )
    if reduced_candidate and _cpu_statevector_batch_adaptive_budget_enabled():
        return min(
            _CPU_STATEVECTOR_BATCH_CHUNK_BUDGET_BYTES,
            _CPU_STATEVECTOR_BATCH_REDUCED_CHUNK_BUDGET_BYTES,
        )
    return _CPU_STATEVECTOR_BATCH_CHUNK_BUDGET_BYTES


def _execute_statevector_batch_windows(
    output: torch.Tensor,
    *,
    circuit: Circuit,
    program: Sequence[_StatevectorProgramStep],
    batch_size: int,
    chunk_size: int,
    bounded_zero_state: bool,
    preallocate: bool,
    execute: Callable[[torch.Tensor, bool], torch.Tensor],
) -> tuple[torch.Tensor, str]:
    """Execute bounded windows and avoid retaining them before final assembly."""

    direct = bool(
        bounded_zero_state
        and preallocate
        and _direct_batch_assembly_beneficial(program, circuit.n_qubits)
        and _cpu_statevector_batch_direct_assembly_enabled()
    )
    if bounded_zero_state:
        output = _materialize_bounded_zero_state(
            circuit,
            batch_size=batch_size,
            window_size=chunk_size,
            direct=direct,
        )
    chunks: list[torch.Tensor] = []
    assembled: torch.Tensor | None = output if direct else None
    for start in range(0, batch_size, chunk_size):
        stop = min(start + chunk_size, batch_size)
        window = (
            output[start:stop]
            if direct or not bounded_zero_state
            else output[: stop - start]
        )
        with _parameter_batch_window(start, stop, batch_size):
            chunk = execute(window, direct)
        if direct:
            if chunk.data_ptr() != window.data_ptr():
                window.copy_(chunk)
            continue
        if assembled is None and not chunks:
            if (
                preallocate
                and _cpu_statevector_batch_preallocated_assembly_enabled()
                and not chunk.requires_grad
            ):
                assembled = torch.empty(
                    (batch_size, *chunk.shape[1:]),
                    dtype=chunk.dtype,
                    device=chunk.device,
                )
        if assembled is None:
            chunks.append(chunk)
        else:
            assembled[start:stop].copy_(chunk)
    if assembled is None:
        return torch.cat(chunks, dim=0), "functional_cat"
    return assembled, "direct_preallocated" if direct else "preallocated_copy"


def _cpu_statevector_batch_chunk_size_for(
    *,
    batch_size: int,
    amplitudes: int,
    element_size: int,
    device_type: str,
    budget_bytes: int | None = None,
) -> int:
    """Return the rows whose logical state fits the measured working-set budget."""

    if (
        device_type != "cpu"
        or batch_size <= 1
        or not _cpu_statevector_batch_chunking_enabled()
    ):
        return batch_size
    row_bytes = int(amplitudes) * int(element_size)
    budget = (
        _CPU_STATEVECTOR_BATCH_CHUNK_BUDGET_BYTES
        if budget_bytes is None
        else int(budget_bytes)
    )
    return min(
        batch_size,
        max(1, budget // row_bytes),
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
    expected_shape = (int(batch_size), 2**circuit.n_qubits)
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
        amplitudes=2**circuit.n_qubits,
        element_size=torch.empty((), dtype=circuit.dtype).element_size(),
        device_type=torch.device(circuit.device).type,
    )
    uses_bounded_zero_state = (
        _cpu_statevector_batch_bounded_initial_state_enabled()
        and circuit._inputs is None
        and initial_window_size < batch_size
    )
    output = (
        torch.empty(
            (initial_window_size, 0), dtype=circuit.dtype, device=circuit.device
        )
        if uses_bounded_zero_state
        else circuit.initial_state()
    )
    return batch_size, initial_window_size, uses_bounded_zero_state, output


def _materialize_bounded_zero_state(
    circuit: Circuit,
    *,
    batch_size: int,
    window_size: int,
    direct: bool,
) -> torch.Tensor:
    """Create either the reusable input window or the owned final result."""

    if not direct:
        return _initial_state_batch_window(circuit, window_size)
    return torch.empty(
        (batch_size, 2**circuit.n_qubits),
        dtype=circuit.dtype,
        device=circuit.device,
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
            2**circuit.n_qubits,
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
