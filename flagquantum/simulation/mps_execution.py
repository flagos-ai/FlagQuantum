"""Native matrix-product-state execution for FlagQuantum."""

from __future__ import annotations

import os
from typing import Any

import torch

from ..circuit import _gate_matrix
from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir

_DENSE_Z_SUM_WEIGHT_CACHE: dict[
    tuple[int, tuple[int, ...], str, torch.dtype], torch.Tensor
] = {}
_MPS_INSTRUCTION_SCHEDULE_CACHE: dict[tuple[Any, ...], tuple[tuple[int, ...], ...]] = {}
from .mps_models import (  # noqa: E402
    CompiledMPSProgram,
    MPSAdaptiveRunResult,
    MPSConfig,
    MPSMonteCarloResult,
)
from .mps_state import MPSState  # noqa: E402


def run_mps(
    circuit_or_ir: Any,
    *,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    fuse_single_qubit: bool = True,
    dense_observable_wires: int = 12,
) -> MPSState:
    """Execute FlagQuantum IR with the native MPS backend."""

    if hasattr(circuit_or_ir, "to_ir"):
        ir = ensure_circuit_ir(circuit_or_ir)
        if getattr(circuit_or_ir, "_inputs", None) is not None:
            state = circuit_or_ir.initial_state()
            mps = MPSState.from_statevector(
                state,
                ir.n_wires,
                config=MPSConfig(
                    max_bond=max_bond,
                    cutoff=cutoff,
                    dense_observable_wires=dense_observable_wires,
                ),
            )
        else:
            mps = MPSState.zero(
                ir.n_wires,
                bsz=getattr(circuit_or_ir, "bsz", bsz),
                device=getattr(circuit_or_ir, "device", device),
                dtype=getattr(circuit_or_ir, "dtype", dtype),
                config=MPSConfig(
                    max_bond=max_bond,
                    cutoff=cutoff,
                    dense_observable_wires=dense_observable_wires,
                ),
            )
    elif isinstance(circuit_or_ir, CircuitIR):
        ir = ensure_circuit_ir(circuit_or_ir)
        mps = MPSState.zero(
            ir.n_wires,
            bsz=bsz,
            device=device,
            dtype=dtype,
            config=MPSConfig(
                max_bond=max_bond,
                cutoff=cutoff,
                dense_observable_wires=dense_observable_wires,
            ),
        )
    else:
        raise TypeError("run_mps expects a Circuit or CircuitIR.")

    instructions = tuple(ir)
    binding_source = getattr(circuit_or_ir, "_parameter_bindings", None)
    parameter_bindings = None if binding_source is None else binding_source.values()
    program_cache = getattr(circuit_or_ir, "_backend_programs", None)
    signature = (
        mps.n_wires,
        mps.bsz,
        str(mps.device),
        mps.dtype,
        max_bond,
        float(cutoff),
        bool(fuse_single_qubit),
    )
    program_key = ("mps", signature)
    program = None if program_cache is None else program_cache.get(program_key)
    if program is None:
        program = CompiledMPSProgram.compile(
            instructions,
            signature=signature,
            fuse_single_qubit=fuse_single_qubit,
        )
        if program_cache is not None:
            program_cache[program_key] = program
    for operation in program.operations:
        group = operation.instruction_indices
        instruction = instructions[group[0]]
        if operation.kind == "adjacent_two_bucket":
            spatial_bucket_enabled = os.getenv(
                "FQ_MPS_SPATIAL_BUCKET", "1"
            ).strip().lower() not in {"0", "false", "off", "no"}
            if not spatial_bucket_enabled:
                for index in group:
                    mps.apply_instruction(instructions[index], parameter_bindings)
                continue
            matrices = [
                _gate_matrix(
                    instructions[index],
                    bsz=mps.bsz,
                    device=mps.device,
                    dtype=mps.dtype,
                    parameter_bindings=parameter_bindings,
                )
                for index in group
            ]
            mps.apply_two_bucket(
                matrices,
                [min(instructions[index].wires) for index in group],
            )
            continue
        if len(group) == 1:
            mps.apply_instruction(instruction, parameter_bindings)
            continue
        matrix = _gate_matrix(
            instruction,
            bsz=mps.bsz,
            device=mps.device,
            dtype=mps.dtype,
            parameter_bindings=parameter_bindings,
        )
        for index in group[1:]:
            matrix = _compose_one_qubit_matrices(
                _gate_matrix(
                    instructions[index],
                    bsz=mps.bsz,
                    device=mps.device,
                    dtype=mps.dtype,
                    parameter_bindings=parameter_bindings,
                ),
                matrix,
                bsz=mps.bsz,
            )
        mps.apply_one(matrix, int(instruction.wires[0]))
    return mps


def _mps_instruction_schedule(
    instructions: tuple[Instruction, ...], fuse_single_qubit: bool
) -> tuple[tuple[int, ...], ...]:
    key = (
        bool(fuse_single_qubit),
        tuple(
            (
                instruction.name,
                instruction.wires,
                tuple(sorted(instruction.params)),
                instruction.matrix is not None,
                bool(instruction.metadata.get("is_channel")),
            )
            for instruction in instructions
        ),
    )
    cached = _MPS_INSTRUCTION_SCHEDULE_CACHE.get(key)
    if cached is not None:
        return cached

    groups: list[tuple[int, ...]] = []
    index = 0
    while index < len(instructions):
        instruction = instructions[index]
        group = [index]
        if fuse_single_qubit and _is_fusible_one_qubit_instruction(instruction):
            wire = int(instruction.wires[0])
            next_index = index + 1
            while (
                next_index < len(instructions)
                and _is_fusible_one_qubit_instruction(instructions[next_index])
                and int(instructions[next_index].wires[0]) == wire
            ):
                group.append(next_index)
                next_index += 1
        elif (
            len(instruction.wires) == 2
            and abs(instruction.wires[0] - instruction.wires[1]) == 1
            and instruction.wires[0] < instruction.wires[1]
        ):
            occupied = set(instruction.wires)
            next_index = index + 1
            while next_index < len(instructions):
                candidate = instructions[next_index]
                if (
                    len(candidate.wires) != 2
                    or abs(candidate.wires[0] - candidate.wires[1]) != 1
                    or candidate.wires[0] > candidate.wires[1]
                    or occupied.intersection(candidate.wires)
                ):
                    break
                group.append(next_index)
                occupied.update(candidate.wires)
                next_index += 1
        groups.append(tuple(group))
        index = group[-1] + 1
    schedule = tuple(groups)
    _MPS_INSTRUCTION_SCHEDULE_CACHE[key] = schedule
    return schedule


def _is_fusible_one_qubit_instruction(instruction: Instruction) -> bool:
    return (
        len(instruction.wires) == 1
        and not instruction.metadata.get("is_channel")
        and instruction.name not in {"measure", "reset"}
    )


def _compose_one_qubit_matrices(
    new_matrix: torch.Tensor,
    current_matrix: torch.Tensor,
    *,
    bsz: int,
) -> torch.Tensor:
    if new_matrix.ndim == 2 and current_matrix.ndim == 2:
        return new_matrix @ current_matrix
    if new_matrix.ndim == 2:
        new_matrix = new_matrix.expand(int(bsz), -1, -1)
    if current_matrix.ndim == 2:
        current_matrix = current_matrix.expand(int(bsz), -1, -1)
    return torch.bmm(new_matrix, current_matrix)


def run_mps_adaptive(
    circuit_or_ir: Any,
    *,
    global_error_budget: float = 0.0,
    initial_max_bond: int | None = 1,
    max_bond_cap: int | None = None,
    cutoff: float = 0.0,
    growth_factor: float = 2.0,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> MPSAdaptiveRunResult:
    """Run MPS with one adaptive rerun driven by observed truncation hotspots."""

    initial = run_mps(
        circuit_or_ir,
        bsz=bsz,
        device=device,
        dtype=dtype,
        max_bond=initial_max_bond,
        cutoff=cutoff,
    )
    initial_plan = initial.adaptive_bond_plan(
        global_error_budget=global_error_budget,
        growth_factor=growth_factor,
    )
    if initial_plan.budget_satisfied:
        return MPSAdaptiveRunResult(
            state=initial,
            initial_state=initial,
            initial_plan=initial_plan,
            final_plan=initial_plan,
            refinement_plan=initial.local_refinement_plan(
                global_error_budget=global_error_budget,
                growth_factor=growth_factor,
            ),
            rerun=False,
        )

    suggested = initial_plan.suggested_max_bond
    if max_bond_cap is not None:
        suggested = min(int(max_bond_cap), int(suggested))
    if initial_max_bond is not None and suggested <= int(initial_max_bond):
        suggested = int(initial_max_bond)

    final = run_mps(
        circuit_or_ir,
        bsz=bsz,
        device=device,
        dtype=dtype,
        max_bond=suggested,
        cutoff=cutoff,
    )
    final_plan = final.adaptive_bond_plan(
        global_error_budget=global_error_budget,
        growth_factor=growth_factor,
    )
    return MPSAdaptiveRunResult(
        state=final,
        initial_state=initial,
        initial_plan=initial_plan,
        final_plan=final_plan,
        refinement_plan=initial.local_refinement_plan(
            global_error_budget=global_error_budget,
            growth_factor=growth_factor,
        ),
        rerun=True,
    )


def run_noisy_mps_trajectory(
    circuit_or_ir: Any,
    noise_model: Any | None = None,
    *,
    generator: torch.Generator | None = None,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    max_bond: int | None = None,
    cutoff: float = 0.0,
) -> MPSState:
    """Execute a noisy circuit as one sampled MPS quantum trajectory."""

    from .noise import lower_noise_model

    lowered = lower_noise_model(circuit_or_ir, noise_model)
    config = MPSConfig(max_bond=max_bond, cutoff=cutoff)
    if (
        hasattr(circuit_or_ir, "initial_state")
        and getattr(circuit_or_ir, "_inputs", None) is not None
    ):
        mps = MPSState.from_statevector(
            circuit_or_ir.initial_state(),
            lowered.n_wires,
            config=config,
        )
    else:
        mps = MPSState.zero(
            lowered.n_wires,
            bsz=bsz if not hasattr(circuit_or_ir, "bsz") else circuit_or_ir.bsz,
            device=(
                device if not hasattr(circuit_or_ir, "device") else circuit_or_ir.device
            ),
            dtype=dtype if not hasattr(circuit_or_ir, "dtype") else circuit_or_ir.dtype,
            config=config,
        )
    for instruction in lowered:
        if instruction.metadata.get("is_channel"):
            mps.apply_channel_trajectory(
                instruction.matrix,
                instruction.wires,
                generator=generator,
            )
        else:
            mps.apply_instruction(instruction)
    return mps


def run_noisy_mps(
    circuit_or_ir: Any,
    noise_model: Any | None = None,
    *,
    trajectories: int = 32,
    generator: torch.Generator | None = None,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    max_bond: int | None = None,
    cutoff: float = 0.0,
) -> MPSMonteCarloResult:
    """Run multiple noisy MPS trajectories and aggregate Z expectations."""

    states = []
    values = []
    for _ in range(int(trajectories)):
        state = run_noisy_mps_trajectory(
            circuit_or_ir,
            noise_model,
            generator=generator,
            bsz=bsz,
            device=device,
            dtype=dtype,
            max_bond=max_bond,
            cutoff=cutoff,
        )
        states.append(state)
        values.append(state.expectation_z())
    stacked = torch.stack(values, dim=0)
    variance = (
        torch.var(stacked, dim=0, unbiased=False)
        if int(trajectories) > 1
        else torch.zeros_like(stacked[0])
    )
    return MPSMonteCarloResult(
        trajectories=tuple(states),
        expectation_z_mean=stacked.mean(dim=0),
        expectation_z_variance=variance,
        n_trajectories=int(trajectories),
    )
