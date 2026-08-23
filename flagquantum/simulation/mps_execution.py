"""Native matrix-product-state execution for FlagQuantum."""

from __future__ import annotations

import os
from typing import Any

import torch

from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ..ops.gate_matrix import gate_matrix

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
                gate_matrix(
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
        matrix = gate_matrix(
            instruction,
            bsz=mps.bsz,
            device=mps.device,
            dtype=mps.dtype,
            parameter_bindings=parameter_bindings,
        )
        for index in group[1:]:
            matrix = _compose_one_qubit_matrices(
                gate_matrix(
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
    seed: int | None = None,
    trajectory_id: int = 0,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    max_bond: int | None = None,
    cutoff: float = 0.0,
) -> MPSState:
    """Execute a noisy circuit as one sampled MPS quantum trajectory."""

    from ..compilation.noise import lower_noise_model
    from ..runtime.trajectories import trajectory_generator

    if generator is not None and seed is not None:
        raise ValueError("pass either generator or seed, not both")
    effective_device = (
        device if not hasattr(circuit_or_ir, "device") else circuit_or_ir.device
    )
    if seed is not None:
        generator = trajectory_generator(
            seed,
            trajectory_id,
            device=effective_device,
        )

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
            device=effective_device,
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
    min_trajectories: int = 1,
    target_standard_error: float | None = None,
    generator: torch.Generator | None = None,
    seed: int | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    resume: bool = False,
    checkpoint_interval: int = 1,
    max_trajectories_per_run: int | None = None,
    retain_trajectories: bool = True,
    retry_failed: bool = True,
    continue_on_error: bool = False,
    rank: int = 0,
    world_size: int = 1,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    max_bond: int | None = None,
    cutoff: float = 0.0,
) -> MPSMonteCarloResult:
    """Run multiple noisy MPS trajectories and aggregate Z expectations."""

    from ..runtime.trajectories import (
        TensorWelford,
        TrajectoryCheckpoint,
        TrajectoryFailure,
        derive_trajectory_seed,
        load_trajectory_checkpoint,
        owned_trajectory_ids,
        save_trajectory_checkpoint,
        trajectory_generator,
    )

    if int(trajectories) <= 0:
        raise ValueError("trajectories must be positive")
    rank = int(rank)
    world_size = int(world_size)
    min_trajectories = int(min_trajectories)
    if min_trajectories <= 0 or min_trajectories > int(trajectories):
        raise ValueError("min_trajectories must be in [1, trajectories]")
    if target_standard_error is not None:
        target_standard_error = float(target_standard_error)
        if target_standard_error <= 0:
            raise ValueError("target_standard_error must be positive")
        if world_size != 1:
            raise ValueError(
                "adaptive trajectory stopping currently requires world_size=1; "
                "distributed convergence needs a collective statistics reduction"
            )
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if max_trajectories_per_run is not None and max_trajectories_per_run <= 0:
        raise ValueError("max_trajectories_per_run must be positive")
    owned_trajectory_ids(0, rank=rank, world_size=world_size)
    if world_size > int(trajectories):
        raise ValueError("world_size cannot exceed trajectory count")
    if generator is not None and seed is not None:
        raise ValueError("pass either generator or seed, not both")
    if checkpoint_path is not None and seed is None:
        raise ValueError("checkpointed trajectory execution requires seed")
    if resume and checkpoint_path is None:
        raise ValueError("resume requires checkpoint_path")
    if resume and retain_trajectories:
        raise ValueError(
            "resumed execution requires retain_trajectories=False because "
            "trajectory states are not stored in the statistics checkpoint"
        )
    if checkpoint_path is not None and world_size > 1:
        checkpoint_template = os.fspath(checkpoint_path)
        if "{rank}" not in checkpoint_template:
            raise ValueError(
                "distributed trajectory checkpoint_path must contain '{rank}'"
            )
        checkpoint_path = checkpoint_template.format(rank=rank)

    states = []
    retained_ids: list[int] = []
    failures: list[TrajectoryFailure] = []
    completed_ids: list[int] = []
    noise_model_identity = getattr(noise_model, "identity", None)
    if resume:
        checkpoint = load_trajectory_checkpoint(checkpoint_path)
        if checkpoint.requested_count != int(trajectories):
            raise ValueError("checkpoint requested trajectory count does not match")
        if checkpoint.base_seed != int(seed):
            raise ValueError("checkpoint base seed does not match")
        if checkpoint.noise_model_identity != noise_model_identity:
            raise ValueError("checkpoint noise model identity does not match")
        accumulator = checkpoint.accumulator()
        completed_ids.extend(checkpoint.completed_ids)
        failures.extend(checkpoint.failures)
        owned_ids = set(
            owned_trajectory_ids(
                int(trajectories),
                rank=rank,
                world_size=world_size,
            )
        )
        if any(item not in owned_ids for item in checkpoint.completed_ids):
            raise ValueError("checkpoint contains trajectories owned by another rank")
        pending_ids = checkpoint.pending_ids(
            rank=rank,
            world_size=world_size,
            retry_failed=retry_failed,
        )
    else:
        accumulator = TensorWelford()
        pending_ids = owned_trajectory_ids(
            int(trajectories),
            rank=rank,
            world_size=world_size,
        )
    if max_trajectories_per_run is not None:
        pending_ids = pending_ids[: int(max_trajectories_per_run)]
    effective_device = (
        device if not hasattr(circuit_or_ir, "device") else circuit_or_ir.device
    )
    completed_since_checkpoint = 0

    def target_reached() -> bool:
        if target_standard_error is None or accumulator.count < min_trajectories:
            return False
        standard_error = accumulator.finalize().standard_error
        return bool(torch.all(standard_error <= target_standard_error))

    def save_progress() -> None:
        if checkpoint_path is None:
            return
        save_trajectory_checkpoint(
            TrajectoryCheckpoint(
                requested_count=int(trajectories),
                base_seed=int(seed),
                completed_ids=tuple(sorted(completed_ids)),
                failures=tuple(sorted(failures, key=lambda item: item.trajectory_id)),
                statistics_state=(
                    accumulator.state_dict() if accumulator.count else None
                ),
                noise_model_identity=noise_model_identity,
            ),
            checkpoint_path,
        )

    for trajectory_id in () if target_reached() else pending_ids:
        trajectory_rng = (
            trajectory_generator(seed, trajectory_id, device=effective_device)
            if seed is not None
            else generator
        )
        try:
            state = run_noisy_mps_trajectory(
                circuit_or_ir,
                noise_model,
                generator=trajectory_rng,
                bsz=bsz,
                device=device,
                dtype=dtype,
                max_bond=max_bond,
                cutoff=cutoff,
            )
        except Exception as error:
            if not continue_on_error:
                save_progress()
                raise
            failures = [
                item for item in failures if item.trajectory_id != trajectory_id
            ]
            failures.append(
                TrajectoryFailure(
                    trajectory_id=trajectory_id,
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )
            save_progress()
            continue
        completed_ids.append(trajectory_id)
        failures = [item for item in failures if item.trajectory_id != trajectory_id]
        expectation_z = state.expectation_z()
        if noise_model is not None and hasattr(
            noise_model, "apply_readout_expectation_z"
        ):
            expectation_z = noise_model.apply_readout_expectation_z(expectation_z)
        accumulator.update(expectation_z)
        if retain_trajectories:
            states.append(state)
            retained_ids.append(trajectory_id)
        completed_since_checkpoint += 1
        if completed_since_checkpoint >= checkpoint_interval:
            save_progress()
            completed_since_checkpoint = 0
        if target_reached():
            break
    save_progress()
    if accumulator.count == 0:
        raise RuntimeError("no noisy MPS trajectories completed successfully")
    statistics = accumulator.finalize()
    converged = target_reached()
    trajectory_ids = tuple(sorted(completed_ids))
    trajectory_seeds = (
        tuple(derive_trajectory_seed(seed, item) for item in trajectory_ids)
        if seed is not None
        else ()
    )
    return MPSMonteCarloResult(
        trajectories=tuple(states),
        expectation_z_mean=statistics.mean,
        expectation_z_variance=statistics.variance,
        n_trajectories=statistics.count,
        trajectory_ids=trajectory_ids,
        trajectory_seeds=trajectory_seeds,
        statistics=statistics,
        requested_trajectories=int(trajectories),
        retained_trajectory_ids=tuple(retained_ids),
        failures=tuple(sorted(failures, key=lambda item: item.trajectory_id)),
        rank=rank,
        world_size=world_size,
        target_standard_error=target_standard_error,
        min_trajectories=min_trajectories,
        converged=converged,
        stopped_early=converged and statistics.count < int(trajectories),
        noise_model_identity=noise_model_identity,
    )


def merge_noisy_mps_results(
    results: Any,
    *,
    require_complete: bool = True,
) -> MPSMonteCarloResult:
    """Merge disjoint rank-local noisy MPS results in rank order."""

    from ..runtime.trajectories import merge_trajectory_statistics

    items = tuple(results)
    if not items:
        raise ValueError("at least one noisy MPS result is required")
    requested = {
        (
            item.n_trajectories
            if item.requested_trajectories is None
            else item.requested_trajectories
        )
        for item in items
    }
    world_sizes = {item.world_size for item in items}
    noise_model_identities = {item.noise_model_identity for item in items}
    ranks = [item.rank for item in items]
    if len(requested) != 1 or len(world_sizes) != 1 or len(noise_model_identities) != 1:
        raise ValueError("rank-local noisy MPS results have incompatible plans")
    world_size = world_sizes.pop()
    if len(ranks) != len(set(ranks)):
        raise ValueError("rank-local noisy MPS results contain duplicate ranks")
    if any(item.statistics is None for item in items):
        raise ValueError("rank-local noisy MPS result is missing statistics")

    trajectory_ids = tuple(
        sorted(item for result in items for item in result.trajectory_ids)
    )
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("rank-local noisy MPS results contain duplicate trajectories")
    requested_count = requested.pop()
    failures = tuple(
        sorted(
            (failure for result in items for failure in result.failures),
            key=lambda failure: failure.trajectory_id,
        )
    )
    if require_complete and (
        trajectory_ids != tuple(range(requested_count)) or failures
    ):
        raise ValueError("distributed noisy MPS result is incomplete")

    statistics = merge_trajectory_statistics(
        item.statistics for item in sorted(items, key=lambda result: result.rank)
    )
    retained = sorted(
        (trajectory_id, state)
        for result in items
        for trajectory_id, state in zip(
            result.retained_trajectory_ids,
            result.trajectories,
            strict=True,
        )
    )
    seed_by_id = {
        trajectory_id: seed
        for result in items
        for trajectory_id, seed in zip(
            result.trajectory_ids,
            result.trajectory_seeds,
            strict=True,
        )
    }
    return MPSMonteCarloResult(
        trajectories=tuple(state for _, state in retained),
        expectation_z_mean=statistics.mean,
        expectation_z_variance=statistics.variance,
        n_trajectories=statistics.count,
        trajectory_ids=trajectory_ids,
        trajectory_seeds=tuple(seed_by_id[item] for item in trajectory_ids),
        statistics=statistics,
        requested_trajectories=requested_count,
        retained_trajectory_ids=tuple(item for item, _ in retained),
        failures=failures,
        rank=0,
        world_size=world_size,
        noise_model_identity=noise_model_identities.pop(),
    )
