"""Batched statevector quantum-trajectory execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from ....circuit import Circuit
from ....compiler import lower_noise_model
from ....noise import NoiseModel
from ....ops.gate_matrix import gate_matrix
from ....ops.matrices import GATE_MAT_DICT
from ...trajectories.ownership import owned_trajectory_ids
from ...trajectories.result import TrajectoryFailure, TrajectoryStatistics
from ...trajectories.rng import derive_trajectory_seed, trajectory_generator
from ...trajectories.statistics import TensorWelford


@dataclass(frozen=True)
class BatchedStatevectorTrajectoryResult:
    """Observable statistics from independently sampled statevector trajectories."""

    statistics: TrajectoryStatistics
    trajectory_ids: tuple[int, ...]
    trajectory_seeds: tuple[int, ...]
    trajectory_batch_size: int
    base_seed: int
    noise_model_identity: str
    pauli_fast_path_events: int
    amplitude_damping_fast_path_events: int
    generic_kraus_events: int
    retained_states: torch.Tensor | None = None
    requested_trajectories: int | None = None
    rank: int = 0
    world_size: int = 1
    min_trajectories: int = 1
    target_standard_error: float | None = None
    converged: bool = False
    stopped_early: bool = False
    failures: tuple[TrajectoryFailure, ...] = ()

    @property
    def expectation_z(self) -> torch.Tensor:
        return self.statistics.mean

    @property
    def variance(self) -> torch.Tensor:
        return self.statistics.variance

    @property
    def standard_error(self) -> torch.Tensor:
        return self.statistics.standard_error

    def summary(self) -> dict[str, object]:
        return {
            "backend": "batched_statevector_trajectory",
            "trajectories": self.statistics.count,
            "trajectory_batch_size": self.trajectory_batch_size,
            "base_seed": self.base_seed,
            "noise_model_identity": self.noise_model_identity,
            "pauli_fast_path_events": self.pauli_fast_path_events,
            "amplitude_damping_fast_path_events": (
                self.amplitude_damping_fast_path_events
            ),
            "generic_kraus_events": self.generic_kraus_events,
            "retained_trajectories": self.retained_states is not None,
            "rank": self.rank,
            "world_size": self.world_size,
            "min_trajectories": self.min_trajectories,
            "target_standard_error": self.target_standard_error,
            "converged": self.converged,
            "stopped_early": self.stopped_early,
            "failure_count": len(self.failures),
        }


def _apply_matrix_batched(
    state: torch.Tensor,
    matrix: torch.Tensor,
    wires: tuple[int, ...],
    n_wires: int,
) -> torch.Tensor:
    trajectories, circuit_batch, _ = state.shape
    width = len(wires)
    dimension = 2**width
    other_wires = tuple(wire for wire in range(n_wires) if wire not in wires)
    permutation = (0, 1) + tuple(wire + 2 for wire in wires + other_wires)
    inverse = [0] * (n_wires + 2)
    for destination, source in enumerate(permutation):
        inverse[source] = destination
    flat = (
        state.reshape(trajectories, circuit_batch, *((2,) * n_wires))
        .permute(permutation)
        .reshape(trajectories * circuit_batch, dimension, -1)
    )
    matrix = torch.as_tensor(matrix, device=state.device, dtype=state.dtype)
    if matrix.ndim == 2:
        applied = torch.matmul(matrix, flat)
    elif matrix.ndim == 3 and matrix.shape[0] == circuit_batch:
        expanded = matrix.unsqueeze(0).expand(trajectories, -1, -1, -1)
        applied = torch.bmm(expanded.reshape(-1, dimension, dimension), flat)
    else:
        raise ValueError(
            "gate matrix must be unbatched or match the circuit batch dimension"
        )
    return (
        applied.reshape(trajectories, circuit_batch, *((2,) * n_wires))
        .permute(tuple(inverse))
        .reshape(trajectories, circuit_batch, -1)
    )


def _sample_rows(
    probabilities: torch.Tensor,
    generators: list[torch.Generator],
) -> torch.Tensor:
    choices = [
        torch.multinomial(row, 1, replacement=True, generator=generator).squeeze(-1)
        for row, generator in zip(probabilities, generators, strict=True)
    ]
    return torch.stack(choices)


def _apply_kraus_batched(
    state: torch.Tensor,
    operators: tuple[torch.Tensor, ...],
    wires: tuple[int, ...],
    n_wires: int,
    generators: list[torch.Generator],
) -> torch.Tensor:
    branches = torch.stack(
        [
            _apply_matrix_batched(state, operator, wires, n_wires)
            for operator in operators
        ],
        dim=2,
    )
    probabilities = torch.sum(torch.abs(branches) ** 2, dim=-1).real
    probabilities = torch.clamp(probabilities, min=0)
    totals = probabilities.sum(dim=-1, keepdim=True)
    if bool(torch.any(totals <= 0)):
        raise RuntimeError("Kraus channel produced zero total branch probability")
    choices = _sample_rows(probabilities / totals, generators)
    selected = torch.gather(
        branches,
        2,
        choices[..., None, None].expand(-1, -1, 1, branches.shape[-1]),
    ).squeeze(2)
    selected_probability = torch.gather(probabilities, 2, choices[..., None]).squeeze(
        -1
    )
    return (
        selected / torch.sqrt(torch.clamp(selected_probability, min=1e-30))[..., None]
    )


def _apply_amplitude_damping_batched(
    state: torch.Tensor,
    operators: tuple[torch.Tensor, ...],
    wire: int,
    n_wires: int,
    generators: list[torch.Generator],
) -> torch.Tensor:
    """Sample and apply amplitude damping without materializing branch states."""

    gamma = torch.abs(operators[1][0, 1]) ** 2
    trajectories, circuit_batch, _ = state.shape
    other_wires = tuple(index for index in range(n_wires) if index != wire)
    permutation = (0, 1, wire + 2) + tuple(index + 2 for index in other_wires)
    inverse = [0] * (n_wires + 2)
    for destination, source in enumerate(permutation):
        inverse[source] = destination
    packed = (
        state.reshape(trajectories, circuit_batch, *((2,) * n_wires))
        .permute(permutation)
        .reshape(trajectories, circuit_batch, 2, -1)
    )
    zero, one = packed[:, :, 0], packed[:, :, 1]
    jump_probability = torch.clamp(
        gamma.real * torch.sum(torch.abs(one) ** 2, dim=-1), min=0, max=1
    )
    probabilities = torch.stack((1 - jump_probability, jump_probability), dim=-1)
    choices = _sample_rows(probabilities, generators)
    jump = choices.bool()[..., None]
    out_zero = torch.where(jump, torch.sqrt(gamma) * one, zero)
    out_one = torch.where(jump, torch.zeros_like(one), torch.sqrt(1 - gamma) * one)
    selected_probability = torch.where(
        jump[..., 0], jump_probability, 1 - jump_probability
    )
    packed_out = (
        torch.stack((out_zero, out_one), dim=2)
        / torch.sqrt(torch.clamp(selected_probability, min=1e-30))[..., None, None]
    )
    return (
        packed_out.reshape(trajectories, circuit_batch, *((2,) * n_wires))
        .permute(tuple(inverse))
        .reshape_as(state)
    )


def _expectation_z(state: torch.Tensor, n_wires: int) -> torch.Tensor:
    indices = torch.arange(state.shape[-1], device=state.device)
    shifts = torch.arange(n_wires - 1, -1, -1, device=state.device)
    signs = 1 - 2 * ((indices[:, None] >> shifts) & 1)
    return torch.matmul(torch.abs(state) ** 2, signs.to(dtype=state.real.dtype))


def _collective_statistics(statistics: TrajectoryStatistics) -> TrajectoryStatistics:
    count = torch.tensor(
        statistics.count, dtype=torch.float64, device=statistics.mean.device
    )
    mean = statistics.mean.to(dtype=torch.float64)
    variance = statistics.variance.to(dtype=torch.float64)
    total = mean * count
    total_square = (variance + mean * mean) * count
    torch.distributed.all_reduce(count)
    torch.distributed.all_reduce(total)
    torch.distributed.all_reduce(total_square)
    if int(count.item()) == 0:
        raise RuntimeError("no statevector trajectories completed successfully")
    global_mean = total / count
    global_variance = torch.clamp(
        total_square / count - global_mean * global_mean, min=0
    )
    return TrajectoryStatistics(
        count=int(count.item()),
        mean=global_mean.to(dtype=statistics.mean.dtype),
        variance=global_variance.to(dtype=statistics.variance.dtype),
        standard_error=torch.sqrt(global_variance / count).to(
            dtype=statistics.standard_error.dtype
        ),
    )


def _finalize_or_empty_statistics(
    accumulator: TensorWelford,
    *,
    circuit_batch: int,
    n_wires: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> TrajectoryStatistics:
    if accumulator.count:
        return accumulator.finalize()
    zeros = torch.zeros(circuit_batch, n_wires, device=device, dtype=dtype)
    return TrajectoryStatistics(
        count=0,
        mean=zeros,
        variance=zeros.clone(),
        standard_error=torch.full_like(zeros, torch.inf),
    )


def _execute_trajectory_batch(
    initial: torch.Tensor,
    ids: tuple[int, ...],
    *,
    seed: int,
    device: torch.device | str,
    ir: Any,
    circuit_batch: int,
    noise_model: NoiseModel,
    pauli_matrices: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    generators = [
        trajectory_generator(seed, trajectory_id, device=device)
        for trajectory_id in ids
    ]
    state = initial.unsqueeze(0).expand(len(ids), -1, -1).clone()
    pauli_events = 0
    amplitude_events = 0
    generic_events = 0
    for instruction in ir.instructions:
        if not instruction.metadata.get("is_channel"):
            matrix = gate_matrix(
                instruction,
                bsz=circuit_batch,
                device=device,
                dtype=initial.dtype,
            )
            state = _apply_matrix_batched(state, matrix, instruction.wires, ir.n_wires)
            continue
        operators = tuple(
            torch.as_tensor(operator, device=device, dtype=initial.dtype)
            for operator in instruction.matrix
        )
        if instruction.name in {"bit_flip", "phase_flip", "depolarizing"}:
            probabilities = torch.tensor(
                [
                    float((torch.real(torch.trace(op.mH @ op)) / 2).item())
                    for op in operators
                ],
                device=state.device,
                dtype=state.real.dtype,
            ).expand(state.shape[0], state.shape[1], -1)
            choices = _sample_rows(probabilities, generators)
            names = (
                ("i", "x")
                if instruction.name == "bit_flip"
                else (
                    ("i", "z")
                    if instruction.name == "phase_flip"
                    else ("i", "x", "y", "z")
                )
            )
            branches = torch.stack(
                [
                    _apply_matrix_batched(
                        state, pauli_matrices[name], instruction.wires, ir.n_wires
                    )
                    for name in names
                ],
                dim=2,
            )
            state = torch.gather(
                branches,
                2,
                choices[..., None, None].expand(-1, -1, 1, state.shape[-1]),
            ).squeeze(2)
            pauli_events += state.shape[0] * circuit_batch
        elif instruction.name == "amplitude_damping" and len(instruction.wires) == 1:
            state = _apply_amplitude_damping_batched(
                state, operators, instruction.wires[0], ir.n_wires, generators
            )
            amplitude_events += state.shape[0] * circuit_batch
        else:
            state = _apply_kraus_batched(
                state, operators, instruction.wires, ir.n_wires, generators
            )
            generic_events += state.shape[0] * circuit_batch
    expectation = noise_model.apply_readout_expectation_z(
        _expectation_z(state, ir.n_wires)
    )
    return state, expectation, pauli_events, amplitude_events, generic_events


def run_noisy_statevector(
    circuit_or_ir: Any,
    noise_model: NoiseModel,
    *,
    trajectories: int,
    trajectory_batch_size: int = 32,
    seed: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    retain_trajectories: bool = False,
    rank: int | None = None,
    world_size: int | None = None,
    collective: bool | None = None,
    min_trajectories: int = 1,
    target_standard_error: float | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    resume: bool = False,
    checkpoint_interval: int = 1,
    max_batches_per_run: int | None = None,
    continue_on_error: bool = False,
    retry_failed: bool = True,
) -> BatchedStatevectorTrajectoryResult:
    """Run Kraus trajectories in bounded statevector batches.

    Random streams are derived from global trajectory IDs, so changing the
    trajectory batch size does not change a seeded result.
    """

    if not isinstance(noise_model, NoiseModel):
        raise TypeError("noise_model must be a NoiseModel")
    if trajectories <= 0:
        raise ValueError("trajectories must be positive")
    if trajectory_batch_size <= 0:
        raise ValueError("trajectory_batch_size must be positive")
    if min_trajectories <= 0 or min_trajectories > trajectories:
        raise ValueError("min_trajectories must be in [1, trajectories]")
    if target_standard_error is not None and target_standard_error <= 0:
        raise ValueError("target_standard_error must be positive")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if max_batches_per_run is not None and max_batches_per_run <= 0:
        raise ValueError("max_batches_per_run must be positive")
    if resume and checkpoint_path is None:
        raise ValueError("resume requires checkpoint_path")
    if resume and retain_trajectories:
        raise ValueError(
            "resumed execution cannot retain states because checkpoints store moments"
        )
    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    distributed_rank = torch.distributed.get_rank() if distributed else 0
    distributed_world_size = torch.distributed.get_world_size() if distributed else 1
    selected_rank = distributed_rank if rank is None else int(rank)
    selected_world_size = (
        distributed_world_size if world_size is None else int(world_size)
    )
    collective = (
        distributed and selected_world_size > 1 if collective is None else collective
    )
    if collective and not distributed:
        raise RuntimeError(
            "collective execution requires initialized torch.distributed"
        )
    if collective and (
        selected_rank != distributed_rank
        or selected_world_size != distributed_world_size
    ):
        raise ValueError("rank/world_size must match the initialized process group")
    if collective and retain_trajectories:
        raise NotImplementedError(
            "collective statevector trajectories do not gather retained states"
        )
    if checkpoint_path is not None and selected_world_size > 1:
        checkpoint_template = os.fspath(checkpoint_path)
        if "{rank}" not in checkpoint_template:
            raise ValueError(
                "distributed checkpoint_path must contain the '{rank}' placeholder"
            )
        checkpoint_path = checkpoint_template.format(rank=selected_rank)
    if (
        collective
        and target_standard_error is not None
        and trajectories % (selected_world_size * trajectory_batch_size) != 0
    ):
        raise ValueError(
            "collective adaptive stopping currently requires trajectories to be "
            "divisible by world_size * trajectory_batch_size"
        )
    owned_ids = owned_trajectory_ids(
        trajectories, rank=selected_rank, world_size=selected_world_size
    )
    if not owned_ids:
        raise ValueError("each rank must own at least one trajectory")
    ir = lower_noise_model(circuit_or_ir, noise_model)
    if isinstance(circuit_or_ir, Circuit):
        initial = circuit_or_ir.initial_state()
        selected_device = device or initial.device
        selected_dtype = dtype or initial.dtype
        initial = initial.to(device=selected_device, dtype=selected_dtype)
    else:
        selected_device = device or "cpu"
        selected_dtype = dtype or getattr(torch, ir.dtype)
        circuit_batch = int(ir.metadata.get("batch_size", 1))
        initial = torch.zeros(
            circuit_batch,
            2**ir.n_wires,
            device=selected_device,
            dtype=selected_dtype,
        )
        initial[:, 0] = 1
    circuit_batch = initial.shape[0]
    from ...trajectories import (
        TrajectoryCheckpoint,
        load_trajectory_checkpoint,
        save_trajectory_checkpoint,
    )

    execution_metadata = {
        "backend": "batched_statevector",
        "rank": selected_rank,
        "world_size": selected_world_size,
        "trajectory_batch_size": trajectory_batch_size,
    }
    completed_ids: list[int] = []
    if resume:
        checkpoint = load_trajectory_checkpoint(checkpoint_path)
        if checkpoint.requested_count != trajectories:
            raise ValueError("checkpoint trajectory count does not match")
        if checkpoint.base_seed != seed:
            raise ValueError("checkpoint base seed does not match")
        if checkpoint.noise_model_identity != noise_model.identity:
            raise ValueError("checkpoint noise model identity does not match")
        if checkpoint.circuit_digest != ir.content_hash:
            raise ValueError("checkpoint circuit digest does not match")
        if checkpoint.execution_metadata != execution_metadata:
            raise ValueError("checkpoint execution metadata does not match")
        if any(item not in set(owned_ids) for item in checkpoint.completed_ids):
            raise ValueError("checkpoint contains trajectory owned by another rank")
        restored_state = checkpoint.accumulator().state_dict()
        if restored_state["count"]:
            accumulator = TensorWelford.from_state_dict(
                {
                    "count": restored_state["count"],
                    "mean": restored_state["mean"].to(
                        device=selected_device, dtype=initial.real.dtype
                    ),
                    "m2": restored_state["m2"].to(
                        device=selected_device, dtype=initial.real.dtype
                    ),
                }
            )
        else:
            accumulator = TensorWelford()
        completed_ids.extend(checkpoint.completed_ids)
        failures: list[TrajectoryFailure] = list(checkpoint.failures)
        trajectory_ids = checkpoint.pending_ids(
            rank=selected_rank,
            world_size=selected_world_size,
            retry_failed=retry_failed,
        )
    else:
        accumulator = TensorWelford()
        trajectory_ids = owned_ids
        failures = []
    retained: list[torch.Tensor] = []
    pauli_events = 0
    amplitude_events = 0
    generic_events = 0
    converged = False
    batches_since_checkpoint = 0
    batches_executed = 0

    def save_progress() -> None:
        if checkpoint_path is None:
            return
        save_trajectory_checkpoint(
            TrajectoryCheckpoint(
                requested_count=trajectories,
                base_seed=seed,
                completed_ids=tuple(sorted(completed_ids)),
                failures=tuple(sorted(failures, key=lambda item: item.trajectory_id)),
                statistics_state=accumulator.state_dict(),
                noise_model_identity=noise_model.identity,
                circuit_digest=ir.content_hash,
                execution_metadata=execution_metadata,
            ),
            checkpoint_path,
        )

    pauli_matrices = {
        name: torch.as_tensor(matrix, device=selected_device, dtype=selected_dtype)
        for name, matrix in GATE_MAT_DICT.items()
        if name in {"i", "x", "y", "z"}
    }
    for start in range(0, len(trajectory_ids), trajectory_batch_size):
        ids = trajectory_ids[start : start + trajectory_batch_size]
        successful_states: list[torch.Tensor] = []
        successful_expectations: list[torch.Tensor] = []
        successful_ids: list[int] = []
        try:
            state, expectation, batch_pauli, batch_amplitude, batch_generic = (
                _execute_trajectory_batch(
                    initial,
                    ids,
                    seed=seed,
                    device=selected_device,
                    ir=ir,
                    circuit_batch=circuit_batch,
                    noise_model=noise_model,
                    pauli_matrices=pauli_matrices,
                )
            )
            successful_states.extend(state)
            successful_expectations.extend(expectation)
            successful_ids.extend(ids)
            pauli_events += batch_pauli
            amplitude_events += batch_amplitude
            generic_events += batch_generic
        except Exception:
            if not continue_on_error:
                save_progress()
                raise
            for trajectory_id in ids:
                try:
                    state, expectation, item_pauli, item_amplitude, item_generic = (
                        _execute_trajectory_batch(
                            initial,
                            (trajectory_id,),
                            seed=seed,
                            device=selected_device,
                            ir=ir,
                            circuit_batch=circuit_batch,
                            noise_model=noise_model,
                            pauli_matrices=pauli_matrices,
                        )
                    )
                except Exception as error:
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
                    continue
                successful_states.extend(state)
                successful_expectations.extend(expectation)
                successful_ids.append(trajectory_id)
                pauli_events += item_pauli
                amplitude_events += item_amplitude
                generic_events += item_generic
        for value in successful_expectations:
            accumulator.update(value)
        completed_ids.extend(successful_ids)
        failures = [
            item for item in failures if item.trajectory_id not in successful_ids
        ]
        batches_executed += 1
        batches_since_checkpoint += 1
        if batches_since_checkpoint >= checkpoint_interval:
            save_progress()
            batches_since_checkpoint = 0
        if retain_trajectories and successful_states:
            retained.append(torch.stack(successful_states).detach())
        if target_standard_error is not None:
            current_statistics = _finalize_or_empty_statistics(
                accumulator,
                circuit_batch=circuit_batch,
                n_wires=ir.n_wires,
                device=selected_device,
                dtype=initial.real.dtype,
            )
            if collective:
                current_statistics = _collective_statistics(current_statistics)
            converged = current_statistics.count >= min_trajectories and bool(
                torch.all(
                    current_statistics.standard_error <= target_standard_error
                ).item()
            )
            if converged:
                break
        if max_batches_per_run is not None and batches_executed >= max_batches_per_run:
            break
    save_progress()
    statistics = _finalize_or_empty_statistics(
        accumulator,
        circuit_batch=circuit_batch,
        n_wires=ir.n_wires,
        device=selected_device,
        dtype=initial.real.dtype,
    )
    if not collective and statistics.count == 0:
        raise RuntimeError("no statevector trajectories completed successfully")
    ordered_completed_ids = tuple(sorted(completed_ids))
    result = BatchedStatevectorTrajectoryResult(
        statistics=statistics,
        trajectory_ids=ordered_completed_ids,
        trajectory_seeds=tuple(
            derive_trajectory_seed(seed, trajectory_id)
            for trajectory_id in ordered_completed_ids
        ),
        trajectory_batch_size=trajectory_batch_size,
        base_seed=seed,
        noise_model_identity=noise_model.identity,
        pauli_fast_path_events=pauli_events,
        amplitude_damping_fast_path_events=amplitude_events,
        generic_kraus_events=generic_events,
        retained_states=torch.cat(retained) if retained else None,
        requested_trajectories=trajectories,
        rank=selected_rank,
        world_size=selected_world_size,
        min_trajectories=min_trajectories,
        target_standard_error=target_standard_error,
        converged=converged,
        stopped_early=converged and len(completed_ids) < len(owned_ids),
        failures=tuple(sorted(failures, key=lambda item: item.trajectory_id)),
    )
    return _collective_reduce_result(result) if collective else result


def _collective_reduce_result(
    result: BatchedStatevectorTrajectoryResult,
) -> BatchedStatevectorTrajectoryResult:
    """Reduce rank-local population moments through the active process group."""

    statistics = result.statistics
    global_statistics = _collective_statistics(statistics)
    events = torch.tensor(
        [
            result.pauli_fast_path_events,
            result.amplitude_damping_fast_path_events,
            result.generic_kraus_events,
        ],
        dtype=torch.int64,
        device=statistics.mean.device,
    )
    torch.distributed.all_reduce(events)
    requested = result.requested_trajectories or global_statistics.count
    completed = global_statistics.count
    gathered: list[Any] = [None] * result.world_size
    torch.distributed.all_gather_object(
        gathered, (result.trajectory_ids, result.failures)
    )
    completed_ids = tuple(
        sorted(trajectory_id for ids, _ in gathered for trajectory_id in ids)
    )
    failures = tuple(
        sorted(
            (failure for _, items in gathered for failure in items),
            key=lambda item: item.trajectory_id,
        )
    )
    if len(completed_ids) != completed:
        raise RuntimeError("collective statistics count does not match completed IDs")
    return BatchedStatevectorTrajectoryResult(
        statistics=global_statistics,
        trajectory_ids=completed_ids,
        trajectory_seeds=tuple(
            derive_trajectory_seed(result.base_seed, trajectory_id)
            for trajectory_id in completed_ids
        ),
        trajectory_batch_size=result.trajectory_batch_size,
        base_seed=result.base_seed,
        noise_model_identity=result.noise_model_identity,
        pauli_fast_path_events=int(events[0].item()),
        amplitude_damping_fast_path_events=int(events[1].item()),
        generic_kraus_events=int(events[2].item()),
        retained_states=None,
        requested_trajectories=requested,
        rank=result.rank,
        world_size=result.world_size,
        min_trajectories=result.min_trajectories,
        target_standard_error=result.target_standard_error,
        converged=result.converged,
        stopped_early=result.converged and completed < requested,
        failures=failures,
    )


def merge_noisy_statevector_results(
    results: Any,
    *,
    require_complete: bool = True,
) -> BatchedStatevectorTrajectoryResult:
    """Merge disjoint rank-local statevector trajectory results."""

    from ...trajectories import merge_trajectory_statistics

    items = tuple(results)
    if not items:
        raise ValueError("at least one statevector trajectory result is required")
    requested = {
        (
            item.statistics.count
            if item.requested_trajectories is None
            else item.requested_trajectories
        )
        for item in items
    }
    world_sizes = {item.world_size for item in items}
    identities = {item.noise_model_identity for item in items}
    base_seeds = {item.base_seed for item in items}
    ranks = [item.rank for item in items]
    if (
        len(requested) != 1
        or len(world_sizes) != 1
        or len(identities) != 1
        or len(base_seeds) != 1
    ):
        raise ValueError("rank-local statevector results have incompatible plans")
    if len(ranks) != len(set(ranks)):
        raise ValueError("rank-local statevector results contain duplicate ranks")
    trajectory_ids = tuple(sorted(i for item in items for i in item.trajectory_ids))
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError(
            "rank-local statevector results contain duplicate trajectories"
        )
    requested_count = requested.pop()
    if require_complete and trajectory_ids != tuple(range(requested_count)):
        raise ValueError("distributed statevector trajectory result is incomplete")
    failures = tuple(
        sorted(
            (failure for item in items for failure in item.failures),
            key=lambda failure: failure.trajectory_id,
        )
    )
    seed_by_id = {
        trajectory_id: trajectory_seed
        for item in items
        for trajectory_id, trajectory_seed in zip(
            item.trajectory_ids, item.trajectory_seeds, strict=True
        )
    }
    retained_by_id = {
        trajectory_id: state
        for item in items
        if item.retained_states is not None
        for trajectory_id, state in zip(
            item.trajectory_ids, item.retained_states, strict=True
        )
    }
    retained_states = (
        torch.stack([retained_by_id[i] for i in trajectory_ids])
        if len(retained_by_id) == len(trajectory_ids)
        else None
    )
    statistics = merge_trajectory_statistics(
        item.statistics for item in sorted(items, key=lambda value: value.rank)
    )
    return BatchedStatevectorTrajectoryResult(
        statistics=statistics,
        trajectory_ids=trajectory_ids,
        trajectory_seeds=tuple(seed_by_id[i] for i in trajectory_ids),
        trajectory_batch_size=max(item.trajectory_batch_size for item in items),
        base_seed=base_seeds.pop(),
        noise_model_identity=identities.pop(),
        pauli_fast_path_events=sum(item.pauli_fast_path_events for item in items),
        amplitude_damping_fast_path_events=sum(
            item.amplitude_damping_fast_path_events for item in items
        ),
        generic_kraus_events=sum(item.generic_kraus_events for item in items),
        retained_states=retained_states,
        requested_trajectories=requested_count,
        rank=0,
        world_size=world_sizes.pop(),
        failures=failures,
    )


__all__ = (
    "BatchedStatevectorTrajectoryResult",
    "merge_noisy_statevector_results",
    "run_noisy_statevector",
)
