"""Runtime composition for noisy MPS trajectory execution."""

from __future__ import annotations

import os
from typing import Any

import torch

from ...trajectories import mps as trajectory_runtime
from ...trajectories.result import MPSMonteCarloResult


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
) -> Any:
    """Compile and execute one sampled MPS trajectory."""

    from ....compiler import lower_noise_model

    lowered = lower_noise_model(circuit_or_ir, noise_model)
    return run_lowered_noisy_mps_trajectory(
        lowered,
        source=circuit_or_ir,
        generator=generator,
        seed=seed,
        trajectory_id=trajectory_id,
        bsz=bsz,
        device=device,
        dtype=dtype,
        max_bond=max_bond,
        cutoff=cutoff,
    )


def run_lowered_noisy_mps_trajectory(
    lowered: Any,
    *,
    source: Any,
    generator: torch.Generator | None = None,
    seed: int | None = None,
    trajectory_id: int = 0,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    max_bond: int | None = None,
    cutoff: float = 0.0,
) -> Any:
    """Apply Runtime trajectory policy to one already-lowered program."""

    from ....simulation.mps.entrypoints import execute_lowered_noisy_mps_trajectory

    effective_device = device if not hasattr(source, "device") else source.device
    return trajectory_runtime.run_single_mps_trajectory_runtime(
        lowered,
        generator=generator,
        seed=seed,
        trajectory_id=trajectory_id,
        device=effective_device,
        trajectory_executor=execute_lowered_noisy_mps_trajectory,
        executor_options={
            "source": source,
            "bsz": bsz,
            "device": effective_device,
            "dtype": dtype,
            "max_bond": max_bond,
            "cutoff": cutoff,
        },
    )


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
    """Compile and schedule noisy MPS trajectories."""

    from ....compiler import lower_noise_model

    lowered = lower_noise_model(circuit_or_ir, noise_model)
    return run_lowered_noisy_mps(
        lowered,
        source=circuit_or_ir,
        noise_model=noise_model,
        trajectories=trajectories,
        min_trajectories=min_trajectories,
        target_standard_error=target_standard_error,
        generator=generator,
        seed=seed,
        checkpoint_path=checkpoint_path,
        resume=resume,
        checkpoint_interval=checkpoint_interval,
        max_trajectories_per_run=max_trajectories_per_run,
        retain_trajectories=retain_trajectories,
        retry_failed=retry_failed,
        continue_on_error=continue_on_error,
        rank=rank,
        world_size=world_size,
        bsz=bsz,
        device=device,
        dtype=dtype,
        max_bond=max_bond,
        cutoff=cutoff,
    )


def run_lowered_noisy_mps(
    lowered: Any,
    *,
    source: Any,
    noise_model: Any | None,
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
) -> Any:
    """Schedule trajectories for an already-lowered noisy MPS program."""

    from ....simulation.mps.entrypoints import execute_lowered_noisy_mps_trajectory

    def execute_trajectory(
        _source: Any,
        _noise_model: Any,
        *,
        generator: torch.Generator | None,
        bsz: int,
        device: torch.device | str,
        dtype: torch.dtype | None,
        max_bond: int | None,
        cutoff: float,
    ) -> Any:
        effective_device = device if not hasattr(source, "device") else source.device
        return execute_lowered_noisy_mps_trajectory(
            lowered,
            source=source,
            generator=generator,
            bsz=bsz,
            device=effective_device,
            dtype=dtype,
            max_bond=max_bond,
            cutoff=cutoff,
        )

    return trajectory_runtime.run_noisy_mps_runtime(
        source,
        noise_model,
        trajectories=trajectories,
        min_trajectories=min_trajectories,
        target_standard_error=target_standard_error,
        generator=generator,
        seed=seed,
        checkpoint_path=checkpoint_path,
        resume=resume,
        checkpoint_interval=checkpoint_interval,
        max_trajectories_per_run=max_trajectories_per_run,
        retain_trajectories=retain_trajectories,
        retry_failed=retry_failed,
        continue_on_error=continue_on_error,
        rank=rank,
        world_size=world_size,
        bsz=bsz,
        device=device,
        dtype=dtype,
        max_bond=max_bond,
        cutoff=cutoff,
        trajectory_executor=execute_trajectory,
    )


def merge_noisy_mps_results(
    results: Any,
    *,
    require_complete: bool = True,
) -> MPSMonteCarloResult:
    """Merge disjoint rank-local noisy MPS results."""

    return trajectory_runtime.merge_noisy_mps_results_runtime(
        results,
        require_complete=require_complete,
    )


__all__ = (
    "merge_noisy_mps_results",
    "run_lowered_noisy_mps",
    "run_lowered_noisy_mps_trajectory",
    "run_noisy_mps",
    "run_noisy_mps_trajectory",
)
