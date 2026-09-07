"""Native matrix-product-state execution for FlagQuantum."""

from __future__ import annotations

import os
from typing import Any

import torch

from ..core.ir import CircuitIR, ensure_circuit_ir
from .mps.local import run_local_mps
from .mps.models import (
    MPSAdaptiveRunResult,
    MPSConfig,
    MPSMonteCarloResult,
)
from .mps.state import MPSState
from .mps_noisy import run_local_noisy_mps_trajectory


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

    binding_source = getattr(circuit_or_ir, "_parameter_bindings", None)
    parameter_bindings = None if binding_source is None else binding_source.values()
    program_cache = getattr(circuit_or_ir, "_backend_programs", None)
    spatial_bucket = os.getenv("FQ_MPS_SPATIAL_BUCKET", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }
    return run_local_mps(
        ir,
        mps,
        parameter_bindings=parameter_bindings,
        program_cache=program_cache,
        fuse_single_qubit=fuse_single_qubit,
        spatial_bucket=spatial_bucket,
    )


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
    """Prepare and execute one sampled MPS quantum trajectory."""

    from ..compiler import lower_noise_model

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
    lowered: CircuitIR,
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
) -> MPSState:
    """Execute a Compiler-lowered trajectory through Runtime and Simulation."""

    from ..runtime.trajectories.mps import run_single_mps_trajectory_runtime

    effective_device = device if not hasattr(source, "device") else source.device
    return run_single_mps_trajectory_runtime(
        lowered,
        generator=generator,
        seed=seed,
        trajectory_id=trajectory_id,
        device=effective_device,
        trajectory_executor=_execute_lowered_noisy_mps_trajectory,
        executor_options={
            "source": source,
            "bsz": bsz,
            "device": effective_device,
            "dtype": dtype,
            "max_bond": max_bond,
            "cutoff": cutoff,
        },
    )


def _execute_lowered_noisy_mps_trajectory(
    lowered: CircuitIR,
    *,
    source: Any,
    generator: torch.Generator | None,
    bsz: int,
    device: torch.device | str,
    dtype: torch.dtype | None,
    max_bond: int | None,
    cutoff: float,
) -> MPSState:
    """Initialize MPS storage and enter the lowered numerical loop."""

    config = MPSConfig(max_bond=max_bond, cutoff=cutoff)
    if (
        hasattr(source, "initial_state")
        and getattr(source, "_inputs", None) is not None
    ):
        mps = MPSState.from_statevector(
            source.initial_state(),
            lowered.n_wires,
            config=config,
        )
    else:
        source_dtype = getattr(source, "dtype", None)
        mps = MPSState.zero(
            lowered.n_wires,
            bsz=bsz if not hasattr(source, "bsz") else source.bsz,
            device=device,
            dtype=source_dtype if isinstance(source_dtype, torch.dtype) else dtype,
            config=config,
        )
    return run_local_noisy_mps_trajectory(
        lowered,
        mps,
        generator=generator,
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
    """Schedule noisy MPS trajectories through the Runtime lifecycle."""

    from ..compiler import lower_noise_model

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
    lowered: CircuitIR,
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
) -> MPSMonteCarloResult:
    """Schedule trajectories for one already-lowered noisy MPS program."""

    from ..runtime.trajectories.mps import run_noisy_mps_runtime

    def execute_lowered_trajectory(
        _source: Any,
        _noise_model: Any,
        *,
        generator: torch.Generator | None,
        bsz: int,
        device: torch.device | str,
        dtype: torch.dtype | None,
        max_bond: int | None,
        cutoff: float,
    ) -> MPSState:
        effective_device = device if not hasattr(source, "device") else source.device
        return _execute_lowered_noisy_mps_trajectory(
            lowered,
            source=source,
            generator=generator,
            bsz=bsz,
            device=effective_device,
            dtype=dtype,
            max_bond=max_bond,
            cutoff=cutoff,
        )

    return run_noisy_mps_runtime(
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
        trajectory_executor=execute_lowered_trajectory,
        result_factory=MPSMonteCarloResult,
    )


def merge_noisy_mps_results(
    results: Any,
    *,
    require_complete: bool = True,
) -> MPSMonteCarloResult:
    """Merge disjoint rank-local noisy MPS results through Runtime."""

    from ..runtime.trajectories.mps import merge_noisy_mps_results_runtime

    return merge_noisy_mps_results_runtime(
        results,
        require_complete=require_complete,
        result_factory=MPSMonteCarloResult,
    )
