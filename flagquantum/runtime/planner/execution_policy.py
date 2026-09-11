"""Pure Runtime execution-mode policy helpers."""

from __future__ import annotations

from .estimates import (
    estimate_density_bytes,
    estimate_mps_bytes,
    estimate_state_bytes,
    estimate_tensor_network_bytes,
)

_VALID_STATE_MODES = frozenset(
    {"statevector", "density_matrix", "mps", "tensor_network"}
)


def normalize_execution_state_mode(
    requested_mode: str,
    *,
    auto_selected_mode: str | None = None,
) -> str:
    """Normalize planner aliases while preserving explicit-mode validation."""

    state_mode = requested_mode
    if state_mode == "auto":
        if auto_selected_mode is None:
            raise ValueError("auto state_mode requires a selected execution mode")
        if auto_selected_mode in {"distributed", "distributed_statevector"}:
            state_mode = "statevector"
        elif auto_selected_mode in {
            "noisy_mps",
            "adaptive_mps",
            "distributed_mps",
        }:
            state_mode = "mps"
        elif auto_selected_mode == "distributed_tensor_network":
            state_mode = "tensor_network"
        else:
            state_mode = auto_selected_mode

    if state_mode in {"adaptive_mps", "distributed_mps"}:
        state_mode = "mps"
    elif state_mode in {"tn", "distributed_tensor_network"}:
        state_mode = "tensor_network"

    if state_mode not in _VALID_STATE_MODES:
        raise ValueError(
            "state_mode must be 'auto', 'statevector', 'density_matrix', "
            "'mps', or 'tensor_network'."
        )
    return state_mode


def estimate_execution_state_bytes(
    state_mode: str,
    *,
    n_wires: int,
    bsz: int,
    complex_bytes: int,
    max_bond: int | None,
) -> int:
    """Estimate storage for one normalized execution state."""

    if state_mode == "density_matrix":
        return estimate_density_bytes(
            n_wires,
            bsz=bsz,
            complex_bytes=complex_bytes,
        )
    if state_mode == "mps":
        return estimate_mps_bytes(
            n_wires,
            bsz=bsz,
            max_bond=max_bond,
            complex_bytes=complex_bytes,
        )
    if state_mode == "tensor_network":
        return estimate_tensor_network_bytes(
            n_wires,
            bsz=bsz,
            complex_bytes=complex_bytes,
        )
    return estimate_state_bytes(
        n_wires,
        bsz=bsz,
        complex_bytes=complex_bytes,
    )


def recommend_execution_mode(
    state_mode: str,
    *,
    world_size: int,
    has_noise: bool,
    state_bytes: int,
    memory_limit_bytes: int | None,
) -> str:
    """Choose the execution label for a normalized state representation."""

    recommended = "distributed_statevector" if world_size > 1 else "local"
    if state_mode == "density_matrix":
        recommended = "density"
    elif state_mode == "mps":
        recommended = (
            "noisy_mps" if has_noise else "distributed_mps" if world_size > 1 else "mps"
        )
    elif state_mode == "tensor_network":
        recommended = (
            "distributed_tensor_network" if world_size > 1 else "tensor_network"
        )
    if memory_limit_bytes is not None and state_bytes > memory_limit_bytes:
        return "distributed_statevector" if world_size > 1 else "mps"
    return recommended
