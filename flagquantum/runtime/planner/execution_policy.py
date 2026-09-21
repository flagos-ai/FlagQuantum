"""Pure Runtime execution-mode policy helpers."""

from __future__ import annotations

from .estimates import (
    estimate_density_bytes,
    estimate_mps_bytes,
    estimate_state_bytes,
    estimate_tensor_network_bytes,
    estimate_tensor_network_working_set_bytes,
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
    contraction_width: int | None = None,
    target_count: int = 1,
    require_gradients: bool = False,
) -> int:
    """Estimate storage for one normalized execution state.

    For ``tensor_network`` the footprint is the larger of the state the
    contraction materializes and the contraction's working-set proxy, because the
    contraction holds at least its result. ``contraction_width`` is the program's
    interaction width; omitting it assumes a fully connected program.

    The working set is a heuristic proxy, so this number is a lower bound on
    nothing and an upper bound on nothing: it is monotone in the inputs and never
    below the materialized state, which is what keeps it no more permissive than
    the dense estimate it replaced. It is deliberately *not* made smaller for
    sparse output targets — a contraction that returns one amplitude can still
    hold more intermediate amplitudes than it returns, so a sparse reduction here
    would admit runs that cannot fit. The executor's
    ``TensorNetworkContractionProfile.peak_size`` is the measured value.
    """

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
        working_set = estimate_tensor_network_working_set_bytes(
            n_wires,
            contraction_width=contraction_width,
            complex_bytes=complex_bytes,
            target_count=target_count,
            require_gradients=require_gradients,
        )
        return max(
            estimate_tensor_network_bytes(
                n_wires,
                bsz=bsz,
                complex_bytes=complex_bytes,
            ),
            working_set,
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
