"""Adapt native MPS results for checked Runtime consumers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .result import ExecutionResult

if TYPE_CHECKING:
    import torch

    from ..simulation.mps.state import MPSState


def native_mps(result: ExecutionResult) -> MPSState | None:
    from ..simulation.mps.state import MPSState

    state = result.native()
    return state if isinstance(state, MPSState) else None


def mps_marginal(state: MPSState, qubits: tuple[int, ...]) -> torch.Tensor:
    from ..simulation.mps.marginal import marginal_probabilities

    return marginal_probabilities(state, qubits)
