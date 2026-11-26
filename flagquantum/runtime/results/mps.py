"""Adapt native MPS results for checked Runtime consumers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flagquantum._import_compat import alias_module as _alias_module

from . import ExecutionResult

if TYPE_CHECKING:
    import torch

    from flagquantum.simulation.mps.state import MPSState


def native_mps(result: ExecutionResult) -> MPSState | None:
    from flagquantum.simulation.mps.state import MPSState

    state = result.native()
    return state if isinstance(state, MPSState) else None


def mps_marginal(state: MPSState, qubits: tuple[int, ...]) -> torch.Tensor:
    from flagquantum.simulation.mps.marginal import marginal_probabilities

    return marginal_probabilities(state, qubits)


# Preserve historical import identities for pickles and public introspection.
for _identity_export in (
    native_mps,
    mps_marginal,
):
    _identity_export.__module__ = "flagquantum.runtime.mps_result"
del _identity_export

_alias_module("flagquantum.runtime.mps_result", __name__)
