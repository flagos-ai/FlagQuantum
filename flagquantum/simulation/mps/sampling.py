"""Memory-bounded computational-basis sampling for MPS states."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .state import MPSState

_DEFAULT_ELEMENT_BUDGET = 4_000_000


def sample_mps_indices(
    state: MPSState,
    shots: int,
    *,
    generator: torch.Generator | None = None,
    element_budget: int = _DEFAULT_ELEMENT_BUDGET,
) -> torch.Tensor:
    """Sample shots in bounded tensor batches without dense materialization."""

    outputs = torch.zeros(
        state.bsz,
        shots,
        dtype=torch.int64,
        device=state.device,
    )
    chunk_size = max(
        1,
        min(int(shots), int(element_budget) // max(1, state.parameter_count)),
    )
    for start in range(0, int(shots), chunk_size):
        count = min(chunk_size, int(shots) - start)
        work = type(state)(
            [tensor.repeat_interleave(count, dim=0) for tensor in state.tensors],
            config=state.config,
        )
        indices = torch.zeros(
            state.bsz * count,
            dtype=torch.int64,
            device=state.device,
        )
        for wire in range(state.n_wires):
            probabilities = work._wire_probabilities(wire)
            bit = torch.multinomial(
                probabilities,
                num_samples=1,
                replacement=True,
                generator=generator,
            ).squeeze(-1)
            indices = (indices << 1) | bit
            work._project_wire(wire, bit)
            work._normalize()
        outputs[:, start : start + count] = indices.reshape(state.bsz, count)
    return outputs


__all__ = ("sample_mps_indices",)
