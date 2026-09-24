"""Memory-bounded computational-basis sampling for MPS states."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .state import MPSState

_DEFAULT_ELEMENT_BUDGET = 4_000_000


def sample_mps_bits(
    state: MPSState,
    shots: int,
    *,
    generator: torch.Generator | None = None,
    element_budget: int = _DEFAULT_ELEMENT_BUDGET,
) -> torch.Tensor:
    """Sample bit strings in bounded batches without dense materialization."""

    outputs = torch.zeros(
        state.bsz,
        shots,
        state.n_wires,
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
        for wire in range(state.n_wires):
            probabilities = work._wire_probabilities(wire)
            bit = torch.multinomial(
                probabilities,
                num_samples=1,
                replacement=True,
                generator=generator,
            ).squeeze(-1)
            outputs[:, start : start + count, wire] = bit.reshape(state.bsz, count)
            work._project_wire(wire, bit)
            work._normalize()
    return outputs


def sample_mps_indices(
    state: MPSState,
    shots: int,
    *,
    generator: torch.Generator | None = None,
    element_budget: int = _DEFAULT_ELEMENT_BUDGET,
) -> torch.Tensor:
    """Sample integer basis indices when they fit in signed int64."""

    if state.n_wires > 63:
        raise ValueError(
            "integer-index MPS samples support at most 63 qubits; "
            "request format='bits' for wider circuits"
        )
    bits = sample_mps_bits(
        state,
        shots,
        generator=generator,
        element_budget=element_budget,
    )
    shifts = torch.arange(state.n_wires - 1, -1, -1, device=state.device)
    return torch.sum(bits << shifts, dim=-1)


__all__ = ("sample_mps_bits", "sample_mps_indices")
