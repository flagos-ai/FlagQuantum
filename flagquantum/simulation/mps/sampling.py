"""Memory-bounded computational-basis sampling for MPS states."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .state import MPSState

_DEFAULT_ELEMENT_BUDGET = 4_000_000


def _collapse_sampled_wire(state: MPSState, wire: int, bits: torch.Tensor) -> None:
    """Collapse one left-to-right sample step without rank-deficient QR."""

    tensor = state.tensors[wire]
    if tensor.shape[1] != 1:
        raise RuntimeError("sequential MPS sampling requires a unit left bond")
    batches = torch.arange(state.bsz, device=state.device)
    boundary = tensor[batches, 0, bits, :]
    norms = torch.linalg.vector_norm(boundary, dim=-1)
    if bool(torch.any(~torch.isfinite(norms))) or bool(torch.any(norms <= 1e-12)):
        raise RuntimeError("sampled MPS branch is not finite and positive")
    boundary = boundary / norms[:, None]

    basis = torch.zeros(
        state.bsz,
        1,
        2,
        1,
        dtype=state.dtype,
        device=state.device,
    )
    basis[batches, 0, bits, 0] = 1
    state.tensors[wire] = basis
    if wire + 1 < state.n_wires:
        state.tensors[wire + 1] = torch.einsum(
            "bl,blsr->bsr",
            boundary,
            state.tensors[wire + 1],
        ).unsqueeze(1)
        state.orthogonality_center = wire + 1
    else:
        state.orthogonality_center = wire
    state._canonical_center_valid = True


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
        work.orthogonality_center = state.orthogonality_center
        work._canonical_center_valid = state._canonical_center_valid
        for wire in range(state.n_wires):
            probabilities = work._wire_probabilities(wire)
            bit = torch.multinomial(
                probabilities,
                num_samples=1,
                replacement=True,
                generator=generator,
            ).squeeze(-1)
            outputs[:, start : start + count, wire] = bit.reshape(state.bsz, count)
            _collapse_sampled_wire(work, wire, bit)
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
