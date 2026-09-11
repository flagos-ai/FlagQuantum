"""Numerical noise kernels for local dynamic statevector trajectories."""

from __future__ import annotations

import torch

from ...noise import KrausChannel, ReadoutError
from .operations import _apply_fixed_permutation


def apply_dynamic_bit_flip(
    state: torch.Tensor,
    channel: KrausChannel,
    wire: int,
    n_wires: int,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    """Sample one bit-flip channel independently for each trajectory row."""

    if channel.name != "bit_flip" or channel.n_wires != 1:
        raise ValueError("dynamic noise kernel supports one-wire bit-flip channels")
    if state.ndim != 2:
        raise ValueError("dynamic noise state must have trajectory and amplitude axes")
    if wire < 0 or wire >= n_wires:
        raise ValueError("dynamic noise wire is outside the statevector")
    operators = tuple(
        torch.as_tensor(item, device=state.device, dtype=state.dtype)
        for item in channel.kraus
    )
    probability = torch.real(torch.trace(operators[1].mH @ operators[1]) / 2)
    events = (
        torch.rand(
            (state.shape[0],),
            generator=generator,
            device=state.device,
            dtype=state.real.dtype,
        )
        < probability
    )
    if not bool(torch.any(events)):
        return state, 0
    flipped = _apply_fixed_permutation(state, "x", (wire,), n_wires)
    return torch.where(events[:, None], flipped, state), int(events.sum().item())


def sample_dynamic_readout(
    true_bits: torch.Tensor,
    error: ReadoutError,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    """Sample independent observed bits from true-to-observed probabilities."""

    bits = torch.as_tensor(true_bits)
    if bits.ndim != 1 or bits.dtype != torch.int64:
        raise ValueError("dynamic readout requires a one-dimensional int64 bit tensor")
    if bool(torch.any((bits < 0) | (bits > 1))):
        raise ValueError("dynamic readout true bits must be binary")
    matrix = torch.as_tensor(
        error.probabilities,
        device=bits.device,
        dtype=torch.float64,
    )
    probabilities = matrix.index_select(0, bits)
    observed = torch.multinomial(
        probabilities,
        1,
        replacement=True,
        generator=generator,
    ).reshape(-1)
    return observed, int(torch.count_nonzero(observed != bits).item())


__all__ = ("apply_dynamic_bit_flip", "sample_dynamic_readout")
