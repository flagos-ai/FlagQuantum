"""Contract exact marginal probabilities directly from an MPS."""

from __future__ import annotations

import torch

from .state import MPSState


def marginal_probabilities(state: MPSState, wires: tuple[int, ...]) -> torch.Tensor:
    """Return unnormalized joint probabilities in the requested MSB-first order."""
    if (
        len(wires) > 16
        or len(wires) != len(set(wires))
        or any(
            type(wire) is not int or not 0 <= wire < state.n_qubits for wire in wires
        )
    ):
        raise ValueError("MPS marginal needs at most sixteen distinct in-range wires")
    count = 1 << len(wires)
    patterns = torch.arange(count, dtype=torch.long, device=state.device)
    positions = {wire: len(wires) - index - 1 for index, wire in enumerate(wires)}
    environment = torch.ones(
        (state.bsz, count, 1, 1), dtype=state.dtype, device=state.device
    )
    for wire, site in enumerate(state.tensors):
        if wire in positions:
            bits = (patterns >> positions[wire]) & 1
            selected = site.index_select(2, bits).permute(0, 2, 1, 3)
            environment = torch.einsum(
                "bklm,bklr,bkms->bkrs", environment, selected, selected.conj()
            )
        else:
            environment = torch.einsum(
                "bklm,blir,bmis->bkrs", environment, site, site.conj()
            )
    return environment[..., 0, 0].real
