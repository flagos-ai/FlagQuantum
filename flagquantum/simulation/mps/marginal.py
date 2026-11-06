"""Contract exact marginal probabilities directly from an MPS."""

from __future__ import annotations

import torch

from .state import MPSState


def marginal_probabilities(state: MPSState, qubits: tuple[int, ...]) -> torch.Tensor:
    """Return unnormalized joint probabilities in the requested MSB-first order."""
    if (
        len(qubits) > 16
        or len(qubits) != len(set(qubits))
        or any(
            type(qubit) is not int or not 0 <= qubit < state.n_qubits
            for qubit in qubits
        )
    ):
        raise ValueError("MPS marginal needs at most sixteen distinct in-range qubits")
    count = 1 << len(qubits)
    patterns = torch.arange(count, dtype=torch.long, device=state.device)
    positions = {qubit: len(qubits) - index - 1 for index, qubit in enumerate(qubits)}
    environment = torch.ones(
        (state.bsz, count, 1, 1), dtype=state.dtype, device=state.device
    )
    for qubit, site in enumerate(state.tensors):
        if qubit in positions:
            bits = (patterns >> positions[qubit]) & 1
            selected = site.index_select(2, bits).permute(0, 2, 1, 3)
            environment = torch.einsum(
                "bklm,bklr,bkms->bkrs", environment, selected, selected.conj()
            )
        else:
            environment = torch.einsum(
                "bklm,blir,bmis->bkrs", environment, site, site.conj()
            )
    return environment[..., 0, 0].real
