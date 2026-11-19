"""Independent Fock-space Hamiltonian for the original two-mode case."""

import torch

COMPLEX = torch.complex128


def annihilation(size: int) -> list[torch.Tensor]:
    modes = []
    for i in range(size):
        a = torch.zeros((1 << size, 1 << size), dtype=COMPLEX)
        for seen in range(1 << size):
            if seen >> i & 1:
                a[seen ^ (1 << i), seen] = (-1) ** (seen & ((1 << i) - 1)).bit_count()
        modes.append(a)
    return modes


def fock_hamiltonian(
    h: torch.Tensor, delta: torch.Tensor, modes: list[torch.Tensor]
) -> torch.Tensor:
    out = torch.zeros_like(modes[0])
    for i, a in enumerate(modes):
        for j, b in enumerate(modes):
            pair = delta[i, j] * (a.mH @ b.mH)
            out += h[i, j] * (a.mH @ b) + (pair + pair.mH) / 2
    return out
