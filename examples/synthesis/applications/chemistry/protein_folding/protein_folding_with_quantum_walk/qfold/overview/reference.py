"""Independently evaluate all sixteen retained QFold amplitudes."""

import math
from typing import cast

import torch

import flagquantum as fq


def walk_state(angles: tuple[float, float]) -> torch.Tensor:
    dtype = torch.complex128
    # The reference wire order is phi, psi, coin, move.
    rotation = torch.eye(16, dtype=dtype)
    for word in range(16):
        if word & 2:
            continue
        moving_psi = bool(word & 1)
        coordinate = word & (4 if moving_psi else 8)
        if coordinate:
            continue
        angle = angles[int(moving_psi)]
        partner = word | 2
        cosine, sine = math.cos(angle / 2), math.sin(angle / 2)
        rotation[word, word] = rotation[partner, partner] = cosine
        rotation[word, partner], rotation[partner, word] = -sine, sine

    transition = torch.zeros((16, 16), dtype=dtype)
    for word in range(16):
        changed = (4 if word & 1 else 8) if word & 2 else 0
        transition[word ^ changed, word] = 1

    wires = (0, 1, 2, 3)
    coin = fq.Circuit(4, dtype=dtype).x(2).h(3).any(*wires, matrix=rotation).x(2)
    oracle = torch.diag(
        torch.tensor([-1 if i & 3 == 0 else 1 for i in range(16)], dtype=dtype)
    )
    walk = (
        fq.Circuit(4, dtype=dtype)
        .h(0)
        .h(1)
        .compose(coin)
        .any(*wires, matrix=transition)
        .compose(coin.adjoint())
        .any(*wires, matrix=oracle)
    )
    return cast(torch.Tensor, walk.state()).reshape(16)


def complete_state(angles: tuple[float, float], wires: int) -> torch.Tensor:
    """Translate the logical reference into the original native allocation order."""
    expected = torch.zeros((16, 1 << (wires - 4)), dtype=torch.complex128)
    expected[:, 0] = walk_state(angles)
    return expected
