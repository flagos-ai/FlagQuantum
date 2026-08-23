"""Measurements computed from exact density matrices."""

from __future__ import annotations

from typing import Iterable

import torch


def expectation_z_density(
    rho: torch.Tensor,
    wires: Iterable[int] | int | None = None,
) -> torch.Tensor:
    """Compute Z expectations from a density matrix."""

    if rho.ndim == 2:
        rho = rho.reshape(1, *rho.shape)
    n_wires = int(torch.log2(torch.tensor(rho.shape[-1], dtype=torch.float32)).item())
    if wires is None:
        target_wires = tuple(range(n_wires))
    elif isinstance(wires, int):
        target_wires = (wires,)
    else:
        target_wires = tuple(int(wire) for wire in wires)

    probs = torch.real(torch.diagonal(rho, dim1=-2, dim2=-1))
    shaped = probs.reshape((rho.shape[0],) + (2,) * n_wires)
    values = []
    for wire in target_wires:
        axes = tuple(axis for axis in range(1, n_wires + 1) if axis != wire + 1)
        marginal = shaped.sum(dim=axes) if axes else shaped
        values.append(marginal[:, 0] - marginal[:, 1])
    return torch.stack(values, dim=-1)


__all__ = ("expectation_z_density",)
