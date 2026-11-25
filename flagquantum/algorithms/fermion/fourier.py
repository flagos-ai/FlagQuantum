"""Radix-2 Fourier factors on adjacent orbitals, independent of circuits."""

from __future__ import annotations

import cmath
import math

import torch

from ...core.algebra.pauli import count
from .decomposition import AdjacentRotation, OrbitalDecomposition


def fourier_decomposition(
    n_modes: int, *, inverse: bool = False
) -> OrbitalDecomposition:
    """Factor the normalized discrete Fourier transform into adjacent rotations.

    The forward convention is ``F[p,q] = exp(-2j*pi*p*q/n_modes)/sqrt(n_modes)``
    on column creation amplitudes. Input and output modes use natural order;
    no bit reversal or residual layout permutation is left to the consumer.
    The inverse realizes F dagger, with the same normalization.

    Radix-2 butterflies are routed by adjacent orbital swaps. There are
    O(n_modes**2) adjacent factors and storage, including routing, rather than
    an all-to-all O(n_modes*log(n_modes)) gate-count claim. No dense Fourier or
    Fock matrix is constructed. Factors are static CPU complex128 tensors.

    Args:
        n_modes: Positive power of two, including one.
        inverse: Whether to produce the inverse transform.

    Returns:
        Existing orbital decomposition, in application order, with unit phases.

    Raises:
        TypeError: Width is not an integer or inverse is not a bool.
        ValueError: Width is not a positive power of two.
    """
    width = count(n_modes, "n_modes", minimum=1)
    if width & (width - 1):
        raise ValueError("n_modes must be a power of two")
    if not isinstance(inverse, bool):
        raise TypeError("inverse must be a bool")
    rotations: list[AdjacentRotation] = []
    swap = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)

    def split(start: int, size: int) -> None:
        if size == 1:
            return
        half = size // 2
        # Stable even/odd partition of the current orbital amplitudes.
        for target in range(half):
            for left in range(start + 2 * target - 1, start + target - 1, -1):
                rotations.append(AdjacentRotation(left, swap.clone()))
        split(start, half)
        split(start + half, half)
        for mode in range(half):
            left, right = start + mode, start + half + mode
            # Bring the odd Fourier component next to its even partner, then
            # restore the intervening modes after the butterfly.
            route = range(right - 1, left, -1)
            for index in route:
                rotations.append(AdjacentRotation(index, swap.clone()))
            phase = cmath.exp(-2j * math.pi * mode / size)
            butterfly = torch.tensor(
                [[1, phase], [1, -phase]], dtype=torch.complex128
            ) / math.sqrt(2)
            rotations.append(AdjacentRotation(left, butterfly))
            for index in reversed(route):
                rotations.append(AdjacentRotation(index, swap.clone()))

    split(0, width)
    if inverse:
        rotations = [
            AdjacentRotation(step.left, step.matrix.mH.contiguous())
            for step in reversed(rotations)
        ]
    return OrbitalDecomposition(
        torch.ones(width, dtype=torch.complex128), tuple(rotations)
    )
