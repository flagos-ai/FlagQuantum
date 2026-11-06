# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 RotationTree.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/state_preparation_via_rotation.py

"""Concrete angle tables for phase-gradient state preparation."""

from cmath import phase
from math import acos, isclose, pi, sqrt, tau


def rotation_tables(
    coefficients: tuple[complex, ...], phase_width: int, *, uncompute: bool = False
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Return big-endian prefix amplitude rows and a full-address phase row."""
    size = len(coefficients)
    if not size or size & (size - 1):
        raise ValueError("Rotation preparation requires a power-of-two state size")
    if type(phase_width) is not int or phase_width < 2:
        raise ValueError("Rotation preparation requires at least two phase bits")
    if type(uncompute) is not bool:
        raise ValueError("Rotation preparation direction must be boolean")

    weights = [0.0] * size + [abs(value) ** 2 for value in coefficients]
    for node in range(size - 1, 0, -1):
        weights[node] = weights[2 * node] + weights[2 * node + 1]
    if not isclose(sqrt(weights[1]), 1, rel_tol=1e-5, abs_tol=1e-8):
        raise ValueError("Rotation preparation requires normalized coefficients")

    modulus = 1 << phase_width

    def quantize(angle: float) -> int:
        return round(modulus * angle / tau) % modulus

    rows = []
    offsets = [0.0] * size
    for level in range(size.bit_length() - 1):
        first = 1 << level
        span = size // first
        row = []
        for address, node in enumerate(range(first, 2 * first)):
            p_zero = weights[2 * node] / weights[node] if weights[node] else 0.0
            angle = 2 * acos(sqrt(p_zero))
            word = quantize(tau - angle if uncompute else angle)
            row.append(word)
            forward_word = -word % modulus if uncompute else word
            offset = (pi * forward_word / modulus) % pi
            for index in range(address * span, (address + 1) * span):
                offsets[index] += offset
        rows.append(tuple(row))
    sign = -1 if uncompute else 1
    phases = tuple(
        quantize(sign * (phase(value) - offset))
        for value, offset in zip(coefficients, offsets, strict=True)
    )
    return tuple(rows), phases
