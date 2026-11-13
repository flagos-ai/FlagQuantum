# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 HammingWeightCompute.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/hamming_weight.py

"""Literal logical sum/carry components shared by emission and width binding."""

from typing import Literal

CompressionComponent = tuple[Literal["cx", "and"], tuple[int, ...]]


def compression_components(
    a: int, b: int, total: int, carry: int
) -> tuple[CompressionComponent, ...]:
    """Return the source forward three-to-two adder over explicit wire identities."""
    return (
        ("cx", (a, b)),
        ("cx", (a, total)),
        ("and", (b, total, carry)),
        ("cx", (a, b)),
        ("cx", (a, carry)),
        ("cx", (b, total)),
    )
