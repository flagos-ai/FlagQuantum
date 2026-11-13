# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 HammingWeightCompute.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/hamming_weight.py

"""Source Hamming tree stages over caller-bound big-endian registers."""

from typing import Literal

HammingStage = tuple[Literal["compress", "cx"], tuple[int, ...]]


def hamming_weight_stages(
    source: tuple[int, ...], junk: tuple[int, ...], output: tuple[int, ...]
) -> tuple[HammingStage, ...]:
    """Preserve little-endian traversal and reversed-junk pop order."""
    x, available = list(source[::-1]), list(junk[::-1])
    stages: list[HammingStage] = []
    for result in reversed(output):
        y = []
        for i in range(0, len(x) - 2, 2):
            anc = available.pop()
            y.append(anc)
            stages.append(("compress", (x[i], x[i + 1], x[i + 2], anc)))
        if len(x) % 2:
            stages.append(("cx", (x[-1], result)))
        else:
            anc = available.pop()
            stages.append(("compress", (x[-2], x[-1], result, anc)))
            y.append(anc)
        x = y
    return tuple(stages)
