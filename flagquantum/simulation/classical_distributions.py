# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ClassicalValDistribution and classical value handlers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/classical_sim.py

"""Resolve numerical output distributions without graph or backend ownership."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ClassicalDistribution:
    """Retain choices and probabilities for the original NumPy choice call.

    Arrays and sequences are retained by identity without validation. Their
    runtime domain and shape errors are reported by the random generator.
    """

    choices: int | NDArray[Any] | Sequence[int | float | complex | str | bytes]
    probabilities: NDArray[np.floating[Any]] | Sequence[float] | None = None

    def __hash__(self) -> int:
        # The upstream frozen distribution explicitly disables hashing.
        raise TypeError("unhashable type: 'ClassicalDistribution'")


def draw_classical_value(
    distribution: ClassicalDistribution,
    *,
    instance_id: int,
    rng: np.random.Generator | None = None,
    fixed_values: Mapping[int, object] | None = None,
) -> object:
    """Prefer the supplied RNG, then fixed values; otherwise reject randomness.

    Fixed results are retrieved by instance identity without checking the
    distribution or copying the result, just as the original fixed handler does.
    No implicit random generator, seeding or fallback is introduced.
    """
    if rng is not None:
        return rng.choice(distribution.choices, p=distribution.probabilities)
    if fixed_values is not None:
        return fixed_values[instance_id]
    raise ValueError(
        f"Instance {instance_id} has non-deterministic classical action."
        "Cannot simulate with classical values."
    )
