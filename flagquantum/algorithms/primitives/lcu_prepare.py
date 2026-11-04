"""Prepare the index register for a real linear combination of unitaries."""

import math
from collections.abc import Sequence

import torch

from ...circuit import Circuit
from .state_preparation import arbitrary_state


def prepare_lcu(coefficients: Sequence[float]) -> tuple[Circuit, float]:
    """Return PREPARE and the L1 normalization for real LCU coefficients.

    The circuit prepares amplitudes proportional to the square roots of
    coefficient magnitudes. SELECT must apply each coefficient's sign.
    The address register has at least one qubit and unused addresses have
    zero amplitude. Coefficients are classical data.
    """
    weights = tuple(abs(coefficient) for coefficient in coefficients)
    if not weights or not all(math.isfinite(weight) for weight in weights):
        raise ValueError("LCU coefficients must be nonempty and finite.")
    alpha = sum(weights)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LCU normalization must be finite and positive.")
    size = 1 << max(1, (len(weights) - 1).bit_length())
    amplitudes = torch.zeros(size, dtype=torch.float64)
    for index, weight in enumerate(weights):
        amplitudes[index] = math.sqrt(weight / alpha)
    return arbitrary_state(amplitudes), alpha
