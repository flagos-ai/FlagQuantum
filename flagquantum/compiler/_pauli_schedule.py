"""Stream ordered product-formula steps for validated Pauli decompositions."""

import math
import random
from collections.abc import Iterator, Sequence

MAX_QDRIFT_SAMPLES = 2048
DEFAULT_QDRIFT_SEED = 0


def sampled_pauli_steps(
    coefficients: Sequence[float], time: float, samples: int, seed: int
) -> Iterator[tuple[int, float]]:
    """Sample ordered term indices and signed angles for a validated qDRIFT call.

    Frontends own finite coefficients, positive bounded sample counts and seed
    validation. Sampling keeps duplicate and identity terms in their declared
    positions, so equivalent sums need not have identical sampled sequences.
    """
    weights = [abs(value) for value in coefficients]
    total = math.fsum(weights)
    if not total or not time:
        return
    angle = time * total / samples
    if not math.isfinite(2 * angle):
        raise ValueError("sampled Pauli angle and doubled angle must be finite")
    for index in random.Random(seed).choices(
        range(len(weights)), weights=weights, k=samples
    ):
        yield index, angle if coefficients[index] > 0 else -angle


def ordered_pauli_steps(
    term_count: int, time: float, order: int, repetitions: int
) -> Iterator[tuple[int, float]]:
    """Yield term index and evolution time without changing declared order.

    Callers own operator validation, finite time, order 1/2, positive repetition
    counts and their resource limits. The schedule owns only Lie/Strang ordering
    and division order; coefficients and live gradient values stay with callers.
    No term sequence or repeated list is materialized here.
    """
    step = time / repetitions
    if order == 2:
        step /= 2
    for _ in range(repetitions):
        for index in range(term_count):
            yield index, step
        if order == 2:
            for index in range(term_count - 1, -1, -1):
                yield index, step
