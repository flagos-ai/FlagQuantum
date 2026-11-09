"""Stream ordered product-formula steps for validated Pauli decompositions."""

from collections.abc import Iterator


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
