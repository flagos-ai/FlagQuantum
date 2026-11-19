"""Construct the source LABS objective, including its unusual lag range."""

from collections import Counter

import flagquantum as fq

WIDTH = 13
LAYERS = 3


def source_cost(word: int, width: int = WIDTH) -> int:
    """Evaluate lags 0..width-2 exactly as written in the original Notebook."""
    if type(width) is not int or not 2 <= width <= WIDTH:
        raise ValueError("width must be an integer in 2..13")
    if type(word) is not int or not 0 <= word < 1 << width:
        raise ValueError("word must fit the declared width")
    spins = tuple(2 * ((word >> bit) & 1) - 1 for bit in range(width))
    return sum(
        sum(spins[bit] * spins[bit + lag] for bit in range(width - lag)) ** 2
        for lag in range(width - 1)
    )


def cost_terms(width: int = WIDTH) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Expand squared correlations by cancelling repeated Z factors."""
    if type(width) is not int or not 2 <= width <= WIDTH:
        raise ValueError("width must be an integer in 2..13")
    coefficients: Counter[tuple[int, ...]] = Counter()
    for lag in range(width - 1):
        factors = tuple(
            frozenset((bit,)) ^ frozenset((bit + lag,)) for bit in range(width - lag)
        )
        for left in factors:
            for right in factors:
                coefficients[tuple(sorted(left ^ right))] += 1
    return tuple(sorted(coefficients.items()))


def model_hamiltonian(width: int = WIDTH) -> fq.Observable:
    """Generate the diagonal Hamiltonian freshly from the classical formula."""
    terms = cost_terms(width)
    operator = dict(terms)[()] * fq.I()
    for support, coefficient in terms:
        if not support:
            continue
        term = fq.Z(support[0])
        for bit in support[1:]:
            term = term @ fq.Z(bit)
        operator += coefficient * term
    return operator
