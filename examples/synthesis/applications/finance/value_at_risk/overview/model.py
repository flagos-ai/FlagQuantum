"""Fresh original seven-bit log-normal loss model and exclusive CDF search."""

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class LossModel:
    mu: float
    sigma: float
    grid: tuple[float, ...]
    probabilities: tuple[float, ...]

    def alpha(self, index: int) -> float:
        """Return the source exclusive probability P(asset < index)."""
        if type(index) is not int or not 0 <= index <= 128:
            raise ValueError("threshold must be an integer in0..128")
        return sum(self.probabilities[:index])


def fresh_model(mu: float = 0.7, sigma: float = 0.13) -> LossModel:
    if (
        any(type(x) not in (int, float) or not math.isfinite(x) for x in (mu, sigma))
        or not -2 <= mu <= 2
        or not 0 < sigma <= 0.3
    ):
        raise ValueError("use finite bounded log-normal parameters")
    mean = math.exp(mu + sigma**2 / 2)
    variance = (math.exp(sigma**2) - 1) * math.exp(2 * mu + sigma**2)
    low, high = max(0.0, mean - 3 * math.sqrt(variance)), mean + 3 * math.sqrt(variance)
    grid = tuple(low + (high - low) * word / 127 for word in range(128))
    density = tuple(
        (
            math.exp(-0.5 * ((math.log(x) - mu) / sigma) ** 2)
            / (x * sigma * math.sqrt(2 * math.pi))
            if x > 0
            else 0.0
        )
        for x in grid
    )
    total = sum(density)
    return LossModel(float(mu), float(sigma), grid, tuple(x / total for x in density))


@dataclass(frozen=True)
class SearchStep:
    index: int
    value: float
    alpha: float
    search_size: int


@dataclass(frozen=True)
class SearchResult:
    index: int
    asset_value: float
    steps: tuple[SearchStep, ...]
    tolerance_reached: bool
    original_return: None = None


def search(model: LossModel, alpha_at: Callable[[int], float]) -> SearchResult:
    """Preserve the source bisection, including exhaustion before convergence.

    The source function prints its result but implicitly returns None. This
    explicit diagnostic record exposes its internal index without claiming a
    repaired source return value. The first printed v is probs[index] in the
    source; subsequent values are asset prices, and both are retained here.
    """
    required_alpha, tolerance = 0.07, 0.007
    index, search_size = 32, 16
    measured = alpha_at(index)
    steps = [SearchStep(index, model.probabilities[index], measured, search_size)]
    close = math.isclose(measured, required_alpha, abs_tol=tolerance, rel_tol=1e-5)
    while not close and search_size > 0:
        index += search_size if measured < required_alpha else -search_size
        search_size //= 2
        measured = alpha_at(index)
        steps.append(SearchStep(index, model.grid[index], measured, search_size))
        close = math.isclose(measured, required_alpha, abs_tol=tolerance, rel_tol=1e-5)
    return SearchResult(index, model.grid[index], tuple(steps), close)


def discrete_quantile(model: LossModel) -> tuple[int, float]:
    accumulated = 0.0
    for index, probability in enumerate(model.probabilities):
        accumulated += probability
        if accumulated > 0.07:
            return index, model.grid[index]
    raise AssertionError("normalized distribution did not reach the requested quantile")
