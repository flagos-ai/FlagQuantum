"""Fresh original-sized log-normal distribution, payoff and native Grover family."""

import math
from dataclasses import dataclass

from examples.synthesis.applications.finance.option_pricing.main import main
from flagquantum import synth


@dataclass(frozen=True)
class OptionModel:
    mu: float
    sigma: float
    strike: float
    grid: tuple[float, ...]
    probabilities: tuple[float, ...]
    angles: tuple[float, ...]
    cutoff: int
    scale: float
    expected_payoff: float


def fresh_model(
    mu: float = 0.7, sigma: float = 0.13, strike: float = 1.9
) -> OptionModel:
    if (
        any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in (mu, sigma, strike)
        )
        or not -2 <= mu <= 2
        or not 0 < sigma <= 0.3
    ):
        raise ValueError("use finite bounded log-normal parameters")
    mean = math.exp(mu + sigma**2 / 2)
    variance = (math.exp(sigma**2) - 1) * math.exp(2 * mu + sigma**2)
    deviation = math.sqrt(variance)
    low, high = max(0.0, mean - 3 * deviation), mean + 3 * deviation
    grid = tuple(low + (high - low) * word / 31 for word in range(32))
    if not low < strike < high:
        raise ValueError("strike must lie strictly inside the original grid")
    raw = tuple(
        (
            math.exp(-(((math.log(price) - mu) / sigma) ** 2) / 2)
            / (price * sigma * math.sqrt(2 * math.pi))
            if price > 0
            else 0.0
        )
        for price in grid
    )
    norm = sum(raw)
    probabilities = tuple(value / norm for value in raw)
    scale = max(grid) - strike
    cutoff = math.ceil((strike - min(grid)) / ((max(grid) - min(grid)) / 31))
    angles = tuple(
        (
            2 * math.asin(math.sqrt(abs((price - strike) / scale)))
            if word >= cutoff
            else 0.0
        )
        for word, price in enumerate(grid)
    )
    expected_payoff = sum(
        max(price - strike, 0) * probability
        for price, probability in zip(grid, probabilities, strict=True)
    )
    return OptionModel(
        float(mu),
        float(sigma),
        float(strike),
        grid,
        probabilities,
        angles,
        cutoff,
        scale,
        expected_payoff,
    )


def build_program(model: OptionModel, k: int, max_width: int = 20) -> synth.Program:
    if (
        type(k) is not int
        or not 0 <= k <= 16
        or type(max_width) is not int
        or max_width not in (20, 28)
    ):
        raise ValueError("retain k in0..16 and source width budget20or28")
    return synth.synthesize(
        main,
        classical_inputs={
            "k": k,
            "probabilities": list(model.probabilities),
            "angles": list(model.angles),
        },
        max_qubits=max_width,
        max_workspace=0,
        memory_limit_bytes=64 * 1024**2,
    )
