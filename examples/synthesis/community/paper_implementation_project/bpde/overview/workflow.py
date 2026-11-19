"""Sample the original observable and update the full Bayesian fitting workflow."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from . import model, quantum

SEED = 42
SHOTS = 1_000_000
TRIALS = tuple(float(x) for x in np.linspace(0.00035, 0.0007, 5))
TIME_STUDY = (20.0, 100.0, 200.0)
THRESHOLD = 0.001
ITERATIONS = 30


@dataclass(frozen=True)
class Scan:
    time: float
    repetitions: int
    energies: tuple[float, ...]
    zero_counts: tuple[int, ...]
    exact_zero: tuple[float, ...]
    maximum_error: float
    resources: int

    @property
    def z_estimates(self) -> tuple[float, ...]:
        return tuple(2 * count / SHOTS - 1 for count in self.zero_counts)


@dataclass(frozen=True)
class Update:
    prior_mu: float
    prior_sigma: float
    posterior_mu: float
    posterior_sigma: float
    scan: Scan


@dataclass(frozen=True)
class Result:
    standard: model.Hydrogen
    curve: tuple[model.Hydrogen, ...]
    a: model.Hydrogen
    b: model.Hydrogen
    check: Scan
    diagnostic: Scan
    study: tuple[Scan, ...]
    history: tuple[Update, ...]


def scan(
    models: tuple[model.Hydrogen, model.Hydrogen], time: float, grid: tuple[float, ...]
) -> Scan:
    circuit = quantum.build(models, time)
    errors = []
    exact = []
    for offset in grid:
        actual = quantum.execute(circuit, offset)
        error = float((actual - quantum.reference(models, time, offset)).abs().max())
        errors.append(error)
        exact.append(float(actual[:2].abs().square().sum()))
    maximum = max(errors)
    if maximum >= 1e-10:
        raise AssertionError(
            f"Packed Strang circuit disagrees with unfolded reference: {maximum}"
        )
    rng = np.random.default_rng(SEED)
    counts = tuple(
        int(rng.binomial(SHOTS, min(1.0, max(0.0, probability))))
        for probability in exact
    )
    return Scan(
        time,
        circuit.repetitions,
        grid,
        counts,
        tuple(exact),
        maximum,
        circuit.resources.n_operations,
    )


def gaussian(
    x: NDArray[np.float64], amplitude: float, mean: float, sigma: float
) -> NDArray[np.float64]:
    return np.asarray(
        amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2), dtype=np.float64
    )


def posterior(models: tuple[model.Hydrogen, model.Hydrogen]) -> tuple[Update, ...]:
    from scipy.optimize import curve_fit

    mu, sigma, time = 0.0, 0.01, 4.0
    history = []
    for _ in range(ITERATIONS):
        energies = tuple(
            float(value) for value in np.linspace(mu - sigma, mu + sigma, 5)
        )
        measured = scan(models, time, energies)
        fit, _ = curve_fit(
            gaussian,
            np.asarray(energies),
            np.asarray(measured.z_estimates),
            p0=[1.0, mu, sigma],
            maxfev=10_000,
        )
        _, fit_mu, fit_sigma = (float(value) for value in fit)
        prior_precision, fit_precision = 1 / sigma**2, 1 / fit_sigma**2
        combined = prior_precision + fit_precision
        next_mu = (mu * prior_precision + fit_mu * fit_precision) / combined
        next_sigma = math.sqrt(1 / combined)
        history.append(Update(mu, sigma, next_mu, next_sigma, measured))
        if next_sigma < THRESHOLD:
            break
        mu, sigma, time = next_mu, next_sigma, 1.8 / next_sigma
    return tuple(history)


def run() -> Result:
    standard = model.hydrogen(0.74)
    rows = model.curve()
    a, b = displaced = model.displaced()
    check = scan(displaced, 1.0, (0.1, 0.2, 0.3))
    diagnostic = scan(displaced, 200.0, TRIALS)
    study = tuple(scan(displaced, time, TRIALS) for time in TIME_STUDY)
    return Result(standard, rows, a, b, check, diagnostic, study, posterior(displaced))
