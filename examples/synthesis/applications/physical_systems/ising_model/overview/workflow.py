"""Bounded native six-spin periodic Ising optimization."""

import math
from dataclasses import dataclass

import flagquantum as fq
from examples.synthesis.applications.physical_systems.ising_model.main import (
    build_program,
)
from flagquantum import synth


def energy(word: int) -> float:
    """Source objective J=10,h=-20,periodic; opposite to the frozen export cost."""
    if type(word) is not int or not 0 <= word < 64:
        raise ValueError("Use a six-bit integer configuration in 0..63.")
    spins = tuple(2 * ((word >> i) & 1) - 1 for i in range(6))
    return float(
        -10 * sum(spins[i] * spins[(i + 1) % 6] for i in range(6)) + 20 * sum(spins)
    )


def source_program(parameters: tuple[float, ...]) -> synth.Program:
    """Reuse the frozen unitary with gamma sign changed to source-energy phases."""
    if len(parameters) != 10 or any(
        type(x) not in (int, float) or not math.isfinite(x) for x in parameters
    ):
        raise ValueError("Five Ising layers require ten finite real parameters.")
    return build_program(tuple(-x for x in parameters[:5]) + parameters[5:])


def lower_tail_cost(
    rows: tuple[tuple[int, float], ...], quantile: float = 0.7
) -> float:
    """Integrate the lowest-energy quantile with fractional boundary mass."""
    if (
        type(quantile) not in (int, float)
        or not math.isfinite(quantile)
        or not 0 < quantile <= 1
    ):
        raise ValueError("Use a finite quantile in(0,1].")
    if (
        len({word for word, _ in rows}) != len(rows)
        or any(type(word) is not int or not 0 <= word < 64 for word, _ in rows)
        or any(
            type(p) not in (int, float) or not math.isfinite(p) or p < 0
            for _, p in rows
        )
        or abs(sum(p for _, p in rows) - 1) > 1e-10
    ):
        raise ValueError(
            "Use a normalized nonnegative finite probability distribution."
        )
    remaining, total = quantile, 0.0
    for bits, p in sorted(rows, key=lambda row: energy(row[0])):
        take = min(remaining, p)
        total += take * energy(bits)
        remaining -= take
        if remaining <= 1e-14:
            break
    if remaining > 1e-10:
        raise ValueError("Distribution does not cover the requested tail.")
    return total / quantile


@dataclass(frozen=True)
class Optimization:
    """Best visited parameters, every evaluated tail cost and exact probabilities."""

    parameters: tuple[float, ...]
    initial_cost: float
    best_cost: float
    trace: tuple[float, ...]
    rows: tuple[tuple[int, float], ...]
    termination: str


def optimize() -> Optimization:
    """Use an explicit 100-evaluation coordinate search; no private solver claim."""
    point = [0.011, 0.017, 0.023, 0.029, 0.031, 0.13, 0.19, 0.23, 0.29, 0.31]
    trace: list[float] = []
    best_parameters = tuple(point)
    best_value = math.inf
    best_rows: tuple[tuple[int, float], ...] = ()

    def evaluate(values: list[float]) -> float:
        nonlocal best_parameters, best_value, best_rows
        program = source_program(tuple(values))
        rows = tuple(
            (key[0], probability)
            for key, probability in program.probabilities(
                fq.run(program), registers=("v",), threshold=0
            ).items()
        )
        value = lower_tail_cost(rows)
        trace.append(value)
        if value < best_value:
            best_parameters, best_value, best_rows = tuple(values), value, rows
        return value

    value = evaluate(point)
    steps = [0.01] * 5 + [0.2] * 5
    for _ in range(5):
        for coordinate in range(10):
            candidates = []
            for sign in (-1, 1):
                if len(trace) >= 100:
                    break
                trial = point[:]
                trial[coordinate] += sign * steps[coordinate]
                candidates.append((evaluate(trial), trial))
            if candidates:
                candidate = min(candidates, key=lambda pair: pair[0])
                if candidate[0] < value:
                    value, point = candidate
            if len(trace) >= 100:
                break
        steps = [step * 0.65 for step in steps]
        if len(trace) >= 100:
            break
    return Optimization(
        best_parameters,
        trace[0],
        best_value,
        tuple(trace),
        best_rows,
        "100-evaluation budget exhausted; no convergence certification",
    )
