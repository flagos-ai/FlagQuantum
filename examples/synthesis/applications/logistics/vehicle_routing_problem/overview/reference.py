"""Independent Circuit compiled from a scalar Boolean cost oracle."""

from typing import cast

import torch

import flagquantum as fq

from .problem import LAYERS, WIDTH, Problem, scalar_cost


def coefficients(
    problem: Problem,
) -> tuple[float, tuple[float, ...], tuple[tuple[int, int, float], ...]]:
    base = scalar_cost(problem, 0)
    linear = tuple(scalar_cost(problem, 1 << bit) - base for bit in range(WIDTH))
    pairs = tuple(
        (
            first,
            second,
            scalar_cost(problem, (1 << first) | (1 << second))
            - base
            - linear[first]
            - linear[second],
        )
        for first in range(WIDTH)
        for second in range(first + 1, WIDTH)
    )
    return base, linear, pairs


def circuit(problem: Problem, parameters: tuple[float, ...]) -> fq.Circuit:
    base, linear, pairs = coefficients(problem)
    result = fq.Circuit(WIDTH, dtype=torch.complex128)
    for bit in range(WIDTH):
        result.h(bit)
    for layer in range(LAYERS):
        gamma, beta = parameters[2 * layer : 2 * layer + 2]
        result.rz(0, theta=-gamma * base).phase(0, theta=gamma * base)
        result.rz(0, theta=-gamma * base).phase(0, theta=gamma * base)
        for bit, coefficient in enumerate(linear):
            result.phase(WIDTH - 1 - bit, theta=gamma * coefficient)
        for first, second, coefficient in pairs:
            if abs(coefficient) > 1e-14:
                result.cphase(
                    WIDTH - 1 - first, WIDTH - 1 - second, theta=gamma * coefficient
                )
        for bit in range(WIDTH):
            result.rx(bit, theta=0.1 * beta)
    return result


def state(problem: Problem, parameters: tuple[float, ...]) -> torch.Tensor:
    return cast(torch.Tensor, circuit(problem, parameters).state()).reshape(-1)
