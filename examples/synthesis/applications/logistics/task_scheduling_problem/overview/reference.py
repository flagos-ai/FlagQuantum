"""Independent Circuit derived by querying the original scalar constraints."""

import torch

import flagquantum as fq

from .problem import Problem, original_cost


def coefficients(
    problem: Problem,
) -> tuple[float, tuple[float, ...], tuple[tuple[int, int, float], ...]]:
    constant = original_cost(problem, 0)
    linear = tuple(
        original_cost(problem, 1 << bit) - constant for bit in range(problem.width)
    )
    pairs = tuple(
        (
            first,
            second,
            original_cost(problem, (1 << first) | (1 << second))
            - constant
            - linear[first]
            - linear[second],
        )
        for first in range(problem.width)
        for second in range(first + 1, problem.width)
    )
    return constant, linear, pairs


def circuit(problem: Problem, parameters: tuple[float, ...]) -> fq.Circuit:
    constant, linear, pairs = coefficients(problem)
    result = fq.Circuit(problem.width, dtype=torch.complex128)
    for bit in range(problem.width):
        result.h(bit)
    for layer in range(8):
        gamma, beta = parameters[2 * layer : 2 * layer + 2]
        # Two equal scalar half-phases preserve the full e^-i*gamma*constant.
        for _ in range(2):
            result.rz(0, theta=gamma * constant).phase(0, theta=-gamma * constant)
        for bit, weight in enumerate(linear):
            result.phase(problem.width - 1 - bit, theta=-gamma * weight)
        for first, second, weight in pairs:
            if weight:
                result.cphase(
                    problem.width - 1 - first,
                    problem.width - 1 - second,
                    theta=-gamma * weight,
                )
        for bit in range(problem.width):
            result.rx(bit, theta=beta)
    return result
