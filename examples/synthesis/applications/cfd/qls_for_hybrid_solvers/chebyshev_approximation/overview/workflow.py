"""Compare reciprocal polynomials and the prepared inverse LCU weights."""

from dataclasses import dataclass

import torch

from .model import (
    APPROXIMATIONS,
    Approximation,
    Model,
    build_model,
    coefficient_approximation,
    fit_alternating,
)
from .quantum import Preparation, compare


@dataclass(frozen=True)
class Result:
    nozzle: Model
    approximations: tuple[Approximation, ...]
    fitted: torch.Tensor
    exact: Preparation
    approximate: Preparation


def approximations(nozzle: Model) -> tuple[Approximation, ...]:
    return tuple(
        coefficient_approximation(nozzle, degree, method)
        for degree, method in APPROXIMATIONS
    )


def run_workflow() -> Result:
    nozzle = build_model()
    cases = approximations(nozzle)
    fitted = fit_alternating(cases[1].coefficients)
    exact, approximate = compare(fitted)
    return Result(nozzle, cases, fitted, exact, approximate)
