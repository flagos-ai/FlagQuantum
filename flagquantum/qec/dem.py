"""Exact detector error models for code-independent memory experiments.

A detector error model names every independent physical error mechanism by the
detectors and logical observables it flips. Construction is exact and does not
sample: the reference gate set is Clifford and every configured channel is a
Pauli channel, so forcing one mechanism through the circuit yields a
deterministic signature.

The model is defined for Pauli noise only. It is built on the memory circuit
*without* in-circuit feedback, because the model describes the physical
noise-to-detection mapping that a decoder inverts; the compiled feedback layer
is what a decoder replaces.

Detector rates the model predicts are compared against rates sampled from the
circuit simulator by the tests, not by this module. The comparison shares the
injection helper with construction, so it does not independently re-derive the
signatures; what it tests is that mechanisms compose by XOR in the simulator and
that merging identical signatures yields the right marginals.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from numbers import Integral, Real

import torch

_DETECTOR_PREFIX = "D"
_OBSERVABLE_PREFIX = "L"


def _probability(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real probability")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return normalized


def _count(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(value)


def _normalized_indices(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    indices: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must contain integer indices")
        index = int(value)
        if index < 0:
            raise ValueError(f"{name} must contain non-negative indices")
        indices.append(index)
    return tuple(sorted(set(indices)))


@dataclass(frozen=True)
class DemError:
    """One independent error mechanism and the signature it flips."""

    probability: float
    detectors: tuple[int, ...] = ()
    observables: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "probability",
            _probability(self.probability, name="error probability"),
        )
        object.__setattr__(
            self,
            "detectors",
            _normalized_indices(self.detectors, name="error detectors"),
        )
        object.__setattr__(
            self,
            "observables",
            _normalized_indices(self.observables, name="error observables"),
        )
        if not self.detectors and not self.observables:
            raise ValueError(
                "an error mechanism must flip at least one detector or observable"
            )


@dataclass(frozen=True)
class DetectorErrorModel:
    """A detector error model over a fixed number of detectors and observables."""

    num_detectors: int
    num_observables: int
    errors: tuple[DemError, ...] = ()

    def __post_init__(self) -> None:
        detectors = _count(self.num_detectors, name="num_detectors")
        observables = _count(self.num_observables, name="num_observables")
        if detectors < 1:
            raise ValueError("a detector error model requires at least one detector")
        object.__setattr__(self, "num_detectors", detectors)
        object.__setattr__(self, "num_observables", observables)

        for error in self.errors:
            if not isinstance(error, DemError):
                raise TypeError("errors must contain DemError records")
        object.__setattr__(
            self,
            "errors",
            tuple(
                sorted(
                    self.errors,
                    key=lambda error: (
                        error.detectors,
                        error.observables,
                        error.probability,
                    ),
                )
            ),
        )
        for error in self.errors:
            for index in error.detectors:
                if index >= detectors:
                    raise ValueError(
                        f"detector index {index} is outside the model shape"
                    )
            for index in error.observables:
                if index >= observables:
                    raise ValueError(
                        f"observable index {index} is outside the model shape"
                    )

    def detector_error_matrix(self) -> torch.Tensor:
        """Return the ``(num_detectors, num_errors)`` parity matrix.

        Entry ``[d, e]`` is one when error ``e`` flips detector ``d``. The
        orientation matches the stim ecosystem's detector error matrix.
        """

        matrix = torch.zeros((self.num_detectors, self.num_errors), dtype=torch.int8)
        for column, error in enumerate(self.errors):
            for index in error.detectors:
                matrix[index, column] = 1
        return matrix

    def observables_flips_matrix(self) -> torch.Tensor:
        """Return the ``(num_observables, num_errors)`` parity matrix.

        Entry ``[o, e]`` is one when error ``e`` flips observable ``o``.
        """

        matrix = torch.zeros((self.num_observables, self.num_errors), dtype=torch.int8)
        for column, error in enumerate(self.errors):
            for index in error.observables:
                matrix[index, column] = 1
        return matrix

    def _marginal_rates(
        self, count: int, select: Callable[[DemError], tuple[int, ...]]
    ) -> torch.Tensor:
        rates = torch.zeros((count,), dtype=torch.float64)
        if count == 0:
            # Reachable only for ``num_observables == 0``; detectors are never zero.
            return rates
        complements = torch.ones((count,), dtype=torch.float64)
        for error in self.errors:
            factor = 1.0 - 2.0 * error.probability
            for index in select(error):
                complements[index] *= factor
        return (1.0 - complements) / 2.0

    def detector_rates(self) -> torch.Tensor:
        """Return the exact marginal flip probability of every detector."""

        return self._marginal_rates(self.num_detectors, lambda error: error.detectors)

    def observable_rates(self) -> torch.Tensor:
        """Return the exact marginal flip probability of every observable."""

        return self._marginal_rates(
            self.num_observables, lambda error: error.observables
        )

    @property
    def num_errors(self) -> int:
        """Number of independent error mechanisms in the model."""

        return len(self.errors)


__all__ = ("DemError", "DetectorErrorModel")
