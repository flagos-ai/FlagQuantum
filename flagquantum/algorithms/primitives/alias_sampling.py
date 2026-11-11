# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 lcu_util.py and the alias preparation factories.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/linalg/lcu_util.py

"""Concrete reversible alias tables for coefficient state preparation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

from .alias_discretization import alias_precision_bits, reversible_sampling_rows


@dataclass(frozen=True)
class AliasSamplingTable:
    """Integer alias probabilities and their logical selection domain."""

    selection_size: int
    alternates: tuple[int, ...]
    keep: tuple[int, ...]
    mu: int
    normalization: float
    indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        rows = len(self.alternates)
        if (
            type(self.selection_size) is not int
            or type(self.mu) is not int
            or self.mu < 1
            or not 1 <= rows <= self.selection_size
            or len(self.keep) != rows
            or not isfinite(self.normalization)
            or self.normalization <= 0
        ):
            raise ValueError("Alias table requires valid dimensions and normalization")
        if self.indices is None:
            if rows != self.selection_size:
                raise ValueError("Dense alias table requires one row per selection")
        elif (
            len(self.indices) != rows
            or len(set(self.indices)) != rows
            or any(
                type(i) is not int or not 0 <= i < self.selection_size
                for i in self.indices
            )
        ):
            raise ValueError("Sparse alias table requires distinct in-range indices")
        if any(
            type(i) is not int or not 0 <= i < self.selection_size
            for i in self.alternates
        ) or any(type(k) is not int or not 0 <= k < 1 << self.mu for k in self.keep):
            raise ValueError(
                "Alias table values must fit the selection and keep registers"
            )


def _normalization(weights: Sequence[float], precision: float) -> float:
    if not weights or any(not isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("Alias sampling requires finite nonnegative weights")
    total = sum(weights)
    if not isfinite(total) or total <= 0 or not isfinite(precision) or precision <= 0:
        raise ValueError("Alias sampling requires positive finite weight and precision")
    return total


def _table(
    weights: Sequence[float], epsilon: float
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    mu = alias_precision_bits(len(weights), epsilon)
    alternate, keep = alias_numerators(weights, mu=mu)
    return alternate, keep, mu


def alias_numerators(
    weights: Sequence[float], *, mu: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Build the source alternate and keep rows at a fixed bit precision."""
    _normalization(weights, 1.0)
    if type(mu) is not int or mu < 1:
        raise ValueError("Alias bit precision must be a positive integer")
    alternate, keep, _ = reversible_sampling_rows(weights, sub_bit_precision=mu)
    return tuple(alternate), tuple(keep)


def alias_sampling_table(
    weights: Sequence[float], *, precision: float = 1e-5
) -> AliasSamplingTable:
    """Build a dense table using the source absolute coefficient precision."""
    total = _normalization(weights, precision)
    alt, keep, mu = _table(weights, precision / total)
    return AliasSamplingTable(len(weights), alt, keep, mu, total)


def sparse_alias_sampling_table(
    weights: Mapping[int, float], selection_size: int, *, precision: float = 1e-5
) -> AliasSamplingTable:
    """Build the source sparse table in mapping order with normalized precision."""
    indices = tuple(weights)
    values = tuple(weights.values())
    total = _normalization(values, precision)
    alt, keep, mu = _table(values, precision)
    return AliasSamplingTable(
        selection_size, tuple(indices[i] for i in alt), keep, mu, total, indices
    )


def sparse_alias_sampling_table_from_dense(
    weights: Sequence[float],
    *,
    precision: float = 1e-5,
    nonzero_threshold: float = 1e-6,
) -> AliasSamplingTable:
    """Drop coefficients within the source absolute zero tolerance."""
    if not isfinite(nonzero_threshold) or nonzero_threshold < 0:
        raise ValueError("Alias sampling requires a nonnegative finite zero threshold")
    return sparse_alias_sampling_table(
        {
            i: weight
            for i, weight in enumerate(weights)
            if abs(weight) > nonzero_threshold or not isfinite(weight)
        },
        len(weights),
        precision=precision,
    )
