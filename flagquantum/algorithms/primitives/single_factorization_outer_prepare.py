"""Outer single-factorization PREPARE from arXiv:2011.03494, Eq. B8."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

from .factor_alias_preparation import FactorAliasPreparation


@dataclass(frozen=True)
class PrepareSingleFactorizationOuter(FactorAliasPreparation):
    """Sample the factor and carry its nonzero flag through the alias swap."""

    _track_nonzero: ClassVar[bool] = True

    @classmethod
    def from_coefficients(
        cls,
        one_body: Sequence[float],
        factors: Sequence[Sequence[float]],
        *,
        probability_bits: int = 8,
        log_block_size: int = 0,
    ) -> "PrepareSingleFactorizationOuter":
        """Build the B8 marginal from triangular coefficients of equal length."""
        if len(one_body) == 0 or any(len(row) != len(one_body) for row in factors):
            raise ValueError(
                "SF coefficients require matching nonempty triangular rows"
            )
        if any(not isfinite(value) for row in (one_body, *factors) for value in row):
            raise ValueError("SF coefficients must be finite and real")
        weights = (
            2 * sum(map(abs, one_body)),
            *(sum(map(abs, row)) ** 2 for row in factors),
        )
        return cls.from_weights(
            weights, probability_bits=probability_bits, log_block_size=log_block_size
        )
