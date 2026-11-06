"""Outer double-factorization PREPARE for Appendix C of arXiv:2011.03494."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Self

from .factor_alias_preparation import FactorAliasPreparation


@dataclass(frozen=True)
class PrepareDoubleFactorizationOuter(FactorAliasPreparation):
    """Sample factor zero for the one-body term and the remaining DF factors."""

    @classmethod
    def from_eigenvalues(
        cls,
        one_body: Sequence[float],
        factors: Sequence[Sequence[float]],
        *,
        probability_bits: int = 8,
        log_block_size: int = 0,
    ) -> Self:
        """Use the trace norm of the effective one-body matrix and Eq. C11."""
        spatial = len(one_body)
        if not spatial or any(not 1 <= len(row) <= spatial for row in factors):
            raise ValueError("DF eigenvalue rows must have a positive spatial rank")
        if any(not isfinite(value) for row in (one_body, *factors) for value in row):
            raise ValueError("DF eigenvalues must be finite and real")
        half_norms = (sum(abs(value) / 2 for value in row) for row in factors)
        weights = (sum(map(abs, one_body)), *(norm * norm for norm in half_norms))
        return cls.from_weights(
            weights, probability_bits=probability_bits, log_block_size=log_block_size
        )
