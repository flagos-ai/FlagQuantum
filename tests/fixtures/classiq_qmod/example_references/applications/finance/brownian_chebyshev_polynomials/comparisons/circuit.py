"""Retained source provenance and independently implemented native Circuit."""

from examples.synthesis.applications.finance.brownian_chebyshev_polynomials.comparisons.circuit import (
    compare,
    reference_circuit,
)

__all__ = ("reference_circuit",)

if __name__ == "__main__":
    compare()
