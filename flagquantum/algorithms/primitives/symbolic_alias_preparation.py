# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 StatePreparationAliasSampling.from_n_coeff.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/state_preparation_alias_sampling.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/linalg/lcu_util.py

"""Shape-only alias preparation with locked precision and resource rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from ...compiler._component_trace.symbolic_math import (
    SymbolicInt,
    SymbolicReal,
    bit_length,
    smax,
)
from ...compiler.registers.comparison.comparison_resource_model import (
    source_less_equal_resource_formula,
)
from ...compiler.registers.preparation.uniform_preparation_resource_model import (
    source_uniform_preparation_resource_formula,
)
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
    source_qrom_resource_formula,
)
from .alias_discretization import alias_precision_bits


def alias_probability_bits(rows: SymbolicInt, precision: SymbolicReal) -> SymbolicInt:
    """Return the source bit width at normalized precision."""
    if isinstance(rows, int) and rows < 1:
        raise ValueError("Alias preparation requires a positive row count")
    if isinstance(precision, (int, float)) and not 0 < precision < float("inf"):
        raise ValueError("Alias precision must be positive and finite")
    return alias_precision_bits(rows, precision)


@dataclass(frozen=True)
class AliasPreparationShape:
    """A table-free alias signature and model with symbolic dimensions."""

    rows: SymbolicInt
    normalization: SymbolicReal
    probability_bits: SymbolicInt
    support_size: SymbolicInt | None = None

    @classmethod
    def from_precision(
        cls,
        rows: SymbolicInt,
        normalization: SymbolicReal,
        precision: SymbolicReal,
    ) -> "AliasPreparationShape":
        """Bind the source dimensions using absolute coefficient precision."""
        if isinstance(normalization, (int, float)) and not 0 < normalization < float(
            "inf"
        ):
            raise ValueError("Alias normalization must be positive and finite")
        return cls(
            rows, normalization, alias_probability_bits(rows, precision / normalization)
        )

    @classmethod
    def from_sparse_precision(
        cls,
        selection_size: SymbolicInt,
        support_size: SymbolicInt,
        normalization: SymbolicReal,
        precision: SymbolicReal,
    ) -> "AliasPreparationShape":
        """Build the three-table source shape at normalized precision."""
        if isinstance(selection_size, int) and isinstance(support_size, int):
            if not 1 <= support_size <= selection_size:
                raise ValueError("Sparse support must fit the selection domain")
        if isinstance(support_size, int) and support_size < 1:
            raise ValueError("Sparse support must be positive")
        if isinstance(normalization, (int, float)) and not 0 < normalization < float(
            "inf"
        ):
            raise ValueError("Alias normalization must be positive and finite")
        return cls(
            selection_size,
            normalization,
            alias_probability_bits(selection_size, precision),
            support_size,
        )

    @property
    def selection_width(self) -> SymbolicInt:
        return bit_length(self.rows - 1)

    @property
    def sparse_index_width(self) -> SymbolicInt:
        return 0 if self.support_size is None else bit_length(self.support_size - 1)

    @property
    def junk_width(self) -> SymbolicInt:
        return (
            self.selection_width
            + self.sparse_index_width
            + 2 * self.probability_bits
            + 1
        )

    def resource_model(self, *, legacy_shims: bool = False) -> SymbolicResourceFormula:
        """Estimate the source call graph for at least two table-free rows."""
        sparse = self.support_size is not None
        rows = self.rows if self.support_size is None else self.support_size
        selection, mu = self.selection_width, self.probability_bits
        sparse_width = self.sparse_index_width
        address = sparse_width if sparse else selection
        if isinstance(rows, int) and rows < 2:
            raise ValueError("The shape-only cost requires at least two rows")
        uniform = source_uniform_preparation_resource_formula(
            rows, legacy_shims=legacy_shims
        )

        targets = (selection, selection, mu) if sparse else (selection, mu)
        qrom = source_qrom_resource_formula(
            (rows,), targets, selection_widths=(address,), legacy_shims=legacy_shims
        )
        comparison = source_less_equal_resource_formula(
            mu, mu, legacy_shims=legacy_shims
        )
        ands = uniform.and_bloq + qrom.and_bloq + comparison.and_bloq
        peak = cast(
            SymbolicInt,
            smax(
                0,
                selection + self.junk_width,
                uniform.logical_qubits + selection + self.junk_width - address,
                qrom.logical_qubits + mu + 1,
                comparison.logical_qubits + 2 * selection + sparse_width,
                smax(3, 2 * selection + 1) + 2 * mu + sparse_width,
                smax(1, mu) + 2 * selection + sparse_width + mu + 1,
            ),
        )
        return SymbolicResourceFormula(
            ands,
            uniform.clifford + qrom.clifford + comparison.clifford + mu,
            ands,
            peak,
            cswap=selection,
            rotation=uniform.rotation,
        )
