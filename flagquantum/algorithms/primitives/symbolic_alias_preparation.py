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
    ceil,
    log2,
    smax,
)
from ...compiler.registers.control_predicate_resource_model import (
    control_predicate_cost,
)
from ...compiler.registers.symbolic_resource_model import (
    SymbolicResourceFormula,
    source_qrom_resource_formula,
)
from ...compiler.registers.uniform_preparation import source_uniform_preparation_cost


def alias_probability_bits(rows: SymbolicInt, precision: SymbolicReal) -> SymbolicInt:
    """Return the source bit width at normalized precision."""
    if isinstance(rows, int) and rows < 1:
        raise ValueError("Alias preparation requires a positive row count")
    if isinstance(precision, (int, float)) and not 0 < precision < float("inf"):
        raise ValueError("Alias precision must be positive and finite")
    bits = ceil(log2(1 / (precision * rows)))
    return max(1, bits) if isinstance(bits, int) else bits


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
        cleanup = control_predicate_cost(
            (1, 1), uncompute=True, legacy_shims=legacy_shims
        )
        cleanup_delta = cleanup.gates.clifford - 1
        sparse = self.support_size is not None
        rows = self.rows if self.support_size is None else self.support_size
        selection, mu = self.selection_width, self.probability_bits
        sparse_width = self.sparse_index_width
        address = sparse_width if sparse else selection
        if isinstance(rows, int) and rows < 2:
            raise ValueError("The shape-only cost requires at least two rows")
        uniform_peak: SymbolicReal
        if isinstance(rows, int):
            uniform = source_uniform_preparation_cost(rows, legacy_shims=legacy_shims)
            uniform_ands = uniform.gates.and_bloq
            uniform_cliffords = uniform.gates.clifford
            uniform_rotations = uniform.gates.rotation
            uniform_peak = uniform.qubits
        else:
            uniform_ands = 3 * address - 1
            uniform_cliffords = 16 * address + 3 + cleanup_delta * uniform_ands
            uniform_rotations = 2
            uniform_peak = smax(3, address + 1, 2 * address - 1)

        targets = (selection, selection, mu) if sparse else (selection, mu)
        qrom = source_qrom_resource_formula(
            (rows,), targets, selection_widths=(address,), legacy_shims=legacy_shims
        )
        compare_ands = 2 * mu - 1 if isinstance(mu, int) else 2 * mu + 1
        compare_cliffords = (
            22 * mu - 9 if isinstance(mu, int) else 22 * mu + 11
        ) + cleanup_delta * compare_ands
        compare_peak = 5 * mu if isinstance(mu, int) else smax(7, 2 * mu + 1)
        ands = uniform_ands + qrom.and_bloq + compare_ands
        peak = cast(
            SymbolicInt,
            smax(
                0,
                selection + self.junk_width,
                uniform_peak + selection + self.junk_width - address,
                qrom.logical_qubits + mu + 1,
                compare_peak + 2 * selection + sparse_width,
                smax(3, 2 * selection + 1) + 2 * mu + sparse_width,
                smax(1, mu) + 2 * selection + sparse_width + mu + 1,
            ),
        )
        return SymbolicResourceFormula(
            ands,
            uniform_cliffords + qrom.clifford + compare_cliffords + mu,
            ands,
            peak,
            cswap=selection,
            rotation=uniform_rotations,
        )
