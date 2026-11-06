# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 StatePreparationAliasSampling.from_n_coeff.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/state_preparation_alias_sampling.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/linalg/lcu_util.py

"""Shape-only alias preparation with locked precision and resource rules."""

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

    @property
    def selection_width(self) -> SymbolicInt:
        return bit_length(self.rows - 1)

    @property
    def junk_width(self) -> SymbolicInt:
        return self.selection_width + 2 * self.probability_bits + 1

    def resource_model(self) -> SymbolicResourceFormula:
        """Estimate the source call graph for at least two table-free rows."""
        rows, address, mu = self.rows, self.selection_width, self.probability_bits
        if isinstance(rows, int) and rows < 2:
            raise ValueError("The shape-only cost requires at least two rows")
        uniform_peak: SymbolicReal
        if isinstance(rows, int):
            uniform = source_uniform_preparation_cost(rows)
            uniform_ands = uniform.gates.and_bloq
            uniform_cliffords = uniform.gates.clifford
            uniform_rotations = uniform.gates.rotation
            uniform_peak = uniform.qubits
        else:
            uniform_ands = 3 * address - 1
            uniform_cliffords = 16 * address + 3
            uniform_rotations = 2
            uniform_peak = smax(3, address + 1, 2 * address - 1)

        qrom = source_qrom_resource_formula((rows,), (address, mu))
        compare_ands = 2 * mu - 1 if isinstance(mu, int) else 2 * mu + 1
        compare_cliffords = 22 * mu - 9 if isinstance(mu, int) else 22 * mu + 11
        compare_peak = 5 * mu if isinstance(mu, int) else smax(7, 2 * mu + 1)
        ands = uniform_ands + qrom.and_bloq + compare_ands
        peak = cast(
            SymbolicInt,
            smax(
                0,
                address + self.junk_width,
                uniform_peak + self.junk_width,
                qrom.logical_qubits + mu + 1,
                compare_peak + 2 * address,
                smax(3, 2 * address + 1) + 2 * mu,
                smax(1, mu) + 2 * address + mu + 1,
            ),
        )
        return SymbolicResourceFormula(
            ands,
            uniform_cliffords + qrom.clifford + compare_cliffords + mu,
            ands,
            peak,
            cswap=address,
            rotation=uniform_rotations,
        )
