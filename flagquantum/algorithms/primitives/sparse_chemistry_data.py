# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 chemistry/sparse/prepare.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/sparse/prepare.py

"""Ordered integral representatives and lookup data for sparse chemistry PREPARE."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import combinations
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike

from ...compiler._component_trace.resource_model import ComponentCost
from ...compiler.registers.layout.partition import RegisterLayout
from ...compiler.registers.lookup.qroam_bank_binding import (
    bind_clean_banks,
    clean_bank_output_layout,
)
from ...compiler.registers.lookup.qroam_clean_erasure import (
    qroam_clean_measurement_erase_ir,
)
from ...compiler.registers.lookup.qroam_resource_model import source_qroam_clean_cost
from ...compiler.registers.lookup.source_qroam_clean import (
    QroamCleanTable,
    optimal_qroam_clean_log_block_size,
    source_qroam_clean_ir,
    source_qroam_clean_uncompute_ir,
)
from ...core.ir import Instruction
from .alias_sampling import alias_numerators

OrbitalIndices = tuple[int, int, int, int]


@dataclass(frozen=True)
class SparseChemistryData:
    """Source-ordered coefficient rows and the 13 named output datasets."""

    spatial_orbitals: int
    probability_bits: int
    rotation_bits: int
    indices: tuple[OrbitalIndices, ...]
    coefficients: tuple[float, ...]
    datasets: tuple[tuple[int, ...], ...]
    normalization: float
    log_block_size: int

    @property
    def address_width(self) -> int:
        return (len(self.coefficients) - 1).bit_length()

    @property
    def target_layout(self) -> RegisterLayout:
        orbital_bits = (self.spatial_orbitals - 1).bit_length()
        return (
            ("p", orbital_bits),
            ("q", orbital_bits),
            ("r", orbital_bits),
            ("s", orbital_bits),
            ("theta", 1),
            ("flag_1b", 1),
            ("alt_p", orbital_bits),
            ("alt_q", orbital_bits),
            ("alt_r", orbital_bits),
            ("alt_s", orbital_bits),
            ("alt_theta", 1),
            ("alt_flag_1b", 1),
            ("keep", self.probability_bits),
        )

    def lookup_table(self) -> QroamCleanTable:
        """Create the concrete clean lookup with all output banks retained."""
        if self.spatial_orbitals < 2:
            raise ValueError("Sparse chemistry lookup requires nonempty orbital words")
        return QroamCleanTable(
            (len(self.coefficients),), (self.log_block_size,), self.datasets
        )

    @property
    def lookup_output_layout(self) -> RegisterLayout:
        """List selected words followed by dataset-major clean bank outputs."""
        return clean_bank_output_layout(
            self.target_layout, 1 << self.log_block_size, junk_prefix="jnk_"
        )

    def lookup_ir(
        self,
        address: tuple[int, ...],
        outputs: Mapping[str, tuple[int, ...]],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        *,
        uncompute: bool = False,
    ) -> tuple[Instruction, ...]:
        """Load clean banks or coherently erase matching signed lookup outputs."""
        table = self.lookup_table()
        banks = self._bind_lookup_banks(address, outputs)
        emit = source_qroam_clean_uncompute_ir if uncompute else source_qroam_clean_ir
        return emit(table, (), (address,), banks, workspace, classical_bits)

    def lookup_measurement_erase_ir(
        self,
        address: tuple[int, ...],
        outputs: Mapping[str, tuple[int, ...]],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        bank_bits: tuple[int, ...],
    ) -> tuple[Instruction, ...]:
        """Measure and clear the matching lookup outputs in their declared order."""
        return qroam_clean_measurement_erase_ir(
            self.lookup_table(),
            (),
            (address,),
            self._bind_lookup_banks(address, outputs),
            workspace,
            classical_bits,
            bank_bits,
        )

    def _bind_lookup_banks(
        self,
        address: tuple[int, ...],
        outputs: Mapping[str, tuple[int, ...]],
    ) -> tuple[tuple[tuple[int, ...], ...], ...]:
        if len(address) != self.address_width:
            raise ValueError("Sparse chemistry address width must match the row count")
        return bind_clean_banks(
            outputs, self.target_layout, 1 << self.log_block_size, junk_prefix="jnk_"
        )

    def lookup_cost(self) -> ComponentCost:
        """Estimate the source forward lookup with the declared target widths."""
        return source_qroam_clean_cost(
            self.lookup_table(),
            0,
            (self.address_width,),
            tuple(width for _, width in self.target_layout),
        )


def _two_body_representatives(spatial: int) -> Iterator[OrbitalIndices]:
    sites = range(spatial)
    for p, q, r, s in combinations(sites, 4):
        yield from ((p, q, r, s), (p, r, q, s), (p, s, r, q))
    for p, q, r in combinations(sites, 3):
        yield from (
            (p, p, q, r),
            (p, q, p, r),
            (q, q, p, r),
            (q, r, p, q),
            (r, r, p, q),
            (r, p, q, r),
        )
    for p, q in combinations(sites, 2):
        yield from ((p, p, q, q), (p, q, p, q), (p, p, p, q), (q, q, q, p))
    for p in sites:
        yield p, p, p, p


def sparse_chemistry_data(
    one_body: ArrayLike,
    two_body: ArrayLike,
    *,
    probability_bits: int = 8,
    rotation_bits: int = 8,
    drop_threshold: float = 0,
    log_block_size: int | None = None,
) -> SparseChemistryData:
    """Prepare lookup data from modified one-body and chemists' ERI tensors."""
    one, two = np.asarray(one_body), np.asarray(two_body)
    if one.ndim != 2 or not one.shape[0] or one.shape[0] != one.shape[1]:
        raise ValueError("One-body integrals must be a nonempty square matrix")
    spatial = one.shape[0]
    if two.shape != (spatial,) * 4:
        raise ValueError("Two-body integrals must have four matching spatial axes")
    if np.iscomplexobj(one) or np.iscomplexobj(two):
        raise ValueError("Sparse chemistry requires finite real integrals")
    one, two = np.asarray(one, dtype=np.float64), np.asarray(two, dtype=np.float64)
    if not np.isfinite(one).all() or not np.isfinite(two).all():
        raise ValueError("Sparse chemistry requires finite real integrals")
    if not isfinite(drop_threshold) or drop_threshold < 0:
        raise ValueError("Drop threshold must be nonnegative and finite")
    if any(
        type(bits) is not int or bits < 1 for bits in (probability_bits, rotation_bits)
    ):
        raise ValueError(
            "State preparation and rotation precision must be positive integers"
        )

    lower = tuple((p, q, 0, 0) for p in range(spatial) for q in range(p + 1))
    retained = tuple(
        row
        for row in _two_body_representatives(spatial)
        if abs(float(two[row])) > drop_threshold
    )
    indices = (*lower, *retained)
    coefficients = (
        *(float(one[p, q]) for p, q, _, _ in lower),
        *(float(two[row]) for row in retained),
    )
    alt, keep = alias_numerators(
        tuple(abs(value) for value in coefficients), mu=probability_bits
    )
    theta = tuple(int(value < 0) for value in coefficients)
    body_flag = (0,) * len(lower) + (1,) * len(retained)
    base = (*zip(*indices, strict=True), theta, body_flag)
    datasets = (*base, *(tuple(column[i] for i in alt) for column in base), keep)
    if log_block_size is None:
        target_bits = 8 * (spatial - 1).bit_length() + 4 + probability_bits
        log_block_size = optimal_qroam_clean_log_block_size(len(indices), target_bits)
    if (
        type(log_block_size) is not int
        or not 0 <= log_block_size <= len(indices).bit_length() - 1
    ):
        raise ValueError("Block exponent must fit the integral address domain")
    normalization = float(np.sum(np.abs(one)) + 0.5 * np.sum(np.abs(two)))
    if not isfinite(normalization):
        raise ValueError("Sparse chemistry normalization must be finite")
    return SparseChemistryData(
        spatial,
        probability_bits,
        rotation_bits,
        indices,
        coefficients,
        datasets,
        normalization,
        log_block_size,
    )
