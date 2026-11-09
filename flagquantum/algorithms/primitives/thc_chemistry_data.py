# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PrepareTHC.from_hamiltonian_coeffs.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/thc/prepare.py

"""THC alias rows and their named lookup in contiguous address order."""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike

from ...compiler._component_trace.resource_model import ComponentCost
from ...compiler.registers.layout.partition import RegisterLayout
from ...compiler.registers.lookup.qroam_bank_binding import (
    bind_clean_banks,
    clean_bank_output_layout,
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


@dataclass(frozen=True)
class THCChemistryData:
    """Locked alias columns and a permutation for the executable lookup."""

    num_mu: int
    num_spin_orb: int
    probability_bits: int
    source_rows: tuple[tuple[int, int], ...]
    source_datasets: tuple[tuple[int, ...], ...]
    normalization: float
    log_block_size: int

    @property
    def address_width(self) -> int:
        return (len(self.source_rows) - 1).bit_length()

    @property
    def target_layout(self) -> RegisterLayout:
        width = self.num_mu.bit_length()
        return (
            ("theta", 1),
            ("alt_theta", 1),
            ("alt_mu", width),
            ("alt_nu", width),
            ("keep", self.probability_bits),
        )

    @property
    def lookup_output_layout(self) -> RegisterLayout:
        return clean_bank_output_layout(
            self.target_layout, 1 << self.log_block_size, junk_prefix="junk_"
        )

    @cached_property
    def address_datasets(self) -> tuple[tuple[int, ...], ...]:
        """Reorder whole rows after alias preprocessing, keeping coordinates intact."""
        source_indices = [0] * len(self.source_rows)
        for index, (mu, nu) in enumerate(self.source_rows):
            address = nu * (nu + 1) // 2 + mu
            source_indices[address] = index
        return tuple(
            tuple(column[index] for index in source_indices)
            for column in self.source_datasets
        )

    def lookup_table(self, *, source_order: bool = False) -> QroamCleanTable:
        """Return the table for contiguous addresses or the source declaration."""
        columns = self.source_datasets if source_order else self.address_datasets
        return QroamCleanTable(
            (len(self.source_rows),), (self.log_block_size,), columns
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
        """Load clean banks or coherently uncompute matching signed outputs."""
        if len(address) != self.address_width:
            raise ValueError("THC lookup address width must match the row count")
        banks = bind_clean_banks(
            outputs, self.target_layout, 1 << self.log_block_size, junk_prefix="junk_"
        )
        emit = source_qroam_clean_uncompute_ir if uncompute else source_qroam_clean_ir
        return emit(
            self.lookup_table(), (), (address,), banks, workspace, classical_bits
        )

    def lookup_cost(self, *, source_order: bool = False) -> ComponentCost:
        """Model the concrete lookup with the requested row order."""
        return source_qroam_clean_cost(
            self.lookup_table(source_order=source_order),
            0,
            (self.address_width,),
            tuple(width for _, width in self.target_layout),
        )


def thc_chemistry_data(
    eigenvalues: ArrayLike,
    eta: ArrayLike,
    zeta: ArrayLike,
    *,
    probability_bits: int = 8,
    log_block_size: int | None = None,
) -> THCChemistryData:
    """Construct the original alias columns and normalization from THC tensors."""
    one, leaves, central = np.asarray(eigenvalues), np.asarray(eta), np.asarray(zeta)
    if one.ndim != 1 or not one.size:
        raise ValueError("THC requires a nonempty vector of one-body eigenvalues")
    if (
        central.ndim != 2
        or not central.shape[0]
        or central.shape[0] != central.shape[1]
    ):
        raise ValueError("THC requires a nonempty square central tensor")
    num_mu = central.shape[0]
    spatial = one.size
    width = num_mu.bit_length()
    if leaves.shape != (num_mu, spatial) or spatial > 1 << width:
        raise ValueError(
            "THC leaf shape or spatial orbital count does not match the addresses"
        )
    if any(np.iscomplexobj(array) for array in (one, leaves, central)):
        raise ValueError("THC tensors must be real and finite")
    one = np.asarray(one, dtype=np.float64)
    leaves = np.asarray(leaves, dtype=np.float64)
    central = np.asarray(central, dtype=np.float64)
    if any(not np.isfinite(array).all() for array in (one, leaves, central)):
        raise ValueError("THC tensors must be real and finite")

    upper = np.triu_indices(num_mu)
    rows = (
        *((int(mu), int(nu)) for mu, nu in zip(*upper, strict=True)),
        *((mu, num_mu) for mu in range(spatial)),
    )
    weights = (*map(float, central[upper]), *map(float, one))
    alternates, keep = alias_numerators(
        tuple(abs(value) for value in weights), mu=probability_bits
    )
    theta = tuple(int(value < 0) for value in weights)
    columns = (
        theta,
        tuple(theta[row] for row in alternates),
        tuple(rows[row][0] for row in alternates),
        tuple(rows[row][1] for row in alternates),
        keep,
    )
    norm_factors = np.diag(np.diag(leaves.dot(leaves.T)))
    normalized = norm_factors.dot(central).dot(norm_factors)
    normalization = float(np.sum(np.abs(one)) + 0.5 * np.sum(np.abs(normalized)))
    if not isfinite(normalization):
        raise ValueError("THC normalization must be finite")
    if log_block_size is None:
        log_block_size = optimal_qroam_clean_log_block_size(
            len(rows), 2 + 2 * width + probability_bits
        )
    if (
        type(log_block_size) is not int
        or not 0 <= log_block_size < len(rows).bit_length()
    ):
        raise ValueError("THC block exponent must fit the alias address domain")
    return THCChemistryData(
        num_mu,
        2 * spatial,
        probability_bits,
        rows,
        columns,
        normalization,
        log_block_size,
    )
