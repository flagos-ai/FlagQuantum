# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 LinearCombination.rescaled_lambd and prepare.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/linear_combination.py

"""Prepare source-rescaled LCU coefficients using shared alias components."""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from ...compiler._component_trace.resource_model import ComponentCost
from ._block_binding import _BoundBlockOperation
from .alias_preparation import (
    alias_black_box_prepare,
    alias_preparation_requirements,
    source_dense_alias_preparation_cost,
)
from .alias_sampling import AliasSamplingTable, alias_numerators
from .black_box_prepare import BlackBoxPrepare


def _rescaled_lcu_coefficients(
    coefficients: tuple[float, ...], normalizations: tuple[float, ...]
) -> tuple[float, ...]:
    """Preserve the original NumPy normalization and trailing zero padding."""
    weights = np.abs(np.array(coefficients) * np.array(normalizations))
    total = np.linalg.norm(weights, ord=1)
    if not np.isfinite(total):
        raise ValueError("Rescaled LCU total weight must be finite")
    weights /= total
    weights.resize(1 << (len(weights) - 1).bit_length(), refcheck=False)
    return tuple(float(x) for x in weights)


@dataclass(frozen=True)
class LinearCombinationPreparation:
    """Concrete coefficient preparation, with logical junk distinct from scratch.

    Coefficients are real and normalizations are the constituent blocks' alpha
    values. Signs belong to SELECT, not to this magnitude preparation. Wires
    are big-endian; packed junk follows the shared alias PREPARE layout.
    """

    coefficients: tuple[float, ...]
    normalizations: tuple[float, ...]
    coefficient_bits: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficients", tuple(self.coefficients))
        object.__setattr__(self, "normalizations", tuple(self.normalizations))
        if len(self.coefficients) < 2 or len(self.coefficients) != len(
            self.normalizations
        ):
            raise ValueError(
                "LCU needs at least two paired coefficients and normalizations"
            )
        if type(self.coefficient_bits) is not int or self.coefficient_bits < 1:
            raise ValueError("LCU coefficient precision must be a positive integer")
        if any(not isfinite(x) for x in (*self.coefficients, *self.normalizations)):
            raise ValueError("LCU preparation requires finite concrete parameters")
        if any(x < 0 for x in self.normalizations):
            raise ValueError("Block normalizations must be nonnegative")
        if not any(
            c * a for c, a in zip(self.coefficients, self.normalizations, strict=True)
        ):
            raise ValueError(
                "Rescaled LCU coefficients must have positive total weight"
            )

    @property
    def rescaled_coefficients(self) -> tuple[float, ...]:
        """Keep the original normalization independently of bit precision."""
        return _rescaled_lcu_coefficients(self.coefficients, self.normalizations)

    @property
    def table(self) -> AliasSamplingTable:
        weights = self.rescaled_coefficients
        alternate, keep = alias_numerators(weights, mu=self.coefficient_bits)
        return AliasSamplingTable(
            len(weights), alternate, keep, self.coefficient_bits, 1.0
        )

    @property
    def prepare(self) -> BlackBoxPrepare:
        return alias_black_box_prepare(self.table)

    @property
    def workspace_requirements(self) -> tuple[int, int]:
        """Return clean quantum scratch and classical slots, excluding junk."""
        return alias_preparation_requirements(self.table)

    def resource_model(self, *, legacy_shims: bool = False) -> ComponentCost:
        """Return the forward alias call graph, independently of emitted IR."""
        return source_dense_alias_preparation_cost(
            self.table, legacy_shims=legacy_shims
        )

    def make_on(
        self,
        *,
        selection: tuple[int, ...],
        junk: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        adjoint: bool = False,
    ) -> _BoundBlockOperation:
        """Bind PREPARE or its source adjoint for circuit.append(op).

        Forward logical junk is a meaningful output; only workspace must return
        to zero. Applying the adjoint to the prepared state restores selection
        and junk too. No SELECT sign phase is introduced by this component.
        """
        instructions = self.prepare.emit(
            selection, junk, workspace, classical_bits, adjoint=adjoint
        )
        return _BoundBlockOperation(
            instructions, max((*selection, *junk, *workspace), default=-1) + 1
        )
