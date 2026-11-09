# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SparseStatePreparationViaRotations.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/sparse_state_preparation_via_rotations.py

"""Sparse rotation preparation by a low-word preparation and basis permutation."""

from dataclasses import dataclass
from math import hypot, isfinite

from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.layout.partition import split_register_pair
from ...compiler.registers.sorting.permutation import (
    permutation_ir,
    source_partial_permutation_cycles,
)
from ...core.ir import Instruction
from .rotation_preparation import RotationPreparation


@dataclass(frozen=True)
class SparseRotationPreparation:
    """Bind concrete sparse coefficients in their given index order."""

    indices: tuple[int, ...]
    coefficients: tuple[complex, ...]
    size: int
    phase_width: int

    def _components(
        self, target_width: int
    ) -> tuple[RotationPreparation, tuple[tuple[int, ...], ...], int]:
        count = len(self.indices)
        if count < 2 or len(self.coefficients) != count:
            raise ValueError(
                "Sparse rotation preparation requires at least two indexed coefficients"
            )
        if (
            type(target_width) is not int
            or target_width < 1
            or self.size > 1 << target_width
        ):
            raise ValueError("Sparse rotation target must fit the state size")
        cycles = source_partial_permutation_cycles(
            self.size, dict(enumerate(self.indices))
        )
        mapped = {
            index: cycle[(i + 1) % len(cycle)]
            for cycle in cycles
            for i, index in enumerate(cycle)
        }
        if any(mapped.get(i, i) != index for i, index in enumerate(self.indices)):
            raise ValueError("Source partial permutation loses an indexed transition")
        norm = hypot(*(abs(value) for value in self.coefficients))
        if not isfinite(norm) or not norm:
            raise ValueError(
                "Sparse rotation coefficients must have a finite nonzero norm"
            )
        low_width = (count - 1).bit_length()
        padded = tuple(value / norm for value in self.coefficients) + (0j,) * (
            (1 << low_width) - count
        )
        return RotationPreparation(padded, self.phase_width), cycles, low_width

    def requirements(self, target_width: int) -> tuple[int, int]:
        """Count clean work qubits and slots reusable between both stages."""
        dense, cycles, _ = self._components(target_width)
        work, slots = dense.requirements(0)
        if cycles:
            return max(work, target_width), max(slots, target_width - 1)
        return work, slots

    def emit(
        self,
        target: tuple[int, ...],
        gradient: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
    ) -> tuple[Instruction, ...]:
        """Prepare on the low target bits, then apply the source cycles."""
        width = len(target)
        dense, cycles, low_width = self._components(width)
        work, slots = dense.requirements(0)
        if len(workspace) != max(work, width if cycles else 0) or len(
            classical_bits
        ) != max(slots, width - 1 if cycles else 0):
            raise ValueError(
                "Sparse rotation preparation requires its declared workspace and slots"
            )
        if len(set(classical_bits)) != len(classical_bits) or any(
            type(bit) is not int or bit < 0 for bit in classical_bits
        ):
            raise ValueError(
                "Sparse rotation preparation requires distinct nonnegative slots"
            )
        _validate_wires((*target, *gradient, *workspace))
        _, low_bits = split_register_pair(target, width - low_width, low_width)
        gates = dense.emit(
            (), low_bits, gradient, workspace[:work], classical_bits[:slots]
        )
        if not cycles:
            return gates
        return (
            *gates,
            *permutation_ir(
                target,
                cycles,
                workspace[0],
                workspace[1:width],
                classical_bits[: width - 1],
                size=self.size,
            ),
        )
