# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 StatePreparationViaRotations.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/state_preparation_via_rotation.py

"""Dense state preparation using programmable phase-gradient lookups."""

from dataclasses import dataclass
from math import pi

from ...compiler.registers.bitwise import _validate_wires
from ...compiler.registers.phase_gradient_lookup import (
    phase_gradient_lookup_ir,
    phase_gradient_lookup_requirements,
)
from ...core.ir import Instruction
from .rotation_tables import rotation_tables


@dataclass(frozen=True)
class RotationPreparation:
    """A concrete preparation with a caller-owned, negative-exponent gradient."""

    coefficients: tuple[complex, ...]
    phase_width: int

    def requirements(
        self, control_count: int, *, uncompute: bool = False
    ) -> tuple[int, int]:
        """Count clean work qubits and reusable measurement slots."""
        rows, phases = rotation_tables(
            self.coefficients, self.phase_width, uncompute=uncompute
        )
        return self._requirements(control_count, (*rows, phases))

    def _requirements(
        self, control_count: int, rows: tuple[tuple[int, ...], ...]
    ) -> tuple[int, int]:
        if len(rows) == 1:
            raise ValueError("Rotation preparation requires a nonempty target register")
        if type(control_count) is not int or control_count < 0:
            raise ValueError("Rotation preparation control count must be nonnegative")
        sizes = tuple(
            phase_gradient_lookup_requirements(
                control_count + 1, width, self.phase_width, words
            )
            for width, words in enumerate(rows)
        )
        work = max(max(size for size, _ in sizes[:-1]), 1 + sizes[-1][0])
        return work, max(slots for _, slots in sizes)

    def emit(
        self,
        controls: tuple[int, ...],
        target: tuple[int, ...],
        gradient: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        *,
        uncompute: bool = False,
    ) -> tuple[Instruction, ...]:
        """Lower the locked forward or uncompute sequence on big-endian ports."""
        rows, phases = rotation_tables(
            self.coefficients, self.phase_width, uncompute=uncompute
        )
        work, slots = self._requirements(len(controls), (*rows, phases))
        if len(target) != len(rows) or len(gradient) != self.phase_width:
            raise ValueError(
                "Rotation preparation ports must match the state and precision"
            )
        if len(workspace) != work or len(classical_bits) != slots:
            raise ValueError(
                "Rotation preparation requires its declared workspace and slots"
            )
        _validate_wires((*controls, *target, *gradient, *workspace))

        def lookup(
            control: int, selection: tuple[int, ...], words: tuple[int, ...]
        ) -> tuple[Instruction, ...]:
            size, bits = phase_gradient_lookup_requirements(
                len(controls) + 1, len(selection), self.phase_width, words
            )
            return phase_gradient_lookup_ir(
                (*controls, control),
                selection,
                gradient,
                words,
                workspace[:size],
                classical_bits[:bits],
            )

        flag = workspace[-1]
        phase_gates = (
            Instruction("x", (flag,)),
            *lookup(flag, target, phases),
            Instruction("x", (flag,)),
        )
        amplitude_gates = []
        levels = reversed(range(len(rows))) if uncompute else range(len(rows))
        for level in levels:
            qubit = target[level]
            amplitude_gates.append(Instruction("rx", (qubit,), {"theta": pi / 2}))
            amplitude_gates.extend(lookup(qubit, target[:level], rows[level]))
            amplitude_gates.append(Instruction("rx", (qubit,), {"theta": -pi / 2}))
        if uncompute:
            return (*phase_gates, *amplitude_gates)
        return (*amplitude_gates, *phase_gates)
