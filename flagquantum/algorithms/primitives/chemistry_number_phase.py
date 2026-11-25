# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ApplyControlledZs.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/black_boxes.py

"""Apply a controlled phase to the most significant system wire."""

from dataclasses import dataclass

from ...compiler._component_trace.resource_model import ComponentCost, GateCost
from ...compiler.registers.control.multi_control import multi_control_z_ir
from ...compiler.registers.primitives.controlled_z_word_component import (
    controlled_z_word_signature as _word_signature,
)
from ...core.ir.program.signatures import Signature
from ._block_binding import _BoundBlockOperation, _validate_block_work


@dataclass(frozen=True)
class ChemistryNumberPhase:
    """Keep the control pattern, system port, and source model together."""

    values: tuple[int, ...]
    system_width: int

    def __post_init__(self) -> None:
        if type(self.system_width) is not int or self.system_width < 1:
            raise ValueError("Number phase requires a nonempty concrete system")
        if not self.values or any(
            type(value) is not int or value not in (0, 1) for value in self.values
        ):
            raise ValueError("Number phase requires binary control values")

    @property
    def signature(self) -> Signature:
        return _word_signature(self.system_width, len(self.values))

    def resource_model(self, name: str = "ApplyControlledZs") -> ComponentCost:
        """Use the locked Toffoli declaration and the predicate's logical peak."""
        controls = len(self.values)
        return ComponentCost(
            name,
            self.system_width + 2 * controls - 1,
            GateCost(toffoli=controls - 1),
        )

    def make_on(
        self,
        controls: tuple[int, ...],
        system: tuple[int, ...],
        workspace: tuple[int, ...] = (),
        classical_bits: tuple[int, ...] = (),
    ) -> _BoundBlockOperation:
        """Bind the through words and append the existing measured Z ladder."""
        if len(system) != self.system_width:
            raise ValueError("Number phase system binding must match its width")
        through = (*controls, *system)
        _validate_block_work(through, workspace, classical_bits)
        instructions = multi_control_z_ir(
            controls, self.values, system[0], workspace, classical_bits
        )
        return _BoundBlockOperation(instructions, max((*through, *workspace)) + 1)
