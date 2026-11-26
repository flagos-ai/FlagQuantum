# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal phase gates follow locked Qualtran v0.7.0 ComplexPhaseOracle.

"""Translate literal signed phase gates and reuse the bounded native ArcTan owner."""

from collections.abc import Callable

from ...compiler.formats.legacy_complex_phase_factories import (
    ComplexPhaseContext,
    complex_phase_factories,
)
from ...compiler.formats.legacy_component_resolution import ComponentFactory
from ...compiler.formats.legacy_unitary_block_factories import UnitaryPairOwner
from ...compiler.registers.lookup.arctan_lookup_component import arctan_lookup_program
from ...core.ir.program import FunctionRecord, ProgramRecord
from ...errors import SerializationError
from ._component_library import _gate_program
from ._gate_metadata import cirq_gate_signature
from .conversion import _cirq


def complex_phase_library_factories(
    *,
    encoder_owner: UnitaryPairOwner,
    target_names: Callable[[FunctionRecord], tuple[str, ...]],
) -> dict[str, ComponentFactory]:
    """Select native lookup explicitly; preserve source SDK phase constructors."""

    def context(encoder: FunctionRecord, precision: int) -> ComplexPhaseContext:
        names = target_names(encoder)
        if encoder.registers is None or not set(names) <= {
            r.name for r in encoder.registers
        }:
            raise SerializationError("Unknown complex phase encoder target ports")
        if any(r.shape_symbolic for r in encoder.registers if r.name in names):
            raise SerializationError(
                "Source complex phase allocation requires scalar target words"
            )
        width = sum(int(r.total_bits()) for r in encoder.registers if r.name in names)
        lookup = arctan_lookup_program(width, precision)
        cirq = _cirq()
        phases = []
        for index in range(precision):
            for sign, value in ((0, 1), (1, -1)):
                gate = (cirq.Z ** (value / 2 ** (1 + index))).controlled(
                    control_values=[sign]
                )
                record = _gate_program(
                    gate, cirq_gate_signature(gate), f"signed_phase_{index}_{sign}"
                )
                phases.append(
                    record
                    if isinstance(record, ProgramRecord)
                    else ProgramRecord((record,))
                )
        return ComplexPhaseContext(
            encoder_owner(encoder), (lookup, lookup), tuple(phases), names
        )

    return complex_phase_factories(context)
