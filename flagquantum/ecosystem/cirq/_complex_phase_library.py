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
    on_pair: (
        Callable[[FunctionRecord, tuple[ProgramRecord, ProgramRecord]], None] | None
    ) = None,
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
        inverses = []
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
                if on_pair is not None:
                    inverse_gate = cirq.inverse(gate)
                    inverse = _gate_program(
                        inverse_gate,
                        cirq_gate_signature(inverse_gate),
                        f"signed_phase_inverse_{index}_{sign}",
                    )
                    inverses.append(
                        inverse
                        if isinstance(inverse, ProgramRecord)
                        else ProgramRecord((inverse,))
                    )
        return ComplexPhaseContext(
            encoder_owner(encoder),
            (lookup, lookup),
            tuple(phases),
            names,
            tuple(inverses) if on_pair is not None else None,
        )

    return complex_phase_factories(context, on_pair=on_pair)
