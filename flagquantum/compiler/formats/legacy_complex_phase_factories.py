# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adapted from locked Qualtran v0.7.0 ComplexPhaseOracle.

"""Resolve phase-oracle metadata using explicit exact encoder and leaf owners."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.phase.complex_phase_component import complex_phase_oracle_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_word_factories import _rename_program


@dataclass(frozen=True)
class ComplexPhaseContext:
    """Retain complete exact child definitions and source target ownership."""

    encoder_pair: tuple[ProgramRecord, ProgramRecord]
    arctan_pair: tuple[ProgramRecord, ProgramRecord]
    phases: tuple[ProgramRecord, ...]
    target_names: tuple[str, ...]


def complex_phase_factories(
    context_owner: Callable[[FunctionRecord, int], ComplexPhaseContext],
) -> dict[str, ComponentFactory]:
    """Keep leaf implementation choices explicit instead of inferring an inverse."""

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        encoder = arguments.get("encoder")
        if (
            not {"encoder"} <= arguments.keys()
            or arguments.keys() - {"encoder", "arctan_bitsize"}
            or not isinstance(encoder, FunctionRecord)
        ):
            raise SerializationError("Complex phase requires encoder and precision")
        precision = _integer({"arctan_bitsize": 32, **arguments}, "arctan_bitsize")
        if type(precision) is not int or precision < 0:
            raise SerializationError("Complex phase requires a concrete precision")
        context = context_owner(encoder, precision)
        if (
            not context.encoder_pair[0].functions
            or context.encoder_pair[0].functions[0].registers != encoder.registers
        ):
            raise SerializationError("Complex phase context must retain encoder ports")
        angle_ports = context.arctan_pair[0].functions[0].registers
        if (
            angle_ports is None
            or sum(int(r.total_bits()) for r in angle_ports if r.name == "target")
            != precision
        ):
            raise SerializationError("Complex phase context precision mismatch")
        record = complex_phase_oracle_program(
            *context.encoder_pair,
            *context.arctan_pair,
            context.phases,
            target_names=context.target_names,
        )
        if registers is not None and record.functions[0].registers != registers:
            raise SerializationError("Invalid complex phase register signature")
        return _rename_program(record, name)

    return {
        "qualtran.bloqs.mean_estimation.complex_phase_oracle.ComplexPhaseOracle": factory
    }
