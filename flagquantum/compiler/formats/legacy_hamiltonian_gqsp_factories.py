# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor from locked Qualtran v0.7.0 hamiltonian_simulation_by_gqsp.py.

"""Bind Hamiltonian GQSP to explicit numerical and preparation ownership."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.phase.hamiltonian_gqsp_component import (
    hamiltonian_gqsp_component_program,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_word_factories import _rename_program


@dataclass(frozen=True)
class HamiltonianGQSPContext:
    """Own the derived signal processor and the actual PREPARE inverse pair."""

    gqsp: ProgramRecord
    prepare_pair: tuple[ProgramRecord, ProgramRecord]
    junk_names: tuple[str, ...] = ()


def hamiltonian_gqsp_factories(
    context_owner: Callable[
        [FunctionRecord, ResolvedArgument, ResolvedArgument], HamiltonianGQSPContext
    ],
) -> dict[str, ComponentFactory]:
    """Keep coefficient/phase computation outside Compiler and require exact owners."""

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        walk = arguments.get("walk_operator")
        if arguments.keys() != {"walk_operator", "t", "precision"} or not isinstance(
            walk, FunctionRecord
        ):
            raise SerializationError(
                "Hamiltonian GQSP requires walk, time and precision context"
            )
        context = context_owner(walk, arguments["t"], arguments["precision"])
        if (
            walk.registers is None
            or not context.gqsp.functions
            or context.gqsp.functions[0].registers
            != (Register("signal", quantum_bit_type()), *walk.registers)
        ):
            raise SerializationError("Hamiltonian GQSP context must retain walk ports")
        record = hamiltonian_gqsp_component_program(
            context.gqsp, *context.prepare_pair, junk_names=context.junk_names
        )
        if registers is not None and record.functions[0].registers != registers:
            raise SerializationError("Invalid Hamiltonian GQSP register signature")
        return _rename_program(record, name)

    return {
        "qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp.HamiltonianSimulationByGQSP": factory
    }
