# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor from locked Qualtran v0.7.0 hamiltonian_simulation_by_gqsp.py.

"""Bind Hamiltonian GQSP to explicit numerical and preparation ownership."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...errors import SerializationError
from ..registers.component_control import (
    ControlAdder,
    _stored_control_adder,
    quantum_control_signature,
)
from ..registers.phase.hamiltonian_gqsp_component import (
    hamiltonian_gqsp_component_program,
)
from .legacy_component_arguments import ControlSpecification
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_controlled_factories import controlled_via_and_factories
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


def controlled_hamiltonian_gqsp_factories(
    context_owner: Callable[
        [FunctionRecord, ResolvedArgument, ResolvedArgument], HamiltonianGQSPContext
    ],
    controlled_context_owner: Callable[
        [FunctionRecord, ResolvedArgument, ResolvedArgument], HamiltonianGQSPContext
    ],
) -> dict[str, ComponentFactory]:
    """Own actual Hamiltonian controls while Always preparations bypass them.

    The controlled context supplies the actual controlled GQSP and ordinary
    PREPARE pair. Signature checks never establish inverse or cleanup semantics;
    those remain the explicit child owners' responsibility.
    """
    key = "qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp.HamiltonianSimulationByGQSP"
    base = hamiltonian_gqsp_factories(context_owner)[key]
    recipes: dict[
        FunctionRecord, tuple[FunctionRecord, ResolvedArgument, ResolvedArgument]
    ] = {}

    def ordinary(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        result = base(name, registers, arguments)
        assert isinstance(result, ProgramRecord)
        walk = arguments["walk_operator"]
        assert isinstance(walk, FunctionRecord)
        recipes[result.functions[0]] = walk, arguments["t"], arguments["precision"]
        return result

    def positive(child: FunctionRecord) -> tuple[ProgramRecord, ControlAdder]:
        if child not in recipes or child.registers is None:
            raise SerializationError("Missing owned Hamiltonian control context")
        context = controlled_context_owner(*recipes[child])
        result = hamiltonian_gqsp_component_program(
            context.gqsp, *context.prepare_pair, junk_names=context.junk_names
        )
        expected = quantum_control_signature(
            Signature(child.registers), ((quantum_bit_type(), ()),)
        )
        if result.functions[0].registers != tuple(expected):
            raise SerializationError("Controlled Hamiltonian must preserve root ports")
        result = _rename_program(result, child.name + "_positive")
        return result, _stored_control_adder(result.functions[0])

    def controlled(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        child, spec = arguments.get("subbloq"), arguments.get("ctrl_spec")
        if (
            arguments.keys() != {"subbloq", "ctrl_spec"}
            or not isinstance(child, FunctionRecord)
            or not isinstance(spec, ControlSpecification)
            or len(spec.clauses) != 1
        ):
            raise SerializationError("Hamiltonian control requires one positive bit")
        dtype, values = spec.clauses[0]
        if dtype != quantum_bit_type() or values.shape or int(values) != 1:
            raise SerializationError("Hamiltonian control requires one positive bit")
        result = _rename_program(positive(child)[0], name)
        if registers is not None and registers != result.functions[0].registers:
            raise SerializationError("Invalid controlled Hamiltonian signature")
        return result

    return {
        key: ordinary,
        "qualtran._infra.controlled.Controlled": controlled,
        **controlled_via_and_factories(control_system=positive),
    }
