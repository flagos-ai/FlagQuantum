# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 black_box_prepare/select.py.

"""Resolve packed oracle roles through explicitly owned typed child metadata."""

from collections.abc import Callable, Mapping
from typing import Literal

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import opaque_register_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import ssum
from ...errors import SerializationError
from ..registers.component_builder import ComponentProgramBuilder
from ..registers.layout.packed_oracle_component import packed_oracle_component_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_word_factories import _rename_program

OracleContext = Callable[
    [FunctionRecord],
    tuple[
        tuple[Register, ...], tuple[Register, ...], tuple[ProgramRecord, ProgramRecord]
    ],
]
_Recipe = tuple[
    Signature,
    tuple[tuple[str, tuple[Register, ...]], ...],
    ProgramRecord,
    ProgramRecord,
    bool,
    bool,
]


def packed_oracle_factories(
    *,
    prepare_context: OracleContext | None = None,
    select_context: OracleContext | None = None,
    adjoint_factory: ComponentFactory | None = None,
) -> dict[str, ComponentFactory]:
    """Require exact selection/junk or selection/target roles and inverse bodies.

    Roles come from the child's native owner, never from widths, names or gate
    counts. Full child helper closures remain native Programs. A bodyless source
    child retains its declaration and physical refusal. Controlled SELECT's
    omitted control partition is not silently repaired.
    """
    recipes: dict[FunctionRecord, _Recipe] = {}

    def make(
        name: str, registers: tuple[Register, ...] | None, recipe: _Recipe
    ) -> ProgramRecord:
        signature, groups, child, inverse, direction, skip = recipe
        if registers is not None and tuple(signature) != registers:
            raise SerializationError("Invalid packed oracle register signature")
        if skip:
            builder = ComponentProgramBuilder(signature)
            record = builder.finalize(builder.inputs)
        else:
            record = packed_oracle_component_program(
                signature, groups, child, inverse=inverse, adjoint=direction
            )
        record = _rename_program(record, name)
        recipes[record.functions[0]] = recipe
        return record

    def forward(
        kind: Literal["prepare", "select"],
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if arguments.keys() != {kind} or not isinstance(
            arguments[kind], FunctionRecord
        ):
            raise SerializationError("Packed oracle requires its resolved child")
        owner = prepare_context if kind == "prepare" else select_context
        if owner is None:
            raise SerializationError("Missing native packed oracle role context")
        child = arguments[kind]
        assert isinstance(child, FunctionRecord)
        selection, auxiliary, pair = owner(child)
        selection_width = ssum(reg.total_bits() for reg in selection)
        auxiliary_width = ssum(reg.total_bits() for reg in auxiliary)
        if kind == "prepare":
            signature = Signature.build(selection=selection_width, junk=auxiliary_width)
            groups = tuple(
                (name, regs)
                for name, regs in (("selection", selection), ("junk", auxiliary))
                if any(reg.total_qubits() != 0 for reg in regs)
            )
            skip = selection_width == 0
        else:
            signature = Signature(
                (
                    Register("selection", opaque_register_type(selection_width)),
                    Register("system", opaque_register_type(auxiliary_width)),
                )
            )
            groups = (("selection", selection), ("system", auxiliary))
            skip = False
        return make(name, registers, (signature, groups, pair[0], pair[1], False, skip))

    def prepare(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        return forward("prepare", name, registers, arguments)

    def select(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        return forward("select", name, registers, arguments)

    def adjoint(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            arguments.keys() == {"subbloq"}
            and isinstance(child, FunctionRecord)
            and child in recipes
        ):
            signature, groups, body, inverse, direction, skip = recipes[child]
            return make(
                name,
                registers,
                (signature.adjoint(), groups, body, inverse, not direction, skip),
            )
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError("Missing exact packed oracle adjoint context")

    return {
        "qualtran.bloqs.state_preparation.black_box_prepare.BlackBoxPrepare": prepare,
        "qualtran.bloqs.multiplexers.black_box_select.BlackBoxSelect": select,
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
