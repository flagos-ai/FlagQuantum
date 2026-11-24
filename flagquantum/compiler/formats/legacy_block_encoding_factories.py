# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 block_encoding/lcu_block_encoding.py.

"""Resolve block encodings with native oracle roles and source control adders."""

from collections.abc import Callable, Mapping
from typing import Literal

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...errors import SerializationError
from ..registers.component_control import ControlAdder
from ..registers.preparation.block_encoding_component import (
    lcu_block_component_program,
    lcu_block_signature,
    select_block_component_program,
    select_block_signature,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_packed_oracle_factories import OracleContext
from .legacy_word_factories import _rename_program

ExactPair = tuple[ProgramRecord, ProgramRecord]
ControlledSelect = Callable[
    [FunctionRecord, int], tuple[ControlAdder, tuple[ExactPair, ...]]
]
_Recipe = tuple[
    Literal["select", "lcu"],
    Signature,
    ExactPair,
    ExactPair,
    tuple[ControlAdder, tuple[ExactPair, ...]] | None,
    bool,
]


def block_encoding_factories(
    *,
    prepare_context: OracleContext,
    select_context: OracleContext,
    controlled_select: ControlledSelect | None = None,
    adjoint_factory: ComponentFactory | None = None,
) -> dict[str, ComponentFactory]:
    """Require original native roles, exact inverses and specialized control calls.

    SELECT blocks execute only SELECT; PREPARE remains signal metadata. LCU
    blocks conjugate SELECT with its exact PREPARE pair. Controlled adders
    preserve the source call sequence rather than substituting an opaque gate.
    Specialized-adjoint enum wire failures retain their independent scope.
    """
    recipes: dict[FunctionRecord, _Recipe] = {}

    def make(
        name: str, registers: tuple[Register, ...] | None, recipe: _Recipe
    ) -> ProgramRecord:
        kind, signature, preparation, selection, control, inverse = recipe
        if kind == "select":
            record = select_block_component_program(
                signature, selection[0], adjoint=inverse, inverse=selection[1]
            )
        else:
            adder, extra_pairs = control if control is not None else (None, ())
            record = lcu_block_component_program(
                signature,
                preparation[0],
                preparation[1],
                selection[0],
                controlled_select=adder,
                adjoint=inverse,
                inverse_pairs=(selection, *extra_pairs),
            )
        if registers is not None and registers != record.functions[0].registers:
            raise SerializationError("Invalid block-encoding register signature")
        record = _rename_program(record, name)
        recipes[record.functions[0]] = recipe
        return record

    def forward(
        kind: Literal["select", "lcu"],
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        allowed = {"select", "prepare", *(("control_val",) if kind == "lcu" else ())}
        if not {"select", "prepare"} <= arguments.keys() or arguments.keys() - allowed:
            raise SerializationError("Block encoding requires SELECT and PREPARE")
        select, prepare = arguments["select"], arguments["prepare"]
        if not isinstance(select, FunctionRecord) or not isinstance(
            prepare, FunctionRecord
        ):
            raise SerializationError(
                "Block encoding requires resolved native oracle functions"
            )
        selection_regs, targets, select_pair = select_context(select)
        prepare_regs, junk, prepare_pair = prepare_context(prepare)
        control = None
        if kind == "select":
            signature = select_block_signature(selection_regs, targets)
        else:
            value = arguments.get("control_val")
            if value is not None and (type(value) is not int or value not in (0, 1)):
                raise SerializationError(
                    "LCU control value must be zero, one or absent"
                )
            if value is not None:
                if controlled_select is None:
                    raise SerializationError("Missing exact SELECT control-adder owner")
                control = controlled_select(select, value)
            signature = lcu_block_signature(
                prepare_regs, junk, targets, controlled=value is not None
            )
        return make(
            name,
            registers,
            (kind, signature, prepare_pair, select_pair, control, False),
        )

    def select(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        return forward("select", name, registers, arguments)

    def lcu(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        return forward("lcu", name, registers, arguments)

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
            kind, signature, prep, select, control, inverse = recipes[child]
            return make(
                name,
                registers,
                (kind, signature.adjoint(), prep, select, control, not inverse),
            )
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError("Missing exact block-encoding adjoint context")

    return {
        "qualtran.bloqs.block_encoding.lcu_block_encoding.SelectBlockEncoding": select,
        "qualtran.bloqs.block_encoding.lcu_block_encoding.LCUBlockEncoding": lcu,
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
