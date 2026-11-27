# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow Qualtran v0.7.0 SelectSwapQROM and Adjoint.

"""Resolve clean and borrowed blocked XOR lookup through native call owners."""

from collections.abc import Mapping

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.lookup.select_swap_declaration import SelectSwapDeclaration
from ..registers.lookup.select_swap_qrom_component import select_swap_qrom_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_qrom_factories import _dimensions, read_qrom_declaration
from .legacy_word_factories import _rename_program


def select_swap_factories(
    *, adjoint_factory: ComponentFactory | None = None
) -> dict[str, ComponentFactory]:
    """Own source lookup recipes and exact inverse calls per library compilation.

    Unknown adjoint children delegate to the supplied existing owner, if any.
    The inverse preserves the original clean allocation from forward Free,
    including when forward banks are borrowed. SDK constructors are not used.
    """
    recipes: dict[FunctionRecord, tuple[SelectSwapDeclaration, bool]] = {}

    def make(
        name: str,
        registers: tuple[Register, ...] | None,
        table: SelectSwapDeclaration,
        inverse: bool,
    ) -> FunctionRecord | ProgramRecord:
        base = table.lookup
        ports = tuple(table.signature)
        if registers is not None and ports != registers:
            raise SerializationError("Invalid select-swap register signature")
        try:
            program = select_swap_qrom_program(
                table,
                tuple(int(width) for width in base.target_bitsizes),
                selection_widths=tuple(int(width) for width in base.selection_bitsizes),
                num_controls=int(base.num_controls),
                dirty=table.dirty,
                adjoint=inverse,
            )
        except (ValueError, TypeError, OverflowError, IndexError, KeyError):
            root = FunctionRecord(name, (), registers=ports)
            recipes[root] = table, inverse
            return root
        record = _rename_program(
            program,
            name,
        )
        root = record.functions[0]
        if registers is not None and root.registers != registers:
            raise SerializationError("Invalid select-swap register signature")
        recipes[root] = table, inverse
        return record

    def forward(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        fields = dict(arguments)
        dirty = bool(fields.pop("use_dirty_ancilla", True))
        if "log_block_sizes" not in fields:
            raise SerializationError("Missing select-swap block metadata")
        logs = _dimensions(fields.pop("log_block_sizes"))
        base = read_qrom_declaration(fields)
        table = SelectSwapDeclaration(base, logs, dirty)
        return make(name, registers, table, False)

    def adjoint(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            set(arguments) != {"subbloq"}
            or not isinstance(child, FunctionRecord)
            or child not in recipes
        ):
            if adjoint_factory is not None:
                return adjoint_factory(name, registers, arguments)
            raise SerializationError("Missing select-swap adjoint constructor context")
        table, inverse = recipes[child]
        return make(name, registers, table, not inverse)

    return {
        "qualtran.bloqs.data_loading.select_swap_qrom.SelectSwapQROM": forward,
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
