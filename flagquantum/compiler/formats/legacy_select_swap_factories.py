# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow Qualtran v0.7.0 SelectSwapQROM and Adjoint.

"""Resolve clean and borrowed blocked XOR lookup through native call owners."""

from collections.abc import Mapping
from typing import cast

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.lookup.qrom_table import QromTable
from ..registers.lookup.select_swap_qrom_component import select_swap_qrom_program
from ..registers.lookup.source_qroam_clean import QroamCleanTable
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_lookup_metadata import read_blocked_lookup_arguments
from .legacy_word_factories import _rename_program


def select_swap_factories(
    *, adjoint_factory: ComponentFactory | None = None
) -> dict[str, ComponentFactory]:
    """Own source lookup recipes and exact inverse calls per library compilation.

    Unknown adjoint children delegate to the supplied existing owner, if any.
    The inverse preserves the original clean allocation from forward Free,
    including when forward banks are borrowed. SDK constructors are not used.
    """
    recipes: dict[FunctionRecord, tuple[QromTable, QroamCleanTable, bool, bool]] = {}

    def make(
        name: str,
        registers: tuple[Register, ...] | None,
        base: QromTable,
        table: QroamCleanTable,
        dirty: bool,
        inverse: bool,
    ) -> ProgramRecord:
        record = _rename_program(
            select_swap_qrom_program(
                table,
                cast(tuple[int, ...], base.target_bitsizes),
                selection_widths=cast(tuple[int, ...], base.selection_bitsizes),
                num_controls=cast(int, base.num_controls),
                dirty=dirty,
                adjoint=inverse,
            ),
            name,
        )
        root = record.functions[0]
        if registers is not None and root.registers != registers:
            raise SerializationError("Invalid select-swap register signature")
        recipes[root] = base, table, dirty, inverse
        return record

    def forward(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        fields = dict(arguments)
        dirty = bool(fields.pop("use_dirty_ancilla", True))
        base, logs = read_blocked_lookup_arguments(fields)
        if any(base.target_shapes):
            raise SerializationError("Select-swap requires scalar dataset targets")
        assert base.datasets is not None
        table = QroamCleanTable(cast(tuple[int, ...], base.shape), logs, base.datasets)
        return make(name, registers, base, table, dirty, False)

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
        base, table, dirty, inverse = recipes[child]
        return make(name, registers, base, table, dirty, not inverse)

    return {
        "qualtran.bloqs.data_loading.select_swap_qrom.SelectSwapQROM": forward,
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
