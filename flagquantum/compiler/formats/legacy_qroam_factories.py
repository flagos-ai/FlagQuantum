# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from Qualtran v0.7.0 data_loading/qroam_clean.py.

"""Resolve clean lookup constructors without replacing the source erasure leaf."""

from collections.abc import Mapping
from dataclasses import replace
from math import prod
from typing import cast

import numpy as np

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register, RegisterSide
from ...core.ir.program.signatures import Signature
from ...errors import SerializationError
from ..registers.component_builder import ComponentProgramBuilder
from ..registers.lookup.qroam_clean_component import qroam_clean_program
from ..registers.lookup.qrom_table import QromTable
from ..registers.lookup.source_qroam_clean import QroamCleanTable
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_qrom_factories import _dimensions
from .legacy_word_factories import _rename_program


def _metadata(
    arguments: Mapping[str, ResolvedArgument],
) -> tuple[QromTable, tuple[int, ...]]:
    required = {
        "data_or_shape",
        "selection_bitsizes",
        "target_bitsizes",
        "log_block_sizes",
    }
    if not required <= arguments.keys() or arguments.keys() - required - {
        "target_shapes",
        "num_controls",
    }:
        raise SerializationError("Invalid clean QROAM constructor arguments")
    data = arguments["data_or_shape"]
    if (
        not isinstance(data, (tuple, list, np.ndarray))
        or isinstance(data, np.ndarray)
        and data.ndim == 0
        or len(data) == 0
    ):
        raise SerializationError("Serialized clean QROAM requires concrete datasets")
    widths = _dimensions(arguments["target_bitsizes"])
    raw_shapes = arguments.get("target_shapes", ((),) * len(widths))
    if not isinstance(raw_shapes, (tuple, list, np.ndarray)):
        raise SerializationError("Clean QROAM target shapes require nested vectors")
    shapes = tuple(_dimensions(shape) for shape in raw_shapes)
    table = QromTable.from_data(
        *data,
        target_bitsizes=widths,
        target_shapes=shapes,
        num_controls=_integer(
            {"num_controls": arguments.get("num_controls", 0)}, "num_controls"
        ),
    )
    table = replace(
        table, selection_bitsizes=_dimensions(arguments["selection_bitsizes"])
    )
    return table, _dimensions(arguments["log_block_sizes"])


def _erasure_signature(table: QromTable) -> tuple[Register, ...]:
    return tuple(
        (
            Register(
                register.name,
                register.dtype,
                register.shape_symbolic,
                RegisterSide.LEFT,
            )
            if register.name.startswith("target")
            else register
        )
        for register in table.signature
    )


def _check_signature(
    expected: tuple[Register, ...], supplied: tuple[Register, ...] | None
) -> None:
    if supplied is not None and expected != supplied:
        raise SerializationError("Invalid clean QROAM register signature")


def qroam_factories() -> dict[str, ComponentFactory]:
    """Keep forward recipes and reshape calls local to one library compilation.

    Native forward lookup uses shared source-reserved unary storage. The source
    adjoint is a model-only leaf, so its declaration remains bodyless. Separately
    certified native measurement erasure is not silently substituted for it.
    """
    recipes: dict[FunctionRecord, tuple[QromTable, QroamCleanTable]] = {}

    def forward(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        base, logs = _metadata(arguments)
        if any(base.target_shapes):
            raise SerializationError("QROAMClean requires scalar dataset targets")
        assert base.datasets is not None
        table = QroamCleanTable(cast(tuple[int, ...], base.shape), logs, base.datasets)
        record = _rename_program(
            qroam_clean_program(
                table,
                cast(tuple[int, ...], base.target_bitsizes),
                selection_widths=cast(tuple[int, ...], base.selection_bitsizes),
                num_controls=cast(int, base.num_controls),
            ),
            name,
        )
        root = record.functions[0]
        assert root.registers is not None
        _check_signature(root.registers, registers)
        recipes[root] = (base, table)
        return record

    def leaf(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord:
        base, _ = _metadata(arguments)
        signature = _erasure_signature(base)
        _check_signature(signature, registers)
        return FunctionRecord(name, (), registers=signature)

    def wrapper(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if set(arguments) != {"qroam_clean", "log_block_sizes"}:
            raise SerializationError(
                "QROAM erasure wrapper requires lookup and block metadata"
            )
        child = arguments["qroam_clean"]
        if not isinstance(child, FunctionRecord) or child not in recipes:
            raise SerializationError("Missing clean QROAM constructor context")
        _dimensions(arguments["log_block_sizes"])
        base, table = recipes[child]
        assert child.registers is not None
        signature = Signature(child.registers).adjoint()
        _check_signature(tuple(signature), registers)
        padded = tuple(
            n * k for n, k in zip(table.batch_shape, table.block_shape, strict=True)
        )
        leaf_table = QromTable.from_shape(
            padded,
            base.target_bitsizes,
            target_shapes=(table.block_shape,) * len(base.target_bitsizes),
            num_controls=base.num_controls,
        )
        erased = FunctionRecord(
            name + "_erase", (), registers=_erasure_signature(leaf_table)
        )
        builder = ComponentProgramBuilder(signature, (erased,), name=name)
        values: dict[str, object] = {
            reg.name: builder.inputs[reg.name]
            for reg in signature
            if not reg.name.startswith(("target", "junk_"))
        }
        for i, width in enumerate(base.target_bitsizes):
            if width == 0:
                continue
            target = builder.inputs[f"target{i}_"]
            junk = (
                np.asarray(builder.inputs[f"junk_target{i}_"]).flat
                if prod(table.block_shape) > 1
                else ()
            )
            values[f"target{i}_"] = np.asarray([target, *junk], dtype=object).reshape(
                table.block_shape
            )
        return builder.finalize(builder.call_dict(erased.name, values))

    prefix = "qualtran.bloqs.data_loading.qroam_clean."
    return {
        prefix + "QROAMClean": forward,
        prefix + "QROAMCleanAdjoint": leaf,
        prefix + "QROAMCleanAdjointWrapper": wrapper,
    }
