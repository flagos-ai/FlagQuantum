# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 data_loading/qrom.py and unary_iteration_bloq.py.

"""Retain source XOR-word calls around shared bounded unary iteration."""

from typing import cast

import numpy as np

from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import unsigned_register_type
from ..bits.bitwise_component import bitwise_word_program
from ..bits.bitwise_component_control import controlled_bitwise_word_program
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..control.conjunction_program import and_component_program, conjunction_program
from ..control.unary_component_calls import append_bounded_unary_calls
from .qrom_data import qrom_break_early, qrom_load_words
from .qrom_declaration import QromDeclaration
from .qrom_table import QromTable


def qrom_program(
    table: QromTable | QromDeclaration, *, reserve_workspace: bool = False
) -> ProgramRecord:
    """XOR row-major data into dirty THRU targets with source predicate lifetimes.

    Selection words are MSB-first and target arrays retain row-major order.
    Unknown data remains a bodyless declaration. Concrete lookup owns and
    measurement-erases all predicate work, while target storage remains live.
    reserve_workspace retains each loop's source-ordered scratch storage,
    including clean slots whose last predicate has already been erased.
    """
    if type(reserve_workspace) is not bool:
        raise ValueError("QROM workspace reservation must be boolean")
    if isinstance(table, QromTable):
        if table.datasets is not None:
            table._require_concrete_dimensions()
        declaration = QromDeclaration(
            tuple(table.shape + target for target in table.target_shapes),
            (
                (None,) * len(table.target_shapes)
                if table.datasets is None
                else table._data()
            ),
            table.selection_bitsizes,
            table.target_bitsizes,
            table.target_shapes,
            table.num_controls,
        )
    else:
        declaration = table
    if not declaration.has_data():
        return ProgramRecord(
            (FunctionRecord("main", (), registers=tuple(declaration.signature)),)
        )
    data = declaration.data
    shape = cast(tuple[int, ...], declaration.data_shape)
    widths = cast(tuple[int, ...], declaration.target_bitsizes)

    def folded(prefix: tuple[int, ...], start: int, stop: int) -> bool:
        return qrom_break_early(data, prefix, start, stop)

    selector_registers = tuple(
        reg for reg in declaration.signature if reg.name.startswith("selection")
    )
    extents = cast(tuple[int, ...], declaration.selection_axes())
    zero_selection = not sum(reg.bitsize for reg in selector_registers) or folded(
        (), 0, extents[0]
    )
    builder = ComponentProgramBuilder(declaration.signature)
    target_values = {
        i: np.asarray(builder.inputs[f"target{i}_"]).copy()
        for i, width in enumerate(widths)
        if width
    }
    selectors: dict[int, tuple[RegisterBinding, ...]] = {}
    names = {axis: reg.name for axis, reg in enumerate(selector_registers)}
    count = int(declaration.num_controls)
    if count < 0:
        raise ValueError("Physical QROM requires nonnegative control width")
    control_word = builder.inputs.get("control")
    controls: list[RegisterBinding] = []
    if count:
        assert isinstance(control_word, RegisterBinding)
        controls = [control_word] if count == 1 else list(builder.split(control_word))
    reduction: dict[str, BindingCollection] = {}
    predicate = next(iter(controls), None)
    if count > 1:
        reduction_program = (
            and_component_program((1, 1), name="qrom_control")
            if count == 2
            else conjunction_program((1,) * count, name="qrom_control")
        )
        reduction = builder.call_program(
            reduction_program,
            {"ctrl": np.asarray(controls, dtype=object)},
            entry="qrom_control",
        )
        target = reduction["target"]
        assert isinstance(target, RegisterBinding)
        predicate = target

    def load(
        row: tuple[int, ...], active: RegisterBinding | None
    ) -> RegisterBinding | None:
        for i, index, value in qrom_load_words(
            data, cast(tuple[tuple[int, ...], ...], declaration.target_shapes), row
        ):
            width = widths[i]
            if not width:
                raise ValueError("Source QROM cannot bind a zero-width scalar load")
            words = target_values[i]
            dtype = unsigned_register_type(width)
            if active is None:
                returned = builder.call_program(
                    bitwise_word_program(dtype, "xor_constant", pattern=value),
                    {"x": words[index]},
                )
            else:
                returned = builder.call_program(
                    controlled_bitwise_word_program(
                        dtype, "xor_constant", pattern=value
                    ),
                    {"ctrl": active, "x": words[index]},
                )
                successor = returned["ctrl"]
                assert isinstance(successor, RegisterBinding)
                active = successor
            words[index] = returned["x"]
        return active

    def visit(
        axis: int, row: tuple[int, ...], active: RegisterBinding | None
    ) -> RegisterBinding | None:
        if axis not in selectors:
            word = builder.inputs[names[axis]]
            assert isinstance(word, RegisterBinding)
            selectors[axis] = (
                (word,) if word.register.bitsize == 1 else tuple(builder.split(word))
            )

        def leaf(index: int, predicate: RegisterBinding) -> RegisterBinding:
            selected = row + (index,)
            successor = (
                visit(axis + 1, selected, predicate)
                if axis + 1 < len(extents)
                else load(selected, predicate)
            )
            assert isinstance(successor, RegisterBinding)
            return successor

        selectors[axis], active = append_bounded_unary_calls(
            builder,
            selectors[axis],
            extents[axis],
            leaf,
            control=active,
            break_early=lambda start, stop: folded(row, start, stop),
            reserve_workspace=reserve_workspace,
        )
        return active

    predicate = (
        load((0,) * len(shape), predicate)
        if zero_selection
        else visit(0, (), predicate)
    )
    if count > 1:
        assert predicate is not None
        reduction["target"] = predicate
        inverse = (
            and_component_program((1, 1), name="qrom_control_erase", uncompute=True)
            if count == 2
            else conjunction_program(
                (1,) * count, name="qrom_control_erase", uncompute=True
            )
        )
        controls = list(
            np.asarray(
                builder.call_program(inverse, reduction, entry="qrom_control_erase")[
                    "ctrl"
                ]
            )
        )
    elif predicate is not None:
        controls = [predicate]
    outputs: dict[str, object] = {}
    if count:
        assert isinstance(control_word, RegisterBinding)
        outputs["control"] = (
            controls[0]
            if count == 1
            else builder.join(controls, control_word.register.dtype)
        )
    for axis, name in names.items():
        word = builder.inputs[name]
        assert isinstance(word, RegisterBinding)
        outputs[name] = (
            word
            if axis not in selectors
            else (
                selectors[axis][0]
                if word.register.bitsize == 1
                else builder.join(selectors[axis], word.register.dtype)
            )
        )
    outputs.update(
        (f"target{i}_", values if values.shape else values[()])
        for i, values in target_values.items()
    )
    return builder.finalize(outputs)
