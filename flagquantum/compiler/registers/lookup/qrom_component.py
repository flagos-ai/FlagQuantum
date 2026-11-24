# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 data_loading/qrom.py and unary_iteration_bloq.py.

"""Retain source XOR-word calls around shared bounded unary iteration."""

from math import prod
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
from .qrom_table import QromTable
from .source_qrom import _uniform


def qrom_program(table: QromTable, *, reserve_workspace: bool = False) -> ProgramRecord:
    """XOR row-major data into dirty THRU targets with source predicate lifetimes.

    Selection words are MSB-first and target arrays retain row-major order.
    Unknown data remains a bodyless declaration. Concrete lookup owns and
    measurement-erases all predicate work, while target storage remains live.
    reserve_workspace retains each loop's source-ordered scratch storage,
    including clean slots whose last predicate has already been erased.
    """
    if type(reserve_workspace) is not bool:
        raise ValueError("QROM workspace reservation must be boolean")
    if table.datasets is None:
        return ProgramRecord(
            (FunctionRecord("main", (), registers=tuple(table.signature)),)
        )
    table._require_concrete_dimensions()
    data = table._data()
    shape = cast(tuple[int, ...], table.shape)
    widths = cast(tuple[int, ...], table.target_bitsizes)
    constant = all(_uniform(values) for values in data)
    if not constant and any(n == 1 for n in shape) and any(n > 1 for n in shape):
        raise ValueError(
            "Source QROM cannot index mixed singleton axes in a nonuniform lookup"
        )
    builder = ComponentProgramBuilder(table.signature)
    target_values = {
        i: np.asarray(builder.inputs[f"target{i}_"]).copy()
        for i, width in enumerate(widths)
        if width
    }
    selectors: dict[int, tuple[RegisterBinding, ...]] = {}
    selection_ports = iter(table._selection_layout())
    names = {
        axis: next(selection_ports)[0]
        for axis, extent in enumerate(shape)
        if extent > 1
    }
    count = cast(int, table.num_controls)
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

    def load(row: int, active: RegisterBinding | None) -> RegisterBinding | None:
        for i, width in enumerate(widths):
            if not width:
                continue
            words = target_values[i]
            for index in np.ndindex(words.shape):
                flat_index = (
                    int(np.ravel_multi_index(index, words.shape)) if words.shape else 0
                )
                value = data[i][row * words.size + flat_index]
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
        axis: int, row: int, active: RegisterBinding | None
    ) -> RegisterBinding | None:
        if axis not in selectors:
            word = builder.inputs[names[axis]]
            assert isinstance(word, RegisterBinding)
            selectors[axis] = (
                (word,) if word.register.bitsize == 1 else tuple(builder.split(word))
            )
        stride = prod(shape[axis + 1 :])

        def folded(start: int, stop: int) -> bool:
            return all(
                _uniform(
                    values[
                        (row + start * stride)
                        * target_values[i].size : (row + stop * stride)
                        * target_values[i].size
                    ]
                )
                for i, values in enumerate(data)
                if widths[i]
            )

        def leaf(index: int, predicate: RegisterBinding) -> RegisterBinding:
            selected = row + index * stride
            successor = (
                visit(axis + 1, selected, predicate)
                if axis + 1 < len(shape)
                else load(selected, predicate)
            )
            assert isinstance(successor, RegisterBinding)
            return successor

        selectors[axis], active = append_bounded_unary_calls(
            builder,
            selectors[axis],
            shape[axis],
            leaf,
            control=active,
            break_early=folded,
            reserve_workspace=reserve_workspace,
        )
        return active

    predicate = (
        load(0, predicate)
        if constant or not any(table.selection_bitsizes)
        else visit(0, 0, predicate)
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
