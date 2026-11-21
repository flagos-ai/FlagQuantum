# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SwapWithZero.build_composite_bloq.

"""Thread typed selector and target ports through the shared exchange stages."""

from math import prod

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ..component_builder import ComponentProgramBuilder
from ..primitives.relative_swap_component import relative_swap_program
from .swap_with_zero import _swap_sequence, swap_with_zero_signature


def swap_with_zero_program(
    widths: tuple[int, ...], word: int, shape: tuple[int, ...], *, adjoint: bool = False
) -> ProgramRecord:
    """Retain each source exchange and return caller-owned bounded selectors.

    Traversal validation and exchange order belong to the shared physical
    network owner. Child bodies come from the typed relative-swap Program.
    Selector Join retains the native bounded descriptor at the return boundary.
    This differs from the original generic QAny Join's nominal result dtype.
    No work or measurement is introduced.
    """
    if any(type(width) is not int or width < 0 for width in widths):
        raise ValueError("Swap with zero requires nonnegative concrete selector widths")
    if type(word) is not int or word < 1:
        raise ValueError("Swap with zero requires a positive concrete word width")
    selections = tuple(
        tuple(range(sum(widths[:i]), sum(widths[: i + 1]))) for i in range(len(widths))
    )
    offset = sum(widths)
    targets = tuple(
        tuple(range(offset + index * word, offset + (index + 1) * word))
        for index in range(prod(shape))
    )
    stages = _swap_sequence(selections, shape, targets, adjoint)
    signature = swap_with_zero_signature(widths, word, shape)
    builder = ComponentProgramBuilder(signature)
    controls: dict[int, RegisterBinding] = {}
    selector_bits = []
    for register, physical in zip(
        signature[:-1], (p for p in selections if p), strict=True
    ):
        binding = builder.inputs[register.name]
        assert isinstance(binding, RegisterBinding)
        bits = builder.split(binding)
        selector_bits.append((register, physical))
        controls.update(zip(physical, bits, strict=True))
    values = np.asarray(builder.inputs["targets"]).reshape(-1).copy()
    child = relative_swap_program(word)
    for control, left, right in stages:
        result = builder.call_program(
            child, {"ctrl": controls[control], "x": values[left], "y": values[right]}
        )
        active = result["ctrl"]
        assert isinstance(active, RegisterBinding)
        controls[control], values[left], values[right] = (
            active,
            result["x"],
            result["y"],
        )
    outputs = {
        register.name: builder.join([controls[i] for i in physical], register.dtype)
        for register, physical in selector_bits
    }
    return builder.finalize({**outputs, "targets": values.reshape(shape)})
