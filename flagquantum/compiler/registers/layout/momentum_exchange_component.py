# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal reshape calls from locked Qualtran v0.7.0 MultiplexedCSwap3D.

"""Keep the source's shaped exchange views and projectile-prefix ownership."""

from math import prod

import numpy as np
from numpy.typing import NDArray

from ....core.ir.program import ProgramRecord
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_calls
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..lookup.multiplexed_swap import MultiplexedSwapLayout
from ..lookup.multiplexed_swap_component import multiplexed_swap_program


def _reshape(
    builder: ComponentProgramBuilder,
    value: BindingCollection,
    shape: tuple[int, ...],
    width: int,
) -> NDArray[np.object_]:
    if isinstance(value, np.ndarray):
        bits = builder.split(
            builder.join(np.concatenate([builder.split(word) for word in value.flat]))
        )
    else:
        bits = builder.split(value)
    return np.array(
        [
            builder.join(bits[index * width : (index + 1) * width])
            for index in range(prod(shape))
        ],
        dtype=object,
    ).reshape(shape)


def momentum_exchange_program(
    signature: Signature, *, controlled: bool = False, adjoint: bool = False
) -> ProgramRecord:
    """Exchange caller-owned axes with original all-one control activation.

    The supplied signature owns the source selector/bank descriptors. For the
    controlled form, only each junk word's leading electron-width prefix enters
    the exchange. Diagram cvs values do not alter the omitted source argument.
    """
    targets = signature.get_left("targets")
    particles, axes = targets.shape
    width = int(targets.bitsize)
    junk_width = int(signature.get_left("junk").bitsize)
    if axes != 3 or junk_width < width:
        raise ValueError(
            "Momentum exchange requires three axes and adequate junk width"
        )
    builder = ComponentProgramBuilder(signature)
    flat_targets = _reshape(builder, builder.inputs["targets"], (particles,), 3 * width)
    junk = builder.inputs["junk"]
    if controlled:
        bits = _reshape(builder, junk, (3, junk_width), 1)
        prefixes = np.array(
            [builder.join(bits[axis, :width]) for axis in range(3)], dtype=object
        )
        output = _reshape(builder, prefixes, (), 3 * width)
    else:
        output = _reshape(builder, junk, (), 3 * width)
    layout = MultiplexedSwapLayout(
        (signature.get_left("sel"),),
        3 * width,
        (signature.get_left("ctrl"),) if controlled else (),
    )
    child = multiplexed_swap_program(layout)
    arguments = {
        "sel": builder.inputs["sel"],
        "targets": flat_targets,
        "output": output,
    }
    if controlled:
        arguments["ctrl"] = builder.inputs["ctrl"]
    values = builder.call_program(child, arguments)
    if controlled:
        prefixes = _reshape(builder, values["output"], (3,), width)
        for axis in range(3):
            bits[axis, :width] = builder.split(prefixes[axis])
    restored_targets = _reshape(builder, values["targets"], (particles, 3), width)
    restored_junk = (
        _reshape(builder, bits, (3,), junk_width)
        if controlled
        else _reshape(builder, values["output"], (3,), width)
    )
    record = builder.finalize(
        {
            **({"ctrl": values["ctrl"]} if controlled else {}),
            "sel": values["sel"],
            "targets": restored_targets,
            "junk": restored_junk,
        }
    )
    if not adjoint:
        return record
    return adjoint_component_calls(
        record, ((child, multiplexed_swap_program(layout, adjoint=True)),)
    )
