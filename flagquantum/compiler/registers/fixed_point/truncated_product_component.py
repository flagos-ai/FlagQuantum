# SPDX-License-Identifier: Apache-2.0
# Shared typed composition of the D4/D5 construction in fixed_point_scaling.py.

"""Owned truncated-product results over shared And and stored CAdd calls."""

import numpy as np

from ....core.ir import CircuitIR
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ..arithmetic.controlled_addition_component import controlled_addition_program
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..primitives.and_compute import and_compute_ir, and_uncompute_ir


def truncated_product_program(
    inputs: tuple[Register, Register],
    result_dtype: RegisterType,
    first_width: int,
    omitted_low_bits: int,
    *,
    adjoint: bool,
) -> ProgramRecord:
    """Compose the retained product terms and their exact logical inverse.

    Callers validate formats and widths. Input words and result are MSB first.
    The first active multiplier bit controls retained And copies. Each later
    bit controls an unsigned prefix addition into a growing result suffix.
    Shared CAdd owns its loaded word and carries for each call. Initial copies
    occupy the result itself. The inverse reverses the logical calls before
    measurement lowering, clears the matching image and releases its storage.
    """
    if type(adjoint) is not bool:
        raise ValueError("Truncated product direction requires a boolean")
    bit = quantum_bit_type()
    copies = bit_gate_function(
        "product_copy",
        (Register("bits", bit, (3,)),),
        CircuitIR(
            3,
            and_uncompute_ir(0, 1, 2, 0) if adjoint else and_compute_ir(0, 1, 2),
        ),
    )
    builder = ComponentProgramBuilder(
        Signature(
            (
                *inputs,
                Register(
                    "result",
                    result_dtype,
                    side=RegisterSide.LEFT if adjoint else RegisterSide.RIGHT,
                ),
            )
        ),
        (copies,),
    )
    words = tuple(builder.inputs[register.name] for register in inputs)
    left, right = words
    target = (
        builder.inputs["result"] if adjoint else builder.allocate(dtype=result_dtype)
    )
    assert (
        isinstance(left, RegisterBinding)
        and isinstance(right, RegisterBinding)
        and isinstance(target, RegisterBinding)
    )
    real = list(builder.split(left))
    multiplier = list(builder.split(right))
    result = list(builder.split(target))
    active = tuple(reversed(range(len(multiplier))))[omitted_low_bits:]
    stages = (
        (
            *(("copy", active[0], i + 1) for i in range(first_width)),
            *(
                ("add", control, width)
                for width, control in enumerate(active[1:], first_width + 1)
            ),
        )
        if active
        else ()
    )
    for kind, control, width in reversed(stages) if adjoint else stages:
        if kind == "copy":
            position = len(result) - first_width + width - 1
            copied = builder.call_dict(
                "product_copy",
                {
                    "bits": np.array(
                        [multiplier[control], real[width - 1], result[position]],
                        dtype=object,
                    )
                },
            )
            multiplier[control], real[width - 1], result[position] = np.asarray(
                copied["bits"]
            )
        else:
            small, wide = RegisterType("uint", width), RegisterType("uint", width + 1)
            added = builder.call_program(
                controlled_addition_program(small, wide, adjoint=adjoint),
                {
                    "ctrl": multiplier[control],
                    "a": builder.join(real[:width], small),
                    "b": builder.join(result[-width - 1 :], wide),
                },
            )
            ctrl, a, b = added["ctrl"], added["a"], added["b"]
            assert (
                isinstance(ctrl, RegisterBinding)
                and isinstance(a, RegisterBinding)
                and isinstance(b, RegisterBinding)
            )
            multiplier[control] = ctrl
            real[:width] = builder.split(a)
            result[-width - 1 :] = builder.split(b)
    output = builder.join(result, result_dtype)
    returned = {
        inputs[0].name: builder.join(real, inputs[0].dtype),
        inputs[1].name: builder.join(multiplier, inputs[1].dtype),
    }
    if adjoint:
        builder.free(output)
    else:
        returned["result"] = output
    return builder.finalize(returned)
