# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Typed composition of FlagQuantum's reversible one's-complement conversion.
# Source signature and model: Qualtran v0.7.0 SignedIntegerToTwosComplement.
# The frozen source supplies no decomposition for this root.

"""Transfer a signed encoding word through shared conjunction lifetimes."""

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..control.conjunction_program import and_component_program
from .signed_encoding import _signed_encoding_components, signed_encoding_workspace_size


def signed_encoding_program(width: int, *, adjoint: bool = False) -> ProgramRecord:
    """Consume one's-complement x and return two's-complement y, or invert it.

    Words are MSB first. The sign-controlled increment has width-2 temporary
    conjunctions, allocated and measured away by their existing component owner.
    The inverse uses zero-payload borrow controls. Every raw word has a reversible
    image, including duplicate negative zero, which maps to the minimum signed
    integer. No assertion of a frozen source root decomposition is made.
    """
    count = signed_encoding_workspace_size(width)
    if type(adjoint) is not bool:
        raise ValueError("Signed conversion direction requires a boolean")
    before, after = RegisterType("ones_complement", width), RegisterType(
        "signed", width
    )
    if adjoint:
        before, after = after, before
    bit = quantum_bit_type()
    cx = bit_gate_function(
        "conversion_cx",
        (Register("ctrl", bit), Register("target", bit)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    values = (1, 0 if adjoint else 1)
    compute = and_component_program(values, name="conversion_and")
    erase = and_component_program(values, name="conversion_erase", uncompute=True)
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("x", before, side=RegisterSide.LEFT),
                Register("y", after, side=RegisterSide.RIGHT),
            )
        ),
        (cx, *compute.functions, *erase.functions),
    )
    word = builder.inputs["x"]
    assert isinstance(word, RegisterBinding)
    bits = builder.split(word)
    wires: dict[int, RegisterBinding] = dict(enumerate(bits))
    source = tuple(range(width))
    work = tuple(range(width, width + count))
    for kind, addresses in _signed_encoding_components(source, work):
        x, y = addresses[:2]
        if kind == "cx":
            result = builder.call_dict(
                "conversion_cx", {"ctrl": wires[x], "target": wires[y]}
            )
            control, target = result["ctrl"], result["target"]
            assert isinstance(control, RegisterBinding) and isinstance(
                target, RegisterBinding
            )
            wires[x], wires[y] = control, target
            continue
        controls = np.array([wires[x], wires[y]], dtype=object)
        if kind == "and":
            result = builder.call_dict("conversion_and", {"ctrl": controls})
            target = result["target"]
            assert isinstance(target, RegisterBinding)
            wires[addresses[2]] = target
        else:
            result = builder.call_dict(
                "conversion_erase",
                {"ctrl": controls, "target": wires.pop(addresses[2])},
            )
        returned = np.asarray(result["ctrl"])
        wires[x], wires[y] = returned[0], returned[1]
    return builder.finalize({"y": builder.join([wires[i] for i in source], after)})
