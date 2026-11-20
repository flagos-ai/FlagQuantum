# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Swap and CSwap declarations/decompositions.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/swap.py

"""Own word-exchange metadata and store the original atomic SWAP child calls."""

from collections.abc import Hashable
from dataclasses import dataclass

from ....core.ir import CircuitIR
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt
from ..._component_trace.resource_generalization import CliffordCall, ControlledSwapCall
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from .controlled_swap_component import controlled_swap_program
from .swap import swap_bit_ir


@dataclass(frozen=True)
class SwapDeclaration:
    """Retain ordinary or controlled word signatures and atomic multiplicity."""

    bitsize: SymbolicInt
    controlled: bool = False

    @property
    def signature(self) -> Signature:
        words = {"x": self.bitsize, "y": self.bitsize}
        return Signature.build(**({"ctrl": 1, **words} if self.controlled else words))

    def direct_calls(self) -> dict[Hashable, SymbolicInt]:
        leaf = ControlledSwapCall(1) if self.controlled else CliffordCall("swap_bit", 2)
        return {leaf: self.bitsize}

    def program(self) -> ProgramRecord:
        if type(self.bitsize) is not int:
            raise ValueError("Swap physical construction requires a concrete width")
        return (
            controlled_swap_program(self.bitsize)
            if self.controlled
            else swap_word_program(self.bitsize)
        )


def swap_word_program(width: int) -> ProgramRecord:
    """Exchange MSB-first through words with no private work or feedback."""
    if type(width) is not int or width < 1:
        raise ValueError("Swap requires a positive concrete width")
    leaf = bit_gate_function(
        "swap_bit", tuple(Signature.build(x=1, y=1)), CircuitIR(2, swap_bit_ir(0, 1))
    )
    signature = Signature.build(x=width, y=width)
    builder = ComponentProgramBuilder(signature, (leaf,))
    x, y = builder.inputs["x"], builder.inputs["y"]
    assert isinstance(x, RegisterBinding) and isinstance(y, RegisterBinding)
    xs, ys = builder.split(x), builder.split(y)
    for index in range(width):
        outputs = builder.call_dict("swap_bit", {"x": xs[index], "y": ys[index]})
        xs[index], ys[index] = outputs["x"], outputs["y"]
    return builder.finalize({"x": builder.join(xs), "y": builder.join(ys)})
