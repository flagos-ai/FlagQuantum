# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Comparator and its composite adjoint.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/sorting.py

"""Compose compare/exchange with owned logical history and exact restoration."""

from dataclasses import replace

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_hierarchy
from ..component_builder import ComponentProgramBuilder
from ..component_graph import _namespace_helpers
from ..primitives.controlled_swap_component import controlled_swap_program
from .ordering_component import greater_than_program


def compare_exchange_program(width: int, *, adjoint: bool = False) -> ProgramRecord:
    """Order unsigned words and retain a>b, or consume the valid output history.

    History is caller-visible logical output. Clean comparison work belongs to
    GreaterThan and returns zero. The inverse clears valid history before free.
    """
    if type(adjoint) is not bool:
        raise ValueError("Compare/exchange direction requires a boolean")
    compare = greater_than_program(width, adjoint=adjoint)
    swap = controlled_swap_program(width)
    bit, word = quantum_bit_type(), unsigned_register_type(width)
    side = RegisterSide.LEFT if adjoint else RegisterSide.RIGHT
    builder = ComponentProgramBuilder(
        Signature(
            (Register("a", word), Register("b", word), Register("out", bit, side=side))
        )
    )
    history = builder.inputs["out"] if adjoint else builder.allocate(dtype=bit)
    a, b = builder.inputs["a"], builder.inputs["b"]
    if adjoint:
        exchanged = builder.call_program(swap, {"ctrl": history, "x": a, "y": b})
        history, a, b = exchanged["ctrl"], exchanged["x"], exchanged["y"]
    compared = builder.call_program(compare, {"a": a, "b": b, "target": history})
    a, b, history = compared["a"], compared["b"], compared["target"]
    outputs = {"a": a, "b": b}
    if adjoint:
        assert isinstance(history, RegisterBinding)
        builder.free(history)
    else:
        exchanged = builder.call_program(swap, {"ctrl": history, "x": a, "y": b})
        outputs = {"a": exchanged["x"], "b": exchanged["y"], "out": exchanged["ctrl"]}
    return builder.finalize(outputs)


def adjoint_comparison_network(record: ProgramRecord, width: int) -> ProgramRecord:
    """Reverse a sorting hierarchy at its exact compare/exchange boundaries.

    Network owners supply graphs built from this module's comparator signature.
    The shared hierarchy adjoint reverses array operand/history connections and
    directional ports. The specialized comparator owns measurement restoration.
    """
    inverse = compare_exchange_program(width, adjoint=True)
    signature = tuple(compare_exchange_program(width).functions[0].registers or ())
    names = {f.name for f in record.functions}
    index = 0
    prefix = "__fq_comparison_inverse_0"
    while any(name.startswith(prefix) for name in names):
        index += 1
        prefix = f"__fq_comparison_inverse_{index}"
    inverse = _namespace_helpers(inverse, prefix)
    mapping = {
        function.name: prefix + "_main"
        for function in record.functions
        if function.registers == signature
    }
    if "main" in mapping:
        raise ValueError(
            "Network adjoint requires a sorting entry above its comparators"
        )
    return adjoint_component_hierarchy(
        replace(record, functions=(*record.functions, *inverse.functions)), mapping
    )
