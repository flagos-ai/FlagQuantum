# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 arithmetic/permutation.py.

"""Own finite cycle wiring around supplied equality and controlled-XOR bodies."""

from collections.abc import Callable

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder


def permutation_cycle_graph(
    dtype: RegisterType,
    cycle: tuple[int, ...],
    equality: Callable[[int], ProgramRecord],
    controlled_xor: Callable[[int], ProgramRecord],
) -> ProgramRecord:
    """Preserve marker ownership and source-ordered predicate/XOR pairs.

    The caller owns cycle/domain validation and supplies typed child definitions.
    Symbolic widths retain the same finite graph. Equality owns its predicate
    work, while this cycle allocates and releases its marker.
    """
    builder = ComponentProgramBuilder(Signature.build_from_dtypes(x=dtype))
    marker: BindingCollection = builder.allocate()
    word = builder.inputs["x"]
    for index, value in enumerate(cycle):
        predicate = equality(value)
        result = builder.call_program(
            predicate, {"x": word, "target": marker}, entry=predicate.functions[0].name
        )
        word, marker = result["x"], result["target"]
        delta = value ^ cycle[(index + 1) % len(cycle)]
        xor = controlled_xor(delta)
        result = builder.call_program(
            xor, {"ctrl": marker, "x": word}, entry=xor.functions[0].name
        )
        marker, word = result["ctrl"], result["x"]
    predicate = equality(cycle[0])
    result = builder.call_program(
        predicate, {"x": word, "target": marker}, entry=predicate.functions[0].name
    )
    marker = result["target"]
    assert isinstance(marker, RegisterBinding)
    builder.free(marker)
    return builder.finalize({"x": result["x"]})
