# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 arithmetic/addition.py.

"""Qualify complete stored addition dependencies against their native owners."""

from collections.abc import Mapping

from examples.algorithms.addition_decomposition import addition_symbols
from examples.algorithms.subtraction_decomposition import subtraction_symbols
from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.compiler.registers.arithmetic.out_of_place_addition_component import (
    out_of_place_addition_program,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.primitives.always_component import (
    always_component_program,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import RegisterSide
from flagquantum.drawer.register_score_data import RegisterWireSymbolSupplier


def addition_child_decomposition(
    record: ProgramRecord, entry: str, parameters: Mapping[str, object]
) -> tuple[str, ProgramRecord, RegisterWireSymbolSupplier] | None:
    """Reuse glyphs only for an identical complete reachable definition.

    Port domains select existing constructors. Literal matching retains every SSA
    operand, phase, measurement and feedback instruction before qualification.
    """
    function = next(f for f in record.functions if f.name == entry)
    registers = function.registers or ()
    ports = tuple(register.name for register in registers)
    candidates: list[tuple[str, ProgramRecord]] = []
    widths = tuple(register.bitsize for register in registers)
    if any(type(width) is not int or width < 1 for width in widths):
        return None
    if (
        ports == ("a", "b")
        and widths[0] <= widths[1]
        and all(
            not register.shape
            and register.dtype.encoding in {"uint", "signed", "montgomery"}
            for register in registers
        )
    ):
        for adjoint in (False, True):
            add = addition_component_program(
                registers[0].bitsize,
                registers[1].bitsize,
                left_dtype=registers[0].dtype,
                right_dtype=registers[1].dtype,
                adjoint=adjoint,
            )
            label = "Add†" if adjoint else "Add"
            candidates.extend(((label, add), ("Always", always_component_program(add))))
    elif ports == ("a", "b", "c"):
        width = registers[0].bitsize
        count = registers[2].bitsize
        if registers[1].bitsize == width and count in (width, width + 1):
            adjoint = registers[2].side == RegisterSide.LEFT
            candidates.append(
                (
                    "OutOfPlaceAdder",
                    out_of_place_addition_program(
                        width, include_msb=count == width + 1, adjoint=adjoint
                    ),
                )
            )
    return next(
        (
            (
                label,
                template,
                (
                    subtraction_symbols(template)
                    if label == "Always"
                    else addition_symbols(template, parameters)
                ),
            )
            for label, template in candidates
            if matches_component_definition(record, entry, template)
        ),
        None,
    )
