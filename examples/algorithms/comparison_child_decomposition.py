# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 arithmetic/comparison.py.

"""Qualify stored comparison trees through their existing executable owners."""

from typing import Literal, cast

from examples.algorithms.comparison_decomposition import (
    comparison_symbols,
    symbolic_greater_than_program,
)
from flagquantum.compiler.registers.comparison.half_component import (
    half_comparison_program,
)
from flagquantum.compiler.registers.comparison.ordering_component import (
    greater_than_program,
    less_equal_program,
)
from flagquantum.compiler.registers.comparison.tree_component import (
    mixer_program,
    single_qubit_comparison_program,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.register_score_data import RegisterWireSymbolSupplier


def comparison_child_decomposition(
    record: ProgramRecord, entry: str
) -> tuple[str, ProgramRecord, RegisterWireSymbolSupplier] | None:
    """Retain constructor direction and require a literal reachable-body match."""
    function = next(f for f in record.functions if f.name == entry)
    registers = function.registers or ()
    ports = tuple(register.name for register in registers)
    candidates: list[tuple[str, ProgramRecord]] = []
    if ports == ("x", "y", "ancilla"):
        candidates.append(
            (
                "BiQubitsMixer",
                mixer_program(adjoint=registers[-1].side == RegisterSide.LEFT),
            )
        )
    elif ports == ("a", "b", "less_than", "greater_than"):
        candidates.append(
            (
                "SingleQubitCompare",
                single_qubit_comparison_program(
                    adjoint=registers[-1].side == RegisterSide.LEFT
                ),
            )
        )
    elif (
        ports == ("a", "b", "c", "target")
        and type(registers[0].bitsize) is int
        and registers[0].bitsize > 0
        and registers[0].dtype.encoding in {"uint", "signed"}
    ):
        for relation, label in (
            ("gt", "LinearDepthHalfGreaterThan"),
            ("ge", "LinearDepthHalfGreaterThanEqual"),
            ("lt", "LinearDepthHalfLessThan"),
            ("le", "LinearDepthHalfLessThanEqual"),
        ):
            candidates.append(
                (
                    label,
                    half_comparison_program(
                        registers[0].dtype,
                        cast(Literal["gt", "ge", "lt", "le"], relation),
                        adjoint=registers[2].side == RegisterSide.LEFT,
                    ),
                )
            )
    elif ports in (("x", "y", "target"), ("a", "b", "target")) and all(
        type(register.bitsize) is int
        and register.bitsize > 0
        and register.dtype.encoding == "uint"
        and not register.shape_symbolic
        for register in registers[:2]
    ):
        if ports[0] == "x":
            candidates.append(
                (
                    "LessThanEqual",
                    less_equal_program(registers[0].bitsize, registers[1].bitsize),
                )
            )
        else:
            candidates.append(
                (
                    "GreaterThan",
                    greater_than_program(registers[0].bitsize, registers[1].bitsize),
                )
            )
    elif (
        ports == ("a", "b", "target")
        and any(type(register.bitsize) is not int for register in registers[:2])
        and all(
            register.dtype.encoding == "uint" and not register.shape_symbolic
            for register in registers[:2]
        )
    ):
        candidates.append(
            ("GreaterThan", symbolic_greater_than_program(Signature(registers)))
        )
    return next(
        (
            (label, template, comparison_symbols(template, label, {}))
            for label, template in candidates
            if matches_component_definition(record, entry, template)
        ),
        None,
    )
