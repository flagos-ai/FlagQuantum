# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 arithmetic/permutation.py and its dependencies.

"""Qualify permutation descendants from the owning cycle values and exact bodies."""

from collections.abc import Mapping

from examples.algorithms.comparison_decomposition import comparison_symbols
from examples.algorithms.control_child_decomposition import control_child_decomposition
from examples.algorithms.permutation_decomposition import permutation_catalog_program
from examples.algorithms.word_arithmetic_decomposition import word_symbols
from flagquantum.compiler.registers.bits.bitwise_component_control import (
    controlled_bitwise_word_program,
)
from flagquantum.compiler.registers.comparison.equality_component import (
    equality_constant_program,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.partition_symbols import partition_wire_symbol
from flagquantum.drawer.register_score_data import RegisterWireSymbolSupplier


def permutation_child_decomposition(
    record: ProgramRecord, entry: str, parameters: Mapping[str, object]
) -> tuple[str, ProgramRecord, RegisterWireSymbolSupplier] | None:
    """Match full reachable definitions before applying source labels and glyphs.

    Cycle values select equality predicates and XOR deltas. The existing control
    provider owns mixed-polarity conjunction and measured-adjoint symbols. No
    constructor is selected from a generated function name. Frozen source cycle
    drawings fail, so their matched bodies retain native partition symbols.
    """
    raw = (
        (parameters["cycle"],)
        if "cycle" in parameters
        else parameters.get("cycles", ())
    )
    if not isinstance(raw, tuple):
        return None
    cycles = tuple(
        cycle
        for cycle in raw
        if isinstance(cycle, tuple)
        and cycle
        and all(type(value) is int for value in cycle)
    )
    if not cycles:
        return None
    function = next(f for f in record.functions if f.name == entry)
    registers = function.registers or ()
    if not registers:
        return None
    ports = tuple(register.name for register in registers)
    candidates: list[tuple[str, ProgramRecord, RegisterWireSymbolSupplier]] = []
    values = tuple(dict.fromkeys(value for cycle in cycles for value in cycle))
    width = registers[0].bitsize
    if (
        ports == ("x",)
        and "N" in parameters
        and dict(registers[0].dtype.parameters).get("bound") == parameters["N"]
    ):
        for cycle in cycles:
            template = permutation_catalog_program(
                "PermutationCycle",
                Signature(registers),
                {"N": parameters["N"], "cycle": cycle},
            )
            assert template is not None
            candidates.append(("PermutationCycle", template, partition_wire_symbol))
    elif ports == ("x", "target") and type(width) is int and width > 0:
        for value in values:
            if 0 <= value < 1 << width:
                template = equality_constant_program(width, value)
                candidates.append(
                    (
                        "EqualsAConstant",
                        template,
                        comparison_symbols(template, "EqualsAConstant", {"val": value}),
                    )
                )
    elif ports == ("ctrl", "x") and type(registers[1].bitsize) is int:
        deltas = dict.fromkeys(
            value ^ cycle[(i + 1) % len(cycle)]
            for cycle in cycles
            for i, value in enumerate(cycle)
        )
        for delta in deltas:
            template = controlled_bitwise_word_program(
                registers[1].dtype, "xor_constant", pattern=delta
            )
            candidates.append(
                ("Controlled", template, word_symbols(template, "Controlled"))
            )
    else:
        for value in values:
            selected = control_child_decomposition(record, entry, {"val": value})
            if selected is not None:
                return selected
    return next(
        (
            candidate
            for candidate in candidates
            if matches_component_definition(record, entry, candidate[1])
        ),
        None,
    )
