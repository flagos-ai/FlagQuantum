# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 arithmetic/bitwise.py and basic_gates/on_each.py.

"""Qualify bitwise and repeated-X descendants through their executable owners."""

from collections.abc import Mapping

from examples.algorithms.word_arithmetic_decomposition import word_program, word_symbols
from flagquantum.compiler.registers.bits.bitwise_component import bitwise_word_program
from flagquantum.compiler.registers.bits.on_each import on_each_word_program
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.register_score_data import RegisterWireSymbolSupplier


def word_child_decomposition(
    record: ProgramRecord, entry: str, parameters: Mapping[str, object]
) -> tuple[str, ProgramRecord, RegisterWireSymbolSupplier] | None:
    """Match concrete word definitions or the symbolic complement graph literally."""
    function = next(f for f in record.functions if f.name == entry)
    registers = function.registers or ()
    if (
        len(registers) == 1
        and registers[0].name == "x"
        and type(registers[0].bitsize) is not int
        and not registers[0].shape_symbolic
        and registers[0].dtype.num_cbits == 0
    ):
        template = word_program("BitwiseNot", Signature(registers))
        assert template is not None
        if matches_component_definition(record, entry, template):
            return "BitwiseNot", template, word_symbols(template, "BitwiseNot")
        return None
    if not registers or any(
        type(register.bitsize) is not int
        or register.bitsize < 1
        or register.shape_symbolic
        or register.dtype.num_cbits != 0
        for register in registers
    ):
        return None
    ports = tuple(register.name for register in registers)
    dtype = registers[0].dtype
    width = registers[0].bitsize
    candidates: list[tuple[str, ProgramRecord]] = []
    if ports == ("x", "y") and registers[1].dtype == dtype:
        candidates.append(("Xor", bitwise_word_program(dtype, "xor")))
    elif ports == ("x",):
        candidates.append(("BitwiseNot", bitwise_word_program(dtype, "bitwise_not")))
        constant = parameters.get("k", 1)
        if type(constant) is int:
            candidates.append(
                (
                    "XorK",
                    bitwise_word_program(
                        dtype, "xor_constant", pattern=constant % (1 << width)
                    ),
                )
            )
    elif ports == ("q",):
        leaf = bit_gate_function(
            "bitwise_x",
            (Register("q", quantum_bit_type()),),
            CircuitIR(1, (Instruction("x", (0,)),)),
        )
        candidates.append(
            ("OnEach", on_each_word_program(width, leaf, target_dtype=dtype))
        )
    return next(
        (
            (
                label,
                template,
                word_symbols(template, label),
            )
            for label, template in candidates
            if matches_component_definition(record, entry, template)
        ),
        None,
    )
