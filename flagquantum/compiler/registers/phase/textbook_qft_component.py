# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 qft/qft_text_book.py.

"""Retain textbook Fourier calls and source-ordered unsigned word views."""

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..layout.component_word_views import call_through_word_program, gather_word_binding
from .phase_gradient_component import phase_gradient_unitary_program


def textbook_qft_program(width: int, *, with_reverse: bool = True) -> ProgramRecord:
    """Transform caller-owned MSB-first q, with private work owned by CZ powers."""
    if type(width) is not int or width < 1 or type(with_reverse) is not bool:
        raise ValueError("QFT requires a positive concrete width and boolean reversal")
    signature = Signature.build_from_dtypes(q=RegisterType("uint", width))
    builder = ComponentProgramBuilder(signature)
    q = tuple(range(width))
    word = builder.inputs["q"]
    assert isinstance(word, RegisterBinding)
    words = {q: word}
    positions = {"q": q}
    nominal_types = {wires: value.register.dtype for wires, value in words.items()}
    h = ProgramRecord(
        (
            bit_gate_function(
                "hadamard",
                (Register("q", quantum_bit_type()),),
                CircuitIR(1, (Instruction("h", (0,)),)),
            ),
        )
    )
    swap = ProgramRecord(
        (
            bit_gate_function(
                "swap",
                (Register("x", quantum_bit_type()), Register("y", quantum_bit_type())),
                CircuitIR(2, (Instruction("swap", (0, 1)),)),
            ),
        )
    )
    call_through_word_program(
        builder, words, h, {"q": (q[0],)}, nominal_types=nominal_types
    )
    for i in range(1, width):
        call_through_word_program(
            builder,
            words,
            phase_gradient_unitary_program(i, 0.5, controlled=True),
            {"ctrl": (q[i],), "phase_grad": q[:i][::-1]},
            nominal_types=nominal_types,
        )
        call_through_word_program(
            builder, words, h, {"q": (q[i],)}, nominal_types=nominal_types
        )
    if with_reverse:
        for i in range(width // 2):
            call_through_word_program(
                builder,
                words,
                swap,
                {"x": (q[i],), "y": (q[-i - 1],)},
                nominal_types=nominal_types,
            )
    return builder.finalize(
        {
            port.name: gather_word_binding(
                builder,
                words,
                positions[port.name],
                port.dtype,
                nominal_types=nominal_types,
            )
            for port in signature
        }
    )
