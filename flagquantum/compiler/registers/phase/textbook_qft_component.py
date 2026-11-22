# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 qft/qft_text_book.py.

"""Retain textbook Fourier calls and source-ordered unsigned word views."""

from dataclasses import replace

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_program
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..component_graph import _namespace_helpers
from ..component_rebindings import adjoint_word_rebinding
from ..layout.component_word_views import call_through_word_program, gather_word_binding
from .phase_gradient_component import phase_gradient_unitary_program


def textbook_qft_program(
    width: int, *, with_reverse: bool = True, adjoint: bool = False
) -> ProgramRecord:
    """Transform caller-owned MSB-first q, with private work owned by CZ powers."""
    if type(width) is not int or width < 1 or type(with_reverse) is not bool:
        raise ValueError("QFT requires a positive concrete width and boolean reversal")
    if type(adjoint) is not bool:
        raise ValueError("QFT adjoint must be boolean")
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
        # Inner swaps share a source Cirq moment, before the final outer swap.
        swaps = list(range(width // 2))
        for i in swaps[1:] + swaps[:1]:
            call_through_word_program(
                builder,
                words,
                swap,
                {"x": (q[i],), "y": (q[-i - 1],)},
                nominal_types=nominal_types,
            )
    record = builder.finalize(
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
    if not adjoint:
        return record
    definitions = list(record.functions)
    adjoints = {
        h.functions[0].name: h.functions[0].name,
        swap.functions[0].name: swap.functions[0].name,
    }
    callees = {op.callee for op in record.functions[0].blocks[0].operations}
    for position, function in enumerate(record.functions[1:]):
        if function.name not in callees:
            continue
        ports = function.registers or ()
        names = tuple(port.name for port in ports)
        if names == ("reg", "reg"):
            inverse = adjoint_word_rebinding(function, f"qft_inverse_view_{position}")
            definitions.append(inverse)
            adjoints[function.name] = inverse.name
        elif names == ("ctrl", "phase_grad"):
            inverse_record = _namespace_helpers(
                phase_gradient_unitary_program(
                    ports[1].bitsize, 0.5, controlled=True, adjoint=True
                ),
                f"qft_inverse_gradient_{position}",
            )
            definitions.extend(inverse_record.functions)
            adjoints[function.name] = inverse_record.functions[0].name
    return adjoint_component_program(
        replace(record, functions=tuple(definitions)), adjoints
    )
