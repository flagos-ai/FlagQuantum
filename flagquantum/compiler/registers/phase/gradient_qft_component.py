# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 qft/qft_phase_gradient.py.

"""Compose the original recursive gradient QFT over caller-owned word slices."""

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import (
    fixed_point_register_type,
    quantum_bit_type,
)
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..arithmetic.product_accumulation_component import product_accumulation_program
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..layout.component_word_views import call_through_word_program, gather_word_binding


def gradient_qft_program(width: int, *, with_reverse: bool = True) -> ProgramRecord:
    """Keep q and gradient live while shared products own their clean workspace."""
    if type(width) is not int or width < 1 or type(with_reverse) is not bool:
        raise ValueError("QFT requires a positive concrete width and boolean reversal")
    signature = Signature.build_from_dtypes(
        q=RegisterType("uint", width),
        phase_grad=fixed_point_register_type(width, width),
    )
    builder = ComponentProgramBuilder(signature)
    q, gradient = tuple(range(width)), tuple(range(width, 2 * width))
    positions = {"q": q, "phase_grad": gradient}
    words: dict[tuple[int, ...], RegisterBinding] = {}
    for name, wires in positions.items():
        value = builder.inputs[name]
        assert isinstance(value, RegisterBinding)
        words[wires] = value
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
    if width == 1:
        call_through_word_program(
            builder, words, h, {"q": q}, nominal_types=nominal_types
        )
    else:
        half = width // 2
        a, b = q[:half], q[half:]
        call_through_word_program(
            builder,
            words,
            gradient_qft_program(len(a), with_reverse=False),
            {"q": a, "phase_grad": gradient[: len(a)]},
            nominal_types=nominal_types,
        )
        call_through_word_program(
            builder,
            words,
            product_accumulation_program(len(a), len(b), width),
            {"a": a[::-1], "b": b, "result": gradient},
            nominal_types=nominal_types,
        )
        call_through_word_program(
            builder,
            words,
            gradient_qft_program(len(b), with_reverse=False),
            {"q": b, "phase_grad": gradient[: len(b)]},
            nominal_types=nominal_types,
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
