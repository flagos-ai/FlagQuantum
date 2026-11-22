# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 qft/approximate_qft.py.

"""Retain bounded gradient-addition slices of the original approximate QFT."""

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import (
    fixed_point_register_type,
    quantum_bit_type,
)
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..layout.component_word_views import call_through_word_program, gather_word_binding
from .scaled_phase_component import phase_add_component_program


def approximate_qft_program(
    width: int, precision: int, *, with_reverse: bool = True
) -> ProgramRecord:
    """Preserve caller gradient and q, with measured work owned by shared CAdd."""
    if type(width) is not int or width < 1 or type(with_reverse) is not bool:
        raise ValueError("QFT requires a positive concrete width and boolean reversal")
    if type(precision) is not int or precision < 2:
        raise ValueError("Stored approximate QFT requires at least two gradient bits")
    signature = Signature.build_from_dtypes(
        q=RegisterType("uint", width),
        phase_grad=fixed_point_register_type(precision, precision),
    )
    builder = ComponentProgramBuilder(signature)
    q, gradient = tuple(range(width)), tuple(range(width, width + precision))
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
    for i in range(width):
        if i:
            active = min(i, precision - 1)
            call_through_word_program(
                builder,
                words,
                phase_add_component_program(
                    active,
                    active + 1,
                    right_shift=1,
                    controlled=True,
                ),
                {
                    "ctrl": (q[i],),
                    "x": q[i - active : i][::-1],
                    "phase_grad": gradient[: active + 1],
                },
                nominal_types=nominal_types,
            )
        call_through_word_program(
            builder, words, h, {"q": (q[i],)}, nominal_types=nominal_types
        )
    if with_reverse:
        # Cirq EARLIEST places swaps disjoint from the last H in that moment.
        # The swap touching its last data bit follows in the next moment.
        swaps = list(range(width // 2))
        for i in swaps[1:] + swaps[:1]:
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
