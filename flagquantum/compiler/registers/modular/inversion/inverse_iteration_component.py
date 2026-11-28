# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _KaliskiIteration.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py

"""Retain the six original inverse steps and the owned temporary branch flags."""

from .....core.ir.program import ProgramRecord
from .....core.ir.program.register_bindings import RegisterBinding
from .....core.ir.program.register_types import (
    montgomery_register_type,
    quantum_bit_type,
)
from .....core.ir.program.registers import RegisterType
from .....core.ir.program.signatures import Signature
from ...component_builder import ComponentProgramBuilder
from .inverse_arithmetic_component import inverse_arithmetic_program
from .inverse_exchange_component import inverse_exchange_program
from .inverse_ordering_component import inverse_ordering_program
from .inverse_parity_component import inverse_parity_program
from .inverse_scaling_component import inverse_scaling_program
from .inverse_termination_component import inverse_termination_program


def inverse_iteration_program(
    width: int, modulus: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Preserve marker/termination history while restoring two internal flags.

    Four MSB-first Montgomery words and m/f/is_terminal are logical data.
    Internal a/b are clean only on source-reachable inputs, or their inverse
    images. Each child owns its measured work and returns its actual aliases.
    """
    word = montgomery_register_type(width, modulus)
    steps = (
        inverse_termination_program(width, adjoint=adjoint),
        inverse_parity_program(width, adjoint=adjoint),
        inverse_ordering_program(width, adjoint=adjoint),
        inverse_exchange_program(width, adjoint=adjoint),
        inverse_arithmetic_program(width, adjoint=adjoint),
        inverse_scaling_program(width, modulus, adjoint=adjoint),
    )
    return inverse_iteration_component_program(word, steps, adjoint=adjoint)


def inverse_iteration_component_program(
    word: RegisterType, steps: tuple[ProgramRecord, ...], *, adjoint: bool = False
) -> ProgramRecord:
    """Retain six explicit source steps and clean branch-flag ownership."""
    if len(steps) != 6 or type(adjoint) is not bool:
        raise ValueError("Inverse iteration requires six steps and a boolean direction")
    bit = quantum_bit_type()
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(
            u=word, v=word, r=word, s=word, m=bit, f=bit, is_terminal=bit
        )
    )
    values = dict(builder.inputs)
    for name in (("b", "a") if adjoint else ("a", "b")):
        values[name] = builder.allocate()
    for step in reversed(steps) if adjoint else steps:
        ports = step.functions[0].registers or ()
        values.update(
            builder.call_program(
                step,
                {r.name: values[r.name] for r in ports},
                entry=step.functions[0].name,
            )
        )
    for name in (("b", "a") if adjoint else ("a", "b")):
        flag = values.pop(name)
        assert isinstance(flag, RegisterBinding)
        builder.free(flag)
    return builder.finalize(values)
