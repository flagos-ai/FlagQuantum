# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Shared literal split/CNOT/join loops from Qualtran v0.7.0 ec_add.py.

"""Keep the five source word-copy loops in one typed bookkeeping owner."""

from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.registers import RegisterType
from ..component_builder import BindingCollection, ComponentProgramBuilder


def copy_word(
    builder: ComponentProgramBuilder,
    source: BindingCollection,
    target: BindingCollection,
    dtype: RegisterType,
    cnot: str,
) -> tuple[RegisterBinding, RegisterBinding]:
    """XOR corresponding MSB-first bits, preserving literal source call order."""
    assert isinstance(source, RegisterBinding) and isinstance(target, RegisterBinding)
    lhs, rhs = builder.split(source), builder.split(target)
    for i in range(len(lhs)):
        lhs[i], rhs[i] = builder.call_tuple(cnot, {"ctrl": lhs[i], "target": rhs[i]})
    return builder.join(lhs, dtype), builder.join(rhs, dtype)
