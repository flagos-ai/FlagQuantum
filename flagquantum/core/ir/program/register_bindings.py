# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Indexed metadata adapted from Qualtran v0.7.0 Soquet.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/quantum_graph.py

"""Indexed register metadata over existing SSA identities.

Proposal 103 prerequisite 0.11 defines this association. A binding alone does
not admit a register-valued operation or prove ownership or quantum cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import ValueRef
from .registers import Register


@dataclass(frozen=True, init=False)
class RegisterBinding:
    """An existing value associated with one indexed logical register element."""

    value: ValueRef
    register: Register
    index: tuple[int, ...]

    def __init__(
        self,
        value: ValueRef,
        register: Register,
        index: int | tuple[int, ...] = (),
    ) -> None:
        indices = (index,) if isinstance(index, int) else index
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "register", register)
        object.__setattr__(self, "index", indices)
        if len(indices) != len(register.shape):
            raise ValueError(f"Bad index shape {indices} for {register}.")
        for element, size in zip(indices, register.shape, strict=False):
            if element >= size:
                raise ValueError(f"Bad index {element} for {register}.")
