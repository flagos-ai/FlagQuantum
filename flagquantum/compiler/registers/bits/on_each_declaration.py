# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 basic_gates/on_each.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/on_each.py

"""Retain repeated-child metadata independently of finite physical binding."""

from dataclasses import dataclass

from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_types import opaque_register_type
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt
from .on_each import on_each_word_program


@dataclass(frozen=True)
class OnEachDeclaration:
    """Own the original child-port checks, target default and multiplicity.

    Constructor checks retain source shape and direction metadata. Physical
    construction delegates to the existing scalar one-bit quantum THRU adapter. Child names
    are identities in the enclosing native component namespace.
    """

    n: SymbolicInt
    child: FunctionRecord
    target_dtype: RegisterType | None = None

    def __post_init__(self) -> None:
        registers = self.child.registers
        if registers is None or len(registers) != 1:
            raise ValueError("OnEach child requires exactly one register")
        if registers[0].bitsize != 1:
            raise ValueError("OnEach child register must have width one")
        if registers[0].name != "q":
            raise ValueError("OnEach child register must be named q")
        if (
            self.target_dtype is not None
            and self.target_dtype.encoding != "bit"
            and self.target_dtype.num_bits != self.n
        ):
            raise ValueError("OnEach target dtype must match the repetition width")

    @property
    def signature(self) -> Signature:
        dtype = (
            opaque_register_type(self.n)
            if self.target_dtype is None
            else self.target_dtype
        )
        return Signature((Register("q", dtype),))

    def direct_calls(self) -> dict[str, SymbolicInt]:
        return {self.child.name: self.n}

    def program(self) -> ProgramRecord:
        """Reuse the finite body with ordered calls and caller-owned storage."""
        if type(self.n) is not int:
            raise ValueError("OnEach physical construction requires a concrete width")
        return on_each_word_program(self.n, self.child, target_dtype=self.target_dtype)
