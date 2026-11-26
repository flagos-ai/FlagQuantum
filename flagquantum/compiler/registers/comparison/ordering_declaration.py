# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 arithmetic/sorting.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/sorting.py

"""Declare compare/exchange ports and offset counts before physical binding."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from functools import cached_property

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt
from ....core.value_parameters import is_symbolic_value
from .compare_exchange_component import compare_exchange_program
from .parallel_comparison_component import parallel_comparison_program


@dataclass(frozen=True)
class ComparatorDeclaration:
    """Retain two unsigned THRU words and the retained RIGHT comparison bit."""

    width: SymbolicInt

    @property
    def signature(self) -> Signature:
        word = unsigned_register_type(self.width)
        return Signature(
            (
                Register("a", word),
                Register("b", word),
                Register("out", quantum_bit_type(), side=RegisterSide.RIGHT),
            )
        )

    def direct_calls(self, *, adjoint: bool = False) -> dict[Hashable, SymbolicInt]:
        return {
            ("free_bit" if adjoint else "allocate_bit",): 1,
            (
                "greater_than_adjoint" if adjoint else "greater_than",
                self.width,
                self.width,
            ): 1,
            ("controlled_swap", self.width): 1,
        }

    def program(self, *, adjoint: bool = False) -> ProgramRecord:
        """Delegate concrete widths to the existing directional comparison owner."""
        if type(self.width) is not int:
            raise ValueError("Comparator execution requires a concrete width")
        return compare_exchange_program(self.width, adjoint=adjoint)


@dataclass(frozen=True)
class ParallelComparatorsDeclaration:
    """Preserve symbolic offset bounds independently of finite block traversal."""

    count: SymbolicInt
    offset: SymbolicInt
    width: SymbolicInt

    def is_symbolic(self) -> bool:
        return is_symbolic_value(self.width, self.count, self.offset)

    @cached_property
    def num_comparisons(self) -> SymbolicInt:
        if is_symbolic_value(self.count, self.offset):
            return self.count // 2
        full = (self.count // (2 * self.offset)) * self.offset
        rest = self.count % (2 * self.offset)
        return full + max(rest - self.offset, 0)

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            (
                Register("xs", unsigned_register_type(self.width), (self.count,)),
                Register(
                    "junk",
                    quantum_bit_type(),
                    (self.num_comparisons,),
                    side=RegisterSide.RIGHT,
                ),
            )
        )

    def direct_calls(self) -> dict[Hashable, SymbolicInt]:
        return {("comparator", self.width): self.num_comparisons}

    def program(self, *, adjoint: bool = False) -> ProgramRecord:
        """Keep finite traversal, retained history and temporary work in one owner."""
        if any(
            type(value) is not int for value in (self.count, self.offset, self.width)
        ):
            raise ValueError(
                "Parallel comparison execution requires concrete dimensions"
            )
        return parallel_comparison_program(
            self.count, self.offset, self.width, adjoint=adjoint
        )
