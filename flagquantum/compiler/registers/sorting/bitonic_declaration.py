# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 arithmetic/sorting.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/sorting.py

"""Declare merge/sort shapes and symbolic histories before finite execution."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from functools import cached_property

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt, bit_length
from ....core.value_parameters import is_symbolic_value
from .bitonic_merge_component import bitonic_merge_program
from .bitonic_sort_component import bitonic_sort_program


@dataclass(frozen=True)
class BitonicMergeDeclaration:
    """Consume two LEFT arrays and return their RIGHT result plus logical history."""

    half_length: SymbolicInt
    width: SymbolicInt

    def __post_init__(self) -> None:
        if not is_symbolic_value(self.half_length) and self.half_length < 1:
            raise ValueError("Length of input lists must be positive")

    @cached_property
    def num_comparisons(self) -> SymbolicInt:
        return self.half_length * (bit_length(self.half_length - 1) + 1)

    @cached_property
    def signature(self) -> Signature:
        word = unsigned_register_type(self.width)
        return Signature(
            (
                Register("xs", word, (self.half_length,), side=RegisterSide.LEFT),
                Register("ys", word, (self.half_length,), side=RegisterSide.LEFT),
                Register(
                    "result", word, (2 * self.half_length,), side=RegisterSide.RIGHT
                ),
                Register(
                    "junk",
                    quantum_bit_type(),
                    (self.num_comparisons,),
                    side=RegisterSide.RIGHT,
                ),
            )
        )

    def is_symbolic(self) -> bool:
        return is_symbolic_value(self.width, self.half_length)

    def direct_calls(self) -> dict[Hashable, SymbolicInt]:
        return {("comparator", self.width): self.num_comparisons}

    def program(self, *, adjoint: bool = False) -> ProgramRecord:
        """Delegate known power-of-two halves to the accepted merge hierarchy."""
        if type(self.half_length) is not int or type(self.width) is not int:
            raise ValueError("Bitonic merge execution requires concrete dimensions")
        return bitonic_merge_program(self.half_length, self.width, adjoint=adjoint)


@dataclass(frozen=True)
class BitonicSortDeclaration:
    """Retain a THRU array and the RIGHT recursive comparison history."""

    count: SymbolicInt
    width: SymbolicInt

    def __post_init__(self) -> None:
        if not is_symbolic_value(self.count):
            if self.count < 1:
                raise ValueError("Length of input list must be positive")
            if self.count & (self.count - 1):
                raise ValueError("Length of input list must be a power of two")

    @cached_property
    def num_comparisons(self) -> SymbolicInt:
        log = bit_length(self.count - 1)
        return (self.count // 2) * ((log * (log + 1)) // 2)

    @property
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

    def is_symbolic(self) -> bool:
        return is_symbolic_value(self.width, self.count)

    def direct_calls(self) -> dict[Hashable, SymbolicInt]:
        return {("comparator", self.width): self.num_comparisons}

    def program(self, *, adjoint: bool = False) -> ProgramRecord:
        """Delegate concrete recursion and inverse history consumption unchanged."""
        if type(self.count) is not int or type(self.width) is not int:
            raise ValueError("Bitonic sort execution requires concrete dimensions")
        return bitonic_sort_program(self.count, self.width, adjoint=adjoint)
