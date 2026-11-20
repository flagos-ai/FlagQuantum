# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 arithmetic/permutation.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/permutation.py

"""Own concrete and shape-only cycle declarations before physical binding."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TypeAlias

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import bounded_unsigned_register_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt, bit_length
from ....core.value_parameters import is_symbolic_value
from .cycle_decomposition import partial_permutation_cycles, permutation_cycles
from .permutation_component import permutation_cycle_program, permutation_program


@dataclass(frozen=True)
class CycleShape:
    """Retain dimensions of unknown cycle values for declaration and model use."""

    shape: tuple[SymbolicInt, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple):
            raise TypeError("Cycle shape must be a tuple")

    def is_symbolic(self) -> bool:
        return True


Cycle: TypeAlias = tuple[int, ...] | CycleShape
Cycles: TypeAlias = tuple[Cycle, ...] | CycleShape


def permutation_signature(size: SymbolicInt, width: SymbolicInt) -> Signature:
    """Keep the bound and MSB-first THRU storage descriptor independent of gates."""
    return Signature((Register("x", bounded_unsigned_register_type(width, size)),))


@dataclass(frozen=True, init=False)
class PermutationCycleDeclaration:
    """A known cycle or unknown shaped values, with the original default width."""

    size: SymbolicInt
    cycle: Cycle
    width: SymbolicInt

    def __init__(
        self,
        size: SymbolicInt,
        cycle: Iterable[int] | CycleShape,
        width: SymbolicInt | None = None,
    ) -> None:
        object.__setattr__(self, "size", size)
        object.__setattr__(
            self, "cycle", cycle if isinstance(cycle, CycleShape) else tuple(cycle)
        )
        object.__setattr__(
            self, "width", bit_length(size - 1) if width is None else width
        )

    @cached_property
    def signature(self) -> Signature:
        return permutation_signature(self.size, self.width)

    def direct_calls(self) -> dict[Hashable, SymbolicInt]:
        """Preserve concrete call insertion order or unknown-value multiplicities."""
        if isinstance(self.cycle, CycleShape):
            length = self.cycle.shape[0]
            return {
                ("equals", self.width, "unknown"): length + 1,
                ("controlled_xor", self.width, "unknown"): length,
            }
        calls: Counter[Hashable] = Counter({("allocate_marker",): 1})
        for value, successor in zip(
            self.cycle, (*self.cycle[1:], self.cycle[0]), strict=True
        ):
            calls[("equals", self.width, value)] += 1
            calls[("controlled_xor", self.width, value ^ successor)] += 1
        calls[("equals", self.width, self.cycle[0])] += 1
        calls[("free_marker",)] = 1
        return dict(calls)

    def program(self) -> ProgramRecord:
        """Bind only known finite data using the existing typed cycle owner."""
        if (
            isinstance(self.cycle, CycleShape)
            or type(self.size) is not int
            or type(self.width) is not int
        ):
            raise ValueError(
                "Permutation execution requires concrete size, width and cycle values"
            )
        return permutation_cycle_program(self.size, self.cycle, self.width)


@dataclass(frozen=True, init=False)
class PermutationDeclaration:
    """Preserve ordered cycles, symbolic shapes and source factory filtering."""

    size: SymbolicInt
    cycles: Cycles
    width: SymbolicInt

    def __init__(
        self,
        size: SymbolicInt,
        cycles: Iterable[Iterable[int] | CycleShape] | CycleShape,
        width: SymbolicInt | None = None,
    ) -> None:
        object.__setattr__(self, "size", size)
        object.__setattr__(
            self,
            "cycles",
            (
                cycles
                if isinstance(cycles, CycleShape)
                else tuple(
                    cycle if isinstance(cycle, CycleShape) else tuple(cycle)
                    for cycle in cycles
                )
            ),
        )
        object.__setattr__(
            self, "width", bit_length(size - 1) if width is None else width
        )

    @cached_property
    def signature(self) -> Signature:
        return permutation_signature(self.size, self.width)

    def is_symbolic(self) -> bool:
        return (
            is_symbolic_value(self.size, self.cycles)
            or isinstance(self.cycles, tuple)
            and is_symbolic_value(*self.cycles)
        )

    @classmethod
    def from_dense(cls, values: Sequence[int]) -> "PermutationDeclaration":
        return cls(len(values), permutation_cycles(values))

    @classmethod
    def from_partial(
        cls, size: SymbolicInt, values: Mapping[int, int]
    ) -> "PermutationDeclaration":
        return cls(size, partial_permutation_cycles(values))

    @classmethod
    def from_cycle_lengths(
        cls, size: SymbolicInt, lengths: Iterable[SymbolicInt]
    ) -> "PermutationDeclaration":
        return cls(
            size,
            (
                CycleShape((length,))
                for length in lengths
                if is_symbolic_value(length) or length >= 2
            ),
        )

    def direct_calls(self) -> dict[Hashable, SymbolicInt]:
        """Keep the source unknown-cycle-count worst case and ordered child keys."""
        cycles = (
            (CycleShape((self.size,)),)
            if isinstance(self.cycles, CycleShape)
            else self.cycles
        )
        return dict(
            Counter(
                (
                    "cycle",
                    self.size,
                    self.width,
                    cycle.shape[0] if isinstance(cycle, CycleShape) else cycle,
                )
                for cycle in cycles
            )
        )

    def program(self) -> ProgramRecord:
        """Delegate known finite cycles to their existing physical implementation."""
        if (
            isinstance(self.cycles, CycleShape)
            or type(self.size) is not int
            or type(self.width) is not int
            or any(isinstance(cycle, CycleShape) for cycle in self.cycles)
        ):
            raise ValueError(
                "Permutation execution requires concrete size, width and cycle values"
            )
        cycles = tuple(cycle for cycle in self.cycles if isinstance(cycle, tuple))
        return permutation_program(self.size, cycles, self.width)
