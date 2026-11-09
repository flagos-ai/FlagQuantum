# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Signature.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/registers.py

"""Ordered register signatures with distinct input and output namespaces.

Proposal 103 prerequisite 0.2 admits standalone metadata, not new Program IR
operations or execution support. Numerical codecs and bound construction do not
belong in this module.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from functools import cached_property
from typing import cast, overload

from ...value_math import SymbolicInt as RegisterWidth
from ...value_math import smax, ssum
from ...value_parameters import is_symbolic_value
from .registers import Register, RegisterSide, RegisterType


def _dedupe(registers: Iterable[tuple[str, Register]]) -> dict[str, Register]:
    result: dict[str, Register] = {}
    for name, register in registers:
        if name in result:
            raise ValueError(
                f"Register {name} is specified more than once per side."
            ) from None
        result[name] = register
    return result


def _width_register(name: str, width: RegisterWidth) -> Register:
    if width == 1:
        dtype = RegisterType("bit", 1)
    else:
        # Preserve QAny constructor validation after the convenience filter.
        if not is_symbolic_value(width) and not isinstance(width, int):
            raise ValueError(f"Bad bitsize for opaque register: {width}")
        dtype = RegisterType("opaque", width)
    return Register(name, dtype)


class Signature:
    """A source-ordered collection with names unique independently per side."""

    def __init__(self, registers: Iterable[Register]) -> None:
        self._registers = tuple(registers)
        self._lefts = _dedupe(
            (reg.name, reg) for reg in self._registers if reg.side & RegisterSide.LEFT
        )
        self._rights = _dedupe(
            (reg.name, reg) for reg in self._registers if reg.side & RegisterSide.RIGHT
        )

    @classmethod
    def build(cls, **registers: RegisterWidth) -> Signature:
        """Use bit/opaque through ports, applying the original truth-value filter."""
        return cls(
            _width_register(name, width) for name, width in registers.items() if width
        )

    @classmethod
    def build_from_dtypes(cls, **registers: RegisterType) -> Signature:
        """Retain the source's quantum-width filter, including classical omission."""
        return cls(
            Register(name, dtype)
            for name, dtype in registers.items()
            if dtype.num_qubits
        )

    @cached_property
    def thru_registers_only(self) -> bool:
        for register in self:
            if register.side != RegisterSide.THRU:
                return False
        return True

    def lefts(self) -> Iterator[Register]:
        yield from self._lefts.values()

    def rights(self) -> Iterator[Register]:
        yield from self._rights.values()

    def get_left(self, name: str) -> Register:
        return self._lefts[name]

    def get_right(self, name: str) -> Register:
        return self._rights[name]

    def groups(self) -> Iterator[tuple[str, list[Register]]]:
        groups: dict[str, list[Register]] = defaultdict(list)
        for register in self._registers:
            groups[register.name].append(register)
        yield from groups.items()

    def adjoint(self) -> Signature:
        return Signature(register.adjoint() for register in self._registers)

    def n_qubits(self) -> RegisterWidth:
        left = ssum(register.total_qubits() for register in self.lefts())
        right = ssum(register.total_qubits() for register in self.rights())
        return cast(RegisterWidth, smax(left, right))

    def n_cbits(self) -> RegisterWidth:
        left = ssum(register.total_cbits() for register in self.lefts())
        right = ssum(register.total_cbits() for register in self.rights())
        return cast(RegisterWidth, smax(left, right))

    def n_bits(self) -> RegisterWidth:
        left = ssum(register.total_bits() for register in self.lefts())
        right = ssum(register.total_bits() for register in self.rights())
        return cast(RegisterWidth, smax(left, right))

    def __repr__(self) -> str:
        return f"Signature({self._registers!r})"

    @overload
    def __getitem__(self, key: int) -> Register: ...

    @overload
    def __getitem__(self, key: slice) -> tuple[Register, ...]: ...

    def __getitem__(self, key: int | slice) -> Register | tuple[Register, ...]:
        return self._registers[key]

    def __contains__(self, register: Register) -> bool:
        return register in self._registers

    def __iter__(self) -> Iterator[Register]:
        yield from self._registers

    def __len__(self) -> int:
        return len(self._registers)

    def __hash__(self) -> int:
        return hash(self._registers)

    def __eq__(self, other: object) -> bool:
        # The source does not normalize a foreign object's equality result.
        return cast(bool, self._registers == getattr(other, "_registers"))
