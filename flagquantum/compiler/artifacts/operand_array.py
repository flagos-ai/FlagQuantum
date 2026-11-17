# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix operand array bindings.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/py/val/array.pybind.cc

"""Immutable typed operand storage with composable strided views."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import overload

from .operand_identity import OperandIdentity, OperandKind


@dataclass(frozen=True, eq=False, init=False, repr=False)
class OperandArray(Sequence[OperandIdentity]):
    """Own dense operands or retain a view into their original storage."""

    _owner: tuple[OperandIdentity, ...]
    _positions: range
    common_type: int

    def __init__(self, operands: Iterable[OperandIdentity] = ()) -> None:
        owner = tuple(operands)
        kinds = {item.kind for item in owner}
        if OperandKind.REGISTER in kinds:
            raise ValueError("Operand arrays cannot contain register identities")
        if not kinds:
            common_type = int(OperandKind.QUANTUM)
        elif len(kinds) == 1:
            common_type = int(next(iter(kinds)))
        elif not (kinds & {OperandKind.X_CLASSICAL, OperandKind.X_BOOLEAN}):
            common_type = 128
        elif not (kinds & {OperandKind.CLASSICAL, OperandKind.BOOLEAN}):
            common_type = 64
        else:
            common_type = 192
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_positions", range(len(owner)))
        object.__setattr__(self, "common_type", common_type)

    @property
    def offset(self) -> int:
        return self._positions.start

    @property
    def stride(self) -> int:
        return self._positions.step

    def __len__(self) -> int:
        return len(self._positions)

    @overload
    def __getitem__(self, index: int) -> OperandIdentity: ...

    @overload
    def __getitem__(self, index: slice) -> OperandArray: ...

    def __getitem__(self, index: int | slice) -> OperandIdentity | OperandArray:
        if isinstance(index, slice):
            view = object.__new__(OperandArray)
            object.__setattr__(view, "_owner", self._owner)
            object.__setattr__(view, "_positions", self._positions[index])
            object.__setattr__(view, "common_type", self.common_type)
            return view
        if not isinstance(index, int):
            raise IndexError("Operand array index must be an integer or slice")
        return self._owner[self._positions[index]]

    def __iter__(self) -> Iterator[OperandIdentity]:
        return (self._owner[index] for index in self._positions)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OperandArray):
            return False
        return len(self) == len(other) and all(
            left == right for left, right in zip(self, other, strict=True)
        )

    def __add__(self, other: OperandArray) -> OperandArray:
        if not isinstance(other, OperandArray):
            raise TypeError("Concatenation requires an operand array")
        return OperandArray((*self, *other))

    def __repr__(self) -> str:
        return f"OperandArray({tuple(self)!r})"
