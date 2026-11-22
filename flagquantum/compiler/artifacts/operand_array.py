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
    _repeated: bool

    def __init__(self, operands: Iterable[OperandIdentity] = ()) -> None:
        owner = tuple(operands)
        if any(not isinstance(item, OperandIdentity) for item in owner):
            raise ValueError("Operand arrays require typed identities")
        common_type = _compute_common_type(owner)
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_positions", range(len(owner)))
        object.__setattr__(self, "common_type", common_type)
        object.__setattr__(self, "_repeated", False)

    @property
    def offset(self) -> int:
        return 0 if self._repeated else self._positions.start

    @property
    def stride(self) -> int:
        return 0 if self._repeated else self._positions.step

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
            object.__setattr__(view, "_repeated", self._repeated)
            return view
        if not isinstance(index, int):
            raise IndexError("Operand array index must be an integer or slice")
        position = self._positions[index]
        return self._owner[0 if self._repeated else position]

    def __iter__(self) -> Iterator[OperandIdentity]:
        return (
            self._owner[0 if self._repeated else index] for index in self._positions
        )

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

    def _view(
        self, offset: int, stride: int, count: int, common_type: int
    ) -> OperandArray:
        """Retain an owner even for empty or zero-stride source views."""
        view = object.__new__(OperandArray)
        positions = (
            range(count)
            if stride == 0
            else range(offset, offset + stride * count, stride)
        )
        object.__setattr__(view, "_owner", self._owner)
        object.__setattr__(view, "_positions", positions)
        object.__setattr__(view, "common_type", common_type)
        object.__setattr__(view, "_repeated", stride == 0)
        return view

    @classmethod
    def repeat_false(cls, count: int) -> OperandArray:
        """Repeat one immutable false cell using constant storage."""
        _validate_count(count)
        return cls((OperandIdentity(OperandKind.BOOLEAN, 0),))._view(0, 0, count, 0)

    def reversed(self) -> OperandArray:
        """Reverse the physical view, preserving its static type hint."""
        return self._view(
            self.offset + self.stride * (len(self) - 1),
            -self.stride,
            len(self),
            self.common_type,
        )

    def skip(self, count: int) -> OperandArray:
        """Discard a clamped prefix without copying storage."""
        _validate_count(count)
        count = min(count, len(self))
        return self._view(
            self.offset + count * self.stride,
            self.stride,
            len(self) - count,
            self.common_type,
        )

    def keep(self, count: int) -> OperandArray:
        """Retain a clamped prefix without copying storage."""
        _validate_count(count)
        return self._view(
            self.offset, self.stride, min(count, len(self)), self.common_type
        )

    def skip_last(self, count: int) -> OperandArray:
        """Discard a clamped suffix without copying storage."""
        _validate_count(count)
        return self.keep(len(self) - min(count, len(self)))

    def keep_last(self, count: int) -> OperandArray:
        """Retain a clamped suffix without copying storage."""
        _validate_count(count)
        return self.skip(len(self) - min(count, len(self)))


def _validate_count(count: int) -> None:
    if type(count) is not int or count < 0:
        raise ValueError("Operand view count must be a nonnegative integer")


def _compute_common_type(operands: Iterable[OperandIdentity]) -> int:
    kinds = {item.kind for item in operands}
    if OperandKind.REGISTER in kinds:
        raise ValueError("Operand arrays cannot contain register identities")
    if not kinds:
        return int(OperandKind.QUANTUM)
    if len(kinds) == 1:
        return int(next(iter(kinds)))
    if not (kinds & {OperandKind.X_CLASSICAL, OperandKind.X_BOOLEAN}):
        return 128
    if not (kinds & {OperandKind.CLASSICAL, OperandKind.BOOLEAN}):
        return 64
    return 192
