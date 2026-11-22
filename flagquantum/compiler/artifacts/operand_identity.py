# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix tagged operand IDs.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/py/val/id.pybind.cc

"""Immutable identities for physical operands before register binding."""

from dataclasses import dataclass
from enum import IntEnum
from types import NotImplementedType

from ._operand_tags import PAYLOAD_MASK


class OperandKind(IntEnum):
    """Operand order follows the locked source's type tags."""

    BOOLEAN = 0
    CLASSICAL = 1
    QUANTUM = 2
    X_BOOLEAN = 3
    X_CLASSICAL = 4
    REGISTER = 5


@dataclass(frozen=True)
class OperandIdentity:
    """Keep equal numeric payloads distinct across operand families."""

    kind: OperandKind
    value: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, OperandKind) or type(self.value) is not int:
            raise ValueError("Operand identity requires a kind and integer payload")
        if self.kind in (OperandKind.BOOLEAN, OperandKind.X_BOOLEAN):
            limit = 2
        elif self.kind is OperandKind.REGISTER:
            limit = 1 << 32
        else:
            limit = PAYLOAD_MASK + 1
        if not 0 <= self.value < limit:
            raise ValueError("Operand identity is outside its family range")

    def __lt__(self, other: object) -> bool | NotImplementedType:
        if not isinstance(other, OperandIdentity):
            return NotImplemented
        if (self.kind is OperandKind.REGISTER) != (other.kind is OperandKind.REGISTER):
            return NotImplemented
        return (self.kind, self.value) < (other.kind, other.value)

    def __bool__(self) -> bool:
        if self.kind in (OperandKind.BOOLEAN, OperandKind.X_BOOLEAN):
            return bool(self.value)
        return True

    def __repr__(self) -> str:
        return f"OperandIdentity(OperandKind.{self.kind.name}, {self.value})"

    @property
    def physical_word(self) -> int:
        """Return the exact uint32 storage word, including scalar family tags.

        Register identities use their plain uint32 index. Their native kind
        discriminant is outside the physical scalar tag vocabulary.
        """
        if self.kind is OperandKind.REGISTER:
            return self.value
        return (int(self.kind) << 29) | self.value

    @property
    def diagnostic_text(self) -> str:
        """Render the original scalar diagnostic spelling without changing repr."""
        if self.kind is OperandKind.BOOLEAN:
            return str(bool(self.value)).lower()
        if self.kind is OperandKind.X_BOOLEAN:
            return f"XBool({str(bool(self.value)).lower()})"
        prefix = {
            OperandKind.CLASSICAL: "b",
            OperandKind.QUANTUM: "q",
            OperandKind.X_CLASSICAL: "xb",
            OperandKind.REGISTER: "r",
        }[self.kind]
        return f"{prefix}{self.value}"

    def conjugated_by_h(self) -> "OperandIdentity":
        """Exchange the basis of a classical identity or Boolean ket value."""
        conjugate = {
            OperandKind.BOOLEAN: OperandKind.X_BOOLEAN,
            OperandKind.CLASSICAL: OperandKind.X_CLASSICAL,
            OperandKind.X_BOOLEAN: OperandKind.BOOLEAN,
            OperandKind.X_CLASSICAL: OperandKind.CLASSICAL,
        }.get(self.kind)
        if conjugate is None:
            raise ValueError("H-basis identity conversion requires a classical value")
        return OperandIdentity(conjugate, self.value)
