"""Checked immutable classical IndexedPauli arrays at helper boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .syntax import Expression, reject

if TYPE_CHECKING:
    from .adapter import _Adapt


def bind_indexed_pauli(
    owner: _Adapt, argument: Expression
) -> tuple[tuple[int, int], ...]:
    if argument.token.text != "IndexedPauli[]":
        binding = owner.indexed_pauli.get(argument.token.text)
        if argument.children or binding is None:
            reject(
                argument.token, "pauli", "helper requires a static IndexedPauli array"
            )
        return binding

    if not 1 <= len(argument.children) <= 256:
        reject(argument.token, "pauli", "IndexedPauli arrays require 1..256 entries")
    entries = []
    for record in argument.children:
        axis, offset = record.children
        index = owner.static_integer(offset)
        if not 0 <= index < 256:
            reject(offset.token, "pauli", "IndexedPauli qubit index must be in 0..255")
        entries.append(("IXYZ".index(axis.token.text), index))
    return tuple(entries)


def indexed_pauli_member(
    owner: _Adapt, node: Expression, array_access: Expression, member: str
) -> int | None:
    if member not in {"index", "pauli"} or array_access.token.text != "[":
        return None
    array, offset = array_access.children
    entries = owner.indexed_pauli.get(array.token.text)
    if array.children or entries is None:
        return None
    index = owner.static_integer(offset)
    if not 0 <= index < len(entries):
        reject(node.token, "binding", "IndexedPauli array index is out of bounds")
    return entries[index][0 if member == "pauli" else 1]
