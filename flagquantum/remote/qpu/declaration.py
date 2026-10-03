"""Fail-closed reading of a vendor's own device declaration.

Every provider adapter in this package starts from the same input: one row, or
one object, from a vendor's device listing. That input is external and
untrusted, and vendors rename fields between API versions, so each adapter must
look a fact up under every spelling its API has used and must refuse a
declaration that omits or mistypes a fact the backend profile depends on.
Sharing the readers keeps those refusals identical in wording and keeps one
definition of what a usable declaration is.

Connectivity is returned as normalised undirected edges rather than as a
Compiler ``CouplingMap``, so this module carries no Compiler dependency; an
adapter makes the ``CouplingMap`` with one constructor call, which is the same
division the ecosystem target SDK uses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_MISSING = object()


def declared_value(declaration: Any, *names: str, default: Any = _MISSING) -> Any:
    """Return the first declared spelling of one fact.

    A vendor listing is variously a mapping or a client object. ``None`` counts
    as absent, because a vendor that reports a null capacity has not declared
    one.
    """

    for name in names:
        if isinstance(declaration, Mapping):
            found = declaration.get(name, _MISSING)
        else:
            found = getattr(declaration, name, _MISSING)
        if found is not _MISSING and found is not None:
            return found
    if default is not _MISSING:
        return default
    raise KeyError(f"declaration exposes none of {', '.join(names)}")


def declared_text(declaration: Any, *names: str, owner: str, fact: str) -> str:
    """Return a required non-empty string fact, or refuse the declaration."""

    value = declared_value(declaration, *names, default=None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner} declaration does not expose a {fact}")
    return value.strip()


def declared_count(declaration: Any, *names: str, owner: str, fact: str) -> int:
    """Return a required positive integer fact, or refuse the declaration.

    A boolean is refused even though Python makes it an integer, because a
    listing that answers a width with ``true`` has not answered it.
    """

    value = declared_value(declaration, *names, default=None)
    if value is None:
        raise ValueError(f"{owner} declaration does not expose a {fact}")
    if type(value) is not int or value <= 0:
        raise ValueError(f"{owner} {fact} must be a positive integer")
    return value


def declared_width(
    declaration: Any, *names: str, owner: str, fact: str = "qubit count"
) -> tuple[int, str]:
    """Return a declared width and how the vendor stated it.

    A device listing states its width either as a count or as the list of its
    qubits, and some listings state both under one name. Both are read here, and
    which one answered is reported so a caller can record the provenance. A
    listing that answers the width with a list carries no width of its own until
    the list is counted, which is exactly what this does.
    """

    raw = declared_value(declaration, *names, default=None)
    if raw is None:
        raise ValueError(f"{owner} declaration does not expose a {fact}")
    if (
        isinstance(raw, bool)
        or not isinstance(raw, int | Sequence)
        or isinstance(raw, str | bytes)
    ):
        raise ValueError(
            f"{owner} {fact} must be a positive integer or a list of qubits"
        )
    if isinstance(raw, int):
        if raw <= 0:
            raise ValueError(f"{owner} {fact} must be a positive integer")
        return raw, "declared_count"
    if not raw:
        raise ValueError(f"{owner} {fact} must be a positive integer")
    for index, entry in enumerate(raw):
        if (
            isinstance(entry, str | bytes)
            or not isinstance(entry, Sequence)
            or not entry
        ):
            raise ValueError(f"{owner} {fact} entry {index} must be a qubit coordinate")
    return len(raw), "declared_qubit_list"


def declared_flag(
    declaration: Any, *names: str, owner: str, fact: str, default: bool
) -> bool:
    """Return a declared boolean fact, refusing a value that is not a boolean.

    The default applies only when the listing is silent, so a caller can tell a
    vendor's own answer from this package's fallback.
    """

    value = declared_value(declaration, *names, default=None)
    if value is None:
        return default
    if type(value) is not bool:
        raise ValueError(f"{owner} {fact} must be a boolean")
    return value


def declared_gate_names(
    raw: Any, *, owner: str, fact: str = "native gate set"
) -> tuple[str, ...]:
    """Normalise a declared gate list, refusing an entry that is not a name."""

    if raw is None:
        return ()
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
        raise ValueError(f"{owner} {fact} must be a sequence of gate names")
    names: list[str] = []
    for entry in raw:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(f"{owner} {fact} entries must be non-empty gate names")
        name = entry.strip().lower()
        if name not in names:
            names.append(name)
    return tuple(names)


def edge_pair(value: Any, *, owner: str) -> tuple[int, int]:
    """Return one declared coupling as an ordered pair of non-negative wires."""
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{owner} coupling entry must be a pair of wires")
    if len(value) != 2:
        raise ValueError(f"{owner} coupling entry must be a pair of wires")
    left, right = value
    if type(left) is not int or type(right) is not int:
        raise ValueError(f"{owner} coupling entry must be a pair of integers")
    if left < 0 or right < 0:
        raise ValueError(f"{owner} coupling entry must not contain a negative wire")
    return (min(left, right), max(left, right))


def connectivity_edges(
    raw: Any, *, owner: str, n_qubits: int
) -> tuple[tuple[int, int], ...]:
    """Normalise a declared coupling map in either shape a vendor lists.

    A vendor publishes connectivity as a ``{source: [targets]}`` graph or as a
    flat sequence of pairs. Both are accepted, self-loops are dropped because
    they are not couplings, and a wire outside the declared width is refused
    rather than silently truncated.
    """

    edges: set[tuple[int, int]] = set()
    if isinstance(raw, Mapping):
        for source, targets in raw.items():
            if isinstance(targets, str | bytes) or not isinstance(targets, Sequence):
                raise ValueError(f"{owner} connectivity targets must be a sequence")
            for target in targets:
                edges.add(edge_pair((source, target), owner=owner))
    elif isinstance(raw, Sequence) and not isinstance(raw, str | bytes):
        for entry in raw:
            edges.add(edge_pair(entry, owner=owner))
    else:
        raise ValueError(f"{owner} connectivity declaration must be a mapping or pairs")
    couplings = {edge for edge in edges if edge[0] != edge[1]}
    for left, right in couplings:
        if left >= n_qubits or right >= n_qubits:
            raise ValueError(
                f"{owner} coupling entry {(left, right)} is outside a "
                f"{n_qubits}-wire device"
            )
    return tuple(sorted(couplings))


__all__ = (
    "connectivity_edges",
    "declared_count",
    "declared_flag",
    "declared_gate_names",
    "declared_text",
    "declared_value",
    "declared_width",
    "edge_pair",
)
