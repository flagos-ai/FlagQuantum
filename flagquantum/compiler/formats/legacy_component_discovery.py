# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Library traversal adapted from Qualtran v0.7.0 serialization/bloq.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/serialization/bloq.py

"""Discover and number legacy wire definitions from an ordered component catalog."""

from collections.abc import Callable, Iterable
from dataclasses import replace

from ...errors import SerializationError
from .legacy_component_arguments import ComponentReference
from .legacy_component_library import (
    ComponentConnection,
    ComponentLibrary,
    ComponentLibraryEntry,
    ComponentOccurrence,
    ComponentPort,
)


def _ordered_occurrences(
    connections: Iterable[ComponentConnection],
) -> tuple[ComponentOccurrence, ...]:
    seen: dict[int, ComponentOccurrence] = {}
    for connection in connections:
        for port in (connection.source, connection.destination):
            owner = port.owner
            if isinstance(owner, ComponentOccurrence):
                previous = seen.setdefault(owner.instance_id, owner)
                if previous != owner:
                    raise SerializationError(
                        f"Conflicting definitions for occurrence {owner.instance_id}"
                    )
    return tuple(seen.values())


def _entry_index(
    entries: Iterable[ComponentLibraryEntry],
) -> dict[int, ComponentLibraryEntry]:
    index = {}
    for entry in entries:
        if entry.identifier in index:
            raise SerializationError(
                f"Duplicate component definition {entry.identifier}"
            )
        index[entry.identifier] = entry
    return index


def _always_include(_: ComponentOccurrence) -> bool:
    return True


def discover_component_library(
    catalog: ComponentLibrary,
    roots: Iterable[int],
    *,
    max_depth: int = 1,
    predicate: Callable[[ComponentOccurrence], bool] = _always_include,
) -> ComponentLibrary:
    """Assign IDs in discovery order and emit the selected legacy library.

    Catalog rows, connections, count dependencies, and argument references
    use caller-supplied definition identities. The result has dense wire IDs.
    Count dependencies follow their supplied declaration order. Depth and
    the predicate limit recursive decomposition. Count and argument targets
    still receive shallow rows when encountered.
    """
    index = _entry_index(catalog.entries)
    ids: dict[int, int] = {}
    full: set[int] = set()
    visiting: set[tuple[int, int]] = set()

    def visit(identifier: int, depth: int) -> None:
        if identifier not in index:
            raise SerializationError(f"Unknown component definition {identifier}")
        ids.setdefault(identifier, len(ids))
        depth = max(depth, 0)
        key = (identifier, depth)
        if key in visiting:
            raise SerializationError(f"Cyclic component discovery at {identifier}")
        visiting.add(key)
        entry = index[identifier]
        if depth:
            full.add(identifier)
            for occurrence in _ordered_occurrences(entry.connections):
                visit(
                    occurrence.definition_id,
                    depth - 1 if predicate(occurrence) else 0,
                )
            for child, _ in entry.counts:
                visit(child, 0)
        for argument in entry.declaration.arguments:
            if isinstance(argument.value, ComponentReference):
                visit(argument.value.identifier, 0)
        visiting.remove(key)

    for root in roots:
        visit(root, max_depth)

    def remap_port(port: ComponentPort) -> ComponentPort:
        owner = port.owner
        if isinstance(owner, ComponentOccurrence):
            return replace(
                port, owner=replace(owner, definition_id=ids[owner.definition_id])
            )
        return port

    rows = []
    for identifier, new_id in ids.items():
        entry = index[identifier]
        is_full = identifier in full
        declaration = replace(
            entry.declaration,
            arguments=tuple(
                (
                    replace(
                        argument,
                        value=ComponentReference(ids[argument.value.identifier]),
                    )
                    if isinstance(argument.value, ComponentReference)
                    else argument
                )
                for argument in entry.declaration.arguments
            ),
            annotation=entry.declaration.annotation if is_full else None,
        )
        connections = (
            tuple(
                ComponentConnection(
                    remap_port(edge.source), remap_port(edge.destination)
                )
                for edge in entry.connections
            )
            if is_full
            else ()
        )
        counts = (
            tuple((ids[child], count) for child, count in entry.counts)
            if is_full
            else ()
        )
        rows.append(ComponentLibraryEntry(new_id, declaration, connections, counts))
    return ComponentLibrary(catalog.name, tuple(rows))
