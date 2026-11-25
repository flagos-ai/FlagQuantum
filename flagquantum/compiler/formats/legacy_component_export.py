# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Endpoint projection adapted from Qualtran v0.7.0 serialization/bloq.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/serialization/bloq.py

"""Project verified Program SSA connections into legacy library rows."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace

from ...core.ir.program import ProgramRecord, ValueRef
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ...errors import SerializationError
from ..registers.component_graph import _call_entry
from .legacy_component_library import (
    ComponentConnection,
    ComponentDeclaration,
    ComponentLibrary,
    ComponentLibraryEntry,
    ComponentOccurrence,
    ComponentPort,
    LibraryBoundary,
    _indexed_ports,
)


def component_entry_from_program(
    record: ProgramRecord,
    entry: str,
    identifiers: Mapping[str, int],
    declaration: ComponentDeclaration,
    *,
    counts: tuple[tuple[int, SymbolicInt], ...] = (),
    instance_ids: tuple[int, ...] | None = None,
) -> ComponentLibraryEntry:
    """Export a named finite call graph with its ordered SSA operand edges.

    The caller supplies the legacy identity and argument/cost declarations.
    Register declarations and connections come from verified native Program.
    Instance IDs default to call insertion order. A supplied tuple preserves
    IDs from an external graph, including non-contiguous values.
    """
    function = _call_entry(record, entry)
    assert function.registers is not None
    if declaration.registers not in (None, function.registers):
        raise SerializationError(
            f"Legacy declaration differs from Program entry {entry!r}"
        )
    if len(set(identifiers.values())) != len(identifiers):
        raise SerializationError("Duplicate exported definition identifiers")
    if entry not in identifiers:
        raise SerializationError(f"Missing export identifier for {entry!r}")

    block = function.blocks[0]
    ids = tuple(range(len(block.operations))) if instance_ids is None else instance_ids
    if len(ids) != len(block.operations) or len(set(ids)) != len(ids):
        raise SerializationError("Instance IDs must identify each call exactly once")
    signature = Signature(function.registers)
    producers = dict(
        zip(
            block.arguments,
            _indexed_ports(LibraryBoundary.INPUT, tuple(signature.lefts())),
            strict=True,
        )
    )
    connections: list[ComponentConnection] = []

    def connect(values: tuple[ValueRef, ...], ports: tuple[ComponentPort, ...]) -> None:
        connections.extend(
            ComponentConnection(producers[value], port)
            for value, port in zip(values, ports, strict=True)
        )

    functions = {fn.name: fn for fn in record.functions}
    for call, instance_id in zip(block.operations, ids, strict=True):
        assert call.callee is not None
        if call.callee not in identifiers:
            raise SerializationError(f"Missing export identifier for {call.callee!r}")
        callee = functions[call.callee]
        assert callee.registers is not None
        ports = Signature(callee.registers)
        occurrence = ComponentOccurrence(instance_id, identifiers[call.callee])
        connect(call.operands, _indexed_ports(occurrence, tuple(ports.lefts())))
        producers.update(
            zip(
                call.results,
                _indexed_ports(occurrence, tuple(ports.rights())),
                strict=True,
            )
        )
    assert block.return_values is not None
    connect(
        block.return_values,
        _indexed_ports(LibraryBoundary.OUTPUT, tuple(signature.rights())),
    )
    return ComponentLibraryEntry(
        identifiers[entry],
        replace(declaration, registers=function.registers),
        tuple(connections),
        counts,
    )


def component_library_from_program(
    record: ProgramRecord,
    entry: str,
    declarations: Mapping[str, ComponentDeclaration],
    *,
    name: str = "",
) -> ComponentLibrary:
    """Project one call layer with complete caller-owned constructor metadata.

    Child rows retain their constructor declarations, not an invented expansion
    of their bodies. Symbolic metadata and static data remain in the owned
    library input until parameters are bound and a Program can be constructed.
    """
    function = _call_entry(record, entry)
    identifiers = {entry: 0}
    counts: Counter[int] = Counter()
    for operation in function.blocks[0].operations:
        assert operation.callee is not None
        identifier = identifiers.setdefault(operation.callee, len(identifiers))
        counts[identifier] += 1
    missing = identifiers.keys() - declarations.keys()
    if missing:
        raise SerializationError(
            f"Missing constructor declarations: {sorted(missing)!r}"
        )
    root = component_entry_from_program(
        record, entry, identifiers, declarations[entry], counts=tuple(counts.items())
    )
    functions = {item.name: item for item in record.functions}
    return ComponentLibrary(
        name,
        (
            root,
            *(
                ComponentLibraryEntry(
                    identifier,
                    replace(
                        declarations[callee], registers=functions[callee].registers
                    ),
                )
                for callee, identifier in identifiers.items()
                if callee != entry
            ),
        ),
    )
