# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Resolution and endpoint binding adapted from Qualtran v0.7.0 serialization/bloq.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/serialization/bloq.py

"""Resolve legacy definitions through explicit native factories and typed Program."""

from collections.abc import Callable, Mapping
from typing import Literal, TypeAlias

from ...core.ir.program import FunctionRecord, ProgramRecord, verify_program_record
from ...core.ir.program.register_bindings import RegisterBinding
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...errors import SerializationError
from .._component_trace.width_schedule import (
    WidthConnection,
    WidthStep,
    schedule_width_steps,
)
from ..registers.component_builder import BindingCollection, ComponentProgramBuilder
from ..registers.component_graph import _collections, _elements
from .legacy_component_arguments import ComponentArgumentValue, ComponentReference
from .legacy_component_discovery import _entry_index, _ordered_occurrences
from .legacy_component_library import (
    ComponentLibrary,
    ComponentLibraryEntry,
    ComponentOccurrence,
    ComponentPort,
    LibraryBoundary,
    _indexed_ports,
)

ResolvedArgument: TypeAlias = "ComponentArgumentValue | FunctionRecord"
ComponentFactory: TypeAlias = Callable[
    [str, tuple[Register, ...] | None, Mapping[str, ResolvedArgument]],
    FunctionRecord | ProgramRecord,
]

_COMPOSITE = "qualtran._infra.composite_bloq.CompositeBloq"
_ALLOCATE = "qualtran.bloqs.bookkeeping.allocate.Allocate"
_FREE = "qualtran.bloqs.bookkeeping.free.Free"
_Owner: TypeAlias = ComponentOccurrence | LibraryBoundary
_StepKind: TypeAlias = Literal["component", "input", "output", "allocate", "free"]


def _step_kind(
    owner: _Owner, entries: Mapping[int, ComponentLibraryEntry]
) -> _StepKind:
    if owner is LibraryBoundary.INPUT:
        return "input"
    if owner is LibraryBoundary.OUTPUT:
        return "output"
    assert isinstance(owner, ComponentOccurrence)
    name = entries[owner.definition_id].declaration.name
    if name == _ALLOCATE:
        return "allocate"
    if name == _FREE:
        return "free"
    return "component"


def _build_composite(
    name: str,
    entry: ComponentLibraryEntry,
    entries: Mapping[int, ComponentLibraryEntry],
    definitions: Mapping[int, FunctionRecord],
    functions: Mapping[str, FunctionRecord],
    *,
    execution_order: tuple[ComponentOccurrence, ...] | None = None,
) -> FunctionRecord:
    registers = entry.declaration.registers
    if registers is None:
        raise SerializationError("Composite component requires a register signature")
    if entry.declaration.name == _COMPOSITE and entry.declaration.arguments:
        raise SerializationError(
            "Composite component cannot have constructor arguments"
        )
    connected = {
        occurrence.definition_id
        for occurrence in _ordered_occurrences(entry.connections)
    }
    if execution_order is not None:
        if len({owner.instance_id for owner in execution_order}) != len(
            execution_order
        ):
            raise SerializationError("Live component order repeats an instance")
        if not set(_ordered_occurrences(entry.connections)).issubset(execution_order):
            raise SerializationError("Live component order omits a connected instance")
        if any(owner.definition_id not in entries for owner in execution_order):
            raise SerializationError(
                "Live component order references an unknown definition"
            )
    for child, count in entry.counts:
        if (
            execution_order is None
            and count != 0
            and child not in connected
            and child in entries
            and entries[child].declaration.registers == ()
        ):
            raise SerializationError(
                "Composite wire omits a zero-port execution child; resource "
                "counts cannot restore its execution order"
            )
    signature = Signature(registers)
    nodes: dict[_Owner, int] = {}
    incoming: dict[ComponentPort, ComponentPort] = {}
    edges = []
    for edge in entry.connections:
        left, right = edge.source, edge.destination
        if right in incoming:
            raise SerializationError(f"Duplicate connection to {right!r}")
        incoming[right] = left
        source = nodes.setdefault(left.owner, len(nodes))
        target = nodes.setdefault(right.owner, len(nodes))
        edges.append(WidthConnection(source, target, left.register.bitsize))
    if execution_order is None:
        order = schedule_width_steps(
            [WidthStep(_step_kind(node, entries)) for node in nodes],
            edges,
            group_components=False,
        )
    else:
        # Only a live producer can supply disconnected execution occurrences.
        # Analytical counts never establish their multiplicity or placement.
        order = tuple(
            (nodes.setdefault(owner, len(nodes)),) for owner in execution_order
        )
    builder = ComponentProgramBuilder(signature, functions.values(), name=name)
    available: dict[ComponentPort, RegisterBinding] = {}

    def add_outputs(
        owner: _Owner,
        port_registers: tuple[Register, ...],
        collections: Mapping[str, BindingCollection],
    ) -> None:
        available.update(
            zip(
                _indexed_ports(owner, port_registers),
                _elements(collections.values()),
                strict=True,
            )
        )

    def read_inputs(
        owner: _Owner, port_registers: tuple[Register, ...]
    ) -> dict[str, BindingCollection]:
        values = []
        for port in _indexed_ports(owner, port_registers):
            if port not in incoming:
                raise SerializationError(f"Missing connection to {port!r}")
            source = incoming.pop(port)
            if source not in available:
                raise SerializationError(f"Invalid connection from {source!r}")
            values.append(available.pop(source))
        return _collections(port_registers, values)

    add_outputs(LibraryBoundary.INPUT, tuple(signature.lefts()), builder.inputs)
    owners = tuple(nodes)
    for group in order:
        for index in group:
            owner = owners[index]
            if isinstance(owner, ComponentOccurrence):
                definition = definitions[owner.definition_id]
                assert definition.registers is not None
                ports = Signature(definition.registers)
                operands = read_inputs(owner, tuple(ports.lefts()))
                result = builder.call_dict(definition.name, operands)
                add_outputs(owner, tuple(ports.rights()), result)
    outputs = read_inputs(LibraryBoundary.OUTPUT, tuple(signature.rights()))
    if incoming:
        raise SerializationError(
            f"Unexpected connection destinations {tuple(incoming)}"
        )
    return builder.finalize(outputs).functions[0]


def resolve_component_program(
    library: ComponentLibrary,
    identifier: int,
    factories: Mapping[str, ComponentFactory],
    *,
    entry_name: str = "main",
) -> ProgramRecord:
    """Resolve stored constructor data without inferring omitted execution effects."""
    return _resolve_component_program(
        library, identifier, factories, entry_name=entry_name
    )


def _resolve_component_program(
    library: ComponentLibrary,
    identifier: int,
    factories: Mapping[str, ComponentFactory],
    *,
    entry_name: str = "main",
    execution_orders: Mapping[int, tuple[ComponentOccurrence, ...]] | None = None,
) -> ProgramRecord:
    """Resolve one library entry and its code dependencies into native Program.

    The supplied factories own named executable definitions. A constructor
    argument referencing another row receives its resolved native function.
    Composites and unregistered constructors with stored connection graphs use
    the existing typed builder. A registered constructor keeps precedence over
    its stored graph. Wire IDs and external gate payloads carry data for this
    boundary and cannot import constructors or supply missing executable leaves.
    """
    entries = _entry_index(library.entries)
    definitions: dict[int, FunctionRecord] = {}
    functions: dict[str, FunctionRecord] = {}
    visiting: set[int] = set()

    def add_function(function: FunctionRecord) -> None:
        if function.name in functions:
            raise SerializationError(f"Conflicting native function {function.name!r}")
        functions[function.name] = function

    def resolve(row_id: int) -> FunctionRecord:
        if row_id in definitions:
            return definitions[row_id]
        if row_id not in entries:
            raise SerializationError(f"Unknown component definition {row_id}")
        if row_id in visiting:
            raise SerializationError(f"Cyclic component definition {row_id}")
        visiting.add(row_id)
        entry = entries[row_id]
        declaration = entry.declaration
        name = entry_name if row_id == identifier else f"legacy_component_{row_id}"
        if declaration.name == _COMPOSITE or (
            declaration.name not in factories and entry.connections
        ):
            execution_order = (
                execution_orders.get(row_id) if execution_orders is not None else None
            )
            for occurrence in (
                _ordered_occurrences(entry.connections)
                if execution_order is None
                else execution_order
            ):
                resolve(occurrence.definition_id)
            function = _build_composite(
                name,
                entry,
                entries,
                definitions,
                functions,
                execution_order=execution_order,
            )
        else:
            if declaration.name not in factories:
                raise SerializationError(f"No native factory for {declaration.name!r}")
            arguments: dict[str, ResolvedArgument] = {}
            for argument in declaration.arguments:
                value = argument.value
                arguments[argument.name] = (
                    resolve(value.identifier)
                    if isinstance(value, ComponentReference)
                    else value
                )
            resolved = factories[declaration.name](
                name, declaration.registers, arguments
            )
            if isinstance(resolved, ProgramRecord):
                if resolved.physical_context is not None:
                    raise SerializationError(
                        "Native factory returned a physical Program"
                    )
                matches = [item for item in resolved.functions if item.name == name]
                if len(matches) != 1:
                    raise SerializationError(
                        f"Native factory must define {name!r} once"
                    )
                function = matches[0]
                for helper in resolved.functions:
                    if helper is not function:
                        add_function(helper)
            else:
                function = resolved
        if function.name != name:
            raise SerializationError(f"Native factory must use assigned name {name!r}")
        if function.registers is None or (
            declaration.registers is not None
            and function.registers != declaration.registers
        ):
            raise SerializationError(
                f"Native signature differs for {declaration.name!r}"
            )
        definitions[row_id] = function
        add_function(function)
        visiting.remove(row_id)
        return function

    try:
        main = resolve(identifier)
        program = ProgramRecord(
            (main, *(fn for fn in functions.values() if fn is not main))
        )
        diagnostics = verify_program_record(program)
        if diagnostics:
            raise SerializationError(f"Invalid reconstructed Program: {diagnostics}")
        return program
    except SerializationError:
        raise
    except ValueError as error:
        raise SerializationError(f"Invalid component Program: {error}") from error
