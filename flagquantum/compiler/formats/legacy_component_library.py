# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Wire layout adapted from Qualtran v0.7.0 protos/bloq.proto and serialization/bloq.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/protos/bloq.proto

"""Translate legacy library bytes to typed interchange records."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from ...core.ir.program.registers import Register
from ...core.value_math import SymbolicInt
from ...errors import SerializationError
from .._component_trace.resource_model import GateCost
from ._legacy_component_schema import _component_library_message
from ._legacy_protobuf import parse_message
from .legacy_arguments import _read_integer_argument, _write_integer_argument
from .legacy_component_arguments import (
    ComponentArgument,
    _read_component_argument,
    _write_component_argument,
)
from .legacy_registers import (
    _read_register,
    _write_register,
    deserialize_registers,
    serialize_registers,
)
from .resource_annotations import (
    deserialize_resource_annotation,
    serialize_resource_annotation,
)


class LibraryBoundary(Enum):
    INPUT = "LeftDangle"
    OUTPUT = "RightDangle"


@dataclass(frozen=True)
class ComponentOccurrence:
    instance_id: int
    definition_id: int


@dataclass(frozen=True)
class ComponentPort:
    owner: ComponentOccurrence | LibraryBoundary
    register: Register
    index: tuple[int, ...] = ()


@dataclass(frozen=True)
class ComponentConnection:
    source: ComponentPort
    destination: ComponentPort


@dataclass(frozen=True)
class ComponentDeclaration:
    name: str
    arguments: tuple[ComponentArgument, ...] = ()
    registers: tuple[Register, ...] | None = None
    annotation: GateCost | None = None


@dataclass(frozen=True)
class ComponentLibraryEntry:
    identifier: int
    declaration: ComponentDeclaration
    connections: tuple[ComponentConnection, ...] = ()
    counts: tuple[tuple[int, SymbolicInt], ...] = ()


@dataclass(frozen=True)
class ComponentLibrary:
    name: str = ""
    entries: tuple[ComponentLibraryEntry, ...] = ()


def _indexed_ports(
    owner: ComponentOccurrence | LibraryBoundary, registers: tuple[Register, ...]
) -> tuple[ComponentPort, ...]:
    return tuple(
        ComponentPort(owner, register, index)
        for register in registers
        for index in register.all_idxs()
    )


def _write_port(message: Any, port: ComponentPort) -> None:
    if isinstance(port.owner, LibraryBoundary):
        message.dangling_t = port.owner.value
    else:
        message.bloq_instance.instance_id = port.owner.instance_id
        message.bloq_instance.bloq_id = port.owner.definition_id
    _write_register(message.register, port.register)
    message.index.extend(port.index)


def _read_port(message: Any) -> ComponentPort:
    arm = message.WhichOneof("binst")
    if arm == "dangling_t":
        try:
            owner: ComponentOccurrence | LibraryBoundary = LibraryBoundary(
                message.dangling_t
            )
        except ValueError as error:
            raise SerializationError(
                f"Unknown component boundary {message.dangling_t!r}"
            ) from error
    elif arm == "bloq_instance":
        owner = ComponentOccurrence(
            message.bloq_instance.instance_id, message.bloq_instance.bloq_id
        )
    else:
        raise SerializationError("Component port has no owner")
    return ComponentPort(owner, _read_register(message.register), tuple(message.index))


def _write_declaration(message: Any, declaration: ComponentDeclaration) -> None:
    message.name = declaration.name
    for argument in declaration.arguments:
        _write_component_argument(message.args.add(), argument)
    if declaration.registers is not None:
        message.registers.ParseFromString(serialize_registers(declaration.registers))
    if declaration.annotation is not None:
        message.t_complexity.ParseFromString(
            serialize_resource_annotation(declaration.annotation)
        )


def _read_declaration(message: Any) -> ComponentDeclaration:
    registers = (
        deserialize_registers(message.registers.SerializeToString())
        if message.HasField("registers")
        else None
    )
    annotation = (
        deserialize_resource_annotation(message.t_complexity.SerializeToString())
        if message.HasField("t_complexity")
        else None
    )
    return ComponentDeclaration(
        message.name,
        tuple(_read_component_argument(arg) for arg in message.args),
        registers,
        annotation,
    )


def _write_entry(message: Any, entry: ComponentLibraryEntry) -> None:
    message.bloq_id = entry.identifier
    _write_declaration(message.bloq, entry.declaration)
    for connection in entry.connections:
        edge = message.decomposition.add()
        _write_port(edge.left, connection.source)
        _write_port(edge.right, connection.destination)
    for identifier, count in entry.counts:
        if identifier in message.bloq_counts:
            raise SerializationError(f"Duplicate count for component {identifier}")
        _write_integer_argument(message.bloq_counts[identifier], count)


def _read_entry(message: Any) -> ComponentLibraryEntry:
    if not message.HasField("bloq"):
        raise SerializationError("Component library row has no declaration")
    return ComponentLibraryEntry(
        message.bloq_id,
        _read_declaration(message.bloq),
        tuple(
            ComponentConnection(_read_port(edge.left), _read_port(edge.right))
            for edge in message.decomposition
        ),
        tuple(
            (identifier, _read_integer_argument(value))
            for identifier, value in sorted(message.bloq_counts.items())
        ),
    )


def serialize_component_library(library: ComponentLibrary) -> bytes:
    """Write definitions, ordered edges, and deterministic count maps."""
    message = _component_library_message()(name=library.name)
    try:
        for entry in library.entries:
            _write_entry(message.table.add(), entry)
        return cast(bytes, message.SerializeToString(deterministic=True))
    except SerializationError:
        raise
    except (ValueError, OverflowError, TypeError) as error:
        raise SerializationError("Invalid component library") from error


def deserialize_component_library(data: bytes) -> ComponentLibrary:
    """Read a library without resolving constructors or admitting it to Program."""
    message = parse_message(_component_library_message(), data, "Component library")
    return ComponentLibrary(
        message.name, tuple(_read_entry(entry) for entry in message.table)
    )
