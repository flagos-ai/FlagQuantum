# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# QREF v1 conversion adapted from Qualtran v0.7.0 qref_interop/_bloq_to_qref.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/qref_interop/_bloq_to_qref.py

"""Export native component metadata and logical resource graphs to QREF v1."""

from __future__ import annotations

from collections.abc import Collection, Hashable, Mapping
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from typing import Any, TypeVar

from ...core.ir.program.registers import Register
from ...core.value_math import SymbolicInt, SymbolicReal
from ...errors import SerializationError
from .._component_trace.call_graph import SymbolicCallGraph
from ._qref_ports import _port_suffix, qref_ports
from .legacy_component_discovery import _entry_index, _ordered_occurrences
from .legacy_component_library import (
    ComponentLibrary,
    ComponentOccurrence,
    ComponentPort,
)

_Node = TypeVar("_Node", bound=Hashable)
_RESOURCE_NAMES = ("t", "clifford", "rotations")
_COMPOSITE_TYPE = "CompositeBloq"


@dataclass(frozen=True)
class QrefComponent:
    """Name, type, and the three declared historical resource expressions.

    Use the existing legacy gate budget projection for factored gate counts.
    Native or external component type labels stay explicit at export.
    """

    name: str
    type_name: str
    resources: Mapping[str, SymbolicReal | str | None]


@cache
def _schema_module() -> Any:
    return import_module("qref.schema_v1")


def _attributes(
    component: QrefComponent,
    registers: tuple[Register, ...] = (),
    *,
    name: str | None = None,
    type_name: str | None = None,
) -> dict[str, Any]:
    from sympy import sympify

    ports = qref_ports(registers)
    locals_by_name = {}
    params: set[str] = set()
    for port in ports:
        size = port["size"]
        if isinstance(size, str):
            if size.isidentifier():
                params.update(str(symbol) for symbol in sympify(size).free_symbols)
            elif not size.removeprefix("-").isdecimal():
                # Bartiq expects each symbolic input port size to be a single name.
                local = f"{port['name']}_size"
                locals_by_name[local] = size
                port["size"] = local
    resources = []
    for resource_name in _RESOURCE_NAMES:
        if resource_name not in component.resources:
            raise SerializationError(f"QREF resource {resource_name!r} is missing")
        value = component.resources[resource_name]
        if value is not None and not isinstance(value, (int, float, str)):
            params.update(str(symbol) for symbol in value.free_symbols)
        resources.append(
            {
                "name": resource_name,
                "value": (
                    value
                    if value is None or isinstance(value, (int, float, str))
                    else str(value)
                ),
                "type": "additive",
            }
        )
    attributes: dict[str, Any] = {
        "name": component.name if name is None else name,
        "type": component.type_name if type_name is None else type_name,
        "ports": ports,
        "resources": resources,
        "input_params": sorted(params - locals_by_name.keys()),
    }
    if locals_by_name:
        attributes["local_variables"] = locals_by_name
    return attributes


def qref_component(
    component: QrefComponent, registers: tuple[Register, ...] = ()
) -> Any:
    """Return the optional QREF package's SchemaV1 for one leaf."""
    schema = _schema_module()
    return schema.SchemaV1(
        version="v1", program=schema.RoutineV1(**_attributes(component, registers))
    )


def _occurrence_name(
    occurrence: ComponentOccurrence, metadata: Mapping[int, QrefComponent]
) -> str:
    return f"{metadata[occurrence.definition_id].type_name}_{occurrence.instance_id}"


def _relative_port_name(
    port: ComponentPort, direction: str, metadata: Mapping[int, QrefComponent]
) -> str:
    suffix = _port_suffix(port.index)
    if isinstance(port.owner, ComponentOccurrence):
        child = _occurrence_name(port.owner, metadata)
        return f"{child}.{direction}_{port.register.name}{suffix}"
    boundary = "out" if direction == "in" else "in"
    return f"{boundary}_{port.register.name}{suffix}"


def qref_component_graph(
    library: ComponentLibrary,
    root: int,
    metadata: Mapping[int, QrefComponent],
    *,
    decomposition_rules: bool | Collection[str] = False,
    decompositions: Mapping[int, tuple[ComponentOccurrence, ...]] | None = None,
) -> Any:
    """Export ordered component edges and selected nested definitions to QREF.

    A native Program entry can supply its edges through the existing component
    export adapter. Composites expose their graph by default. Named types can
    be unfolded by a type selector or by enabling all available decompositions.
    """
    schema = _schema_module()
    entries = _entry_index(library.entries)
    active: set[int] = set()

    def routine(identifier: int, name: str | None = None) -> Any:
        if identifier not in entries or identifier not in metadata:
            raise SerializationError(f"Missing QREF definition {identifier}")
        if identifier in active:
            raise SerializationError(f"Cyclic QREF decomposition at {identifier}")
        entry = entries[identifier]
        component = metadata[identifier]
        composite = component.type_name == _COMPOSITE_TYPE
        selected = decomposition_rules is True or (
            not isinstance(decomposition_rules, bool)
            and component.type_name in decomposition_rules
        )
        declared = None if decompositions is None else decompositions.get(identifier)
        expand = composite or (
            selected and (bool(entry.connections) or declared is not None)
        )
        attrs = _attributes(
            component,
            entry.declaration.registers or (),
            name=name,
            type_name=_COMPOSITE_TYPE if expand else None,
        )
        if expand:
            active.add(identifier)
            connected = _ordered_occurrences(entry.connections)
            occurrences = sorted(
                connected if declared is None else declared,
                key=lambda node: node.instance_id,
            )
            by_id = {node.instance_id: node for node in occurrences}
            if len(by_id) != len(occurrences) or any(
                by_id.get(node.instance_id) != node for node in connected
            ):
                raise SerializationError(
                    "QREF decomposition occurrences differ from connected endpoints"
                )
            for occurrence in occurrences:
                if occurrence.definition_id not in metadata:
                    raise SerializationError(
                        f"Missing QREF definition {occurrence.definition_id}"
                    )
            attrs["children"] = [
                routine(child.definition_id, _occurrence_name(child, metadata))
                for child in occurrences
            ]
            attrs["connections"] = [
                {
                    "source": _relative_port_name(edge.source, "out", metadata),
                    "target": _relative_port_name(edge.destination, "in", metadata),
                }
                for edge in entry.connections
            ]
            active.remove(identifier)
        return schema.RoutineV1(**attrs)

    return schema.SchemaV1(version="v1", program=routine(root))


def qref_call_graph(
    graph: SymbolicCallGraph[_Node],
    metadata: Mapping[_Node, QrefComponent],
    *,
    registers: Mapping[_Node, tuple[Register, ...]] | None = None,
    component_routines: Mapping[_Node, Mapping[str, object]] | None = None,
) -> Any:
    """Export an existing logical call graph with symbolic repetition counts.

    A node with an existing QREF component routine keeps its declared child
    routines. Its outer ports and connections are cleared for the count view.
    """
    schema = _schema_module()
    children: dict[_Node, list[tuple[_Node, SymbolicInt]]] = {}
    for (parent, child), count in graph.direct_calls.items():
        children.setdefault(parent, []).append((child, count))
    names: dict[_Node, str] = {}
    used: set[str] = set()
    collision = 0
    pending = [graph.root]
    while pending:
        node = pending.pop()
        if node in names:
            continue
        if node not in metadata:
            raise SerializationError(f"Missing QREF resource node {node!r}")
        base = metadata[node].name
        name = base
        while name in used:
            collision += 1
            name = f"{base}_{collision}"
        names[node] = name
        used.add(name)
        pending.extend(child for child, _ in reversed(children.get(node, ())))

    routines: dict[_Node, Any] = {}
    visiting: set[_Node] = set()

    def routine(node: _Node) -> Any:
        if node in routines:
            return routines[node]
        if node in visiting:
            raise SerializationError("QREF resource graph contains a cycle")
        visiting.add(node)
        base = None if component_routines is None else component_routines.get(node)
        if base is None:
            declared_ports = () if registers is None else registers.get(node, ())
            attrs = _attributes(metadata[node], declared_ports)
        else:
            attrs = schema.RoutineV1(**base).model_dump()
        attrs["name"] = names[node]
        attrs["ports"] = []
        attrs["connections"] = []
        subroutines = list(attrs.get("children", ()))
        for child, count in reversed(children.get(node, ())):
            descendant = routine(child)
            if count != 1:
                descendant = schema.RoutineV1(
                    name=f"{descendant.name}_repeated",
                    input_params=descendant.input_params,
                    children=[descendant],
                    repetition={"count": str(count), "sequence": {"type": "constant"}},
                )
            subroutines.append(descendant)
        attrs["children"] = subroutines
        result = schema.RoutineV1(**attrs)
        visiting.remove(node)
        routines[node] = result
        return result

    return schema.SchemaV1(version="v1", program=routine(graph.root))
