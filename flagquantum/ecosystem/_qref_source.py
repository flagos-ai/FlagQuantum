# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Metadata projection follows locked Qualtran v0.7.0 qref_interop/_bloq_to_qref.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/qref_interop/_bloq_to_qref.py

"""Translate live SDK metadata into the existing owned QREF format inputs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from ..compiler._component_trace.call_graph import SymbolicCallGraph
from ..compiler.formats.legacy_component_library import (
    ComponentConnection,
    ComponentDeclaration,
    ComponentLibrary,
    ComponentLibraryEntry,
    ComponentOccurrence,
    ComponentPort,
    LibraryBoundary,
)
from ..compiler.formats.qref_resources import (
    QrefComponent,
    _schema_module,
    qref_call_graph,
    qref_component_graph,
)
from ..core.ir.program.registers import Register, RegisterSide, RegisterType
from ..core.value_math import SymbolicReal


def _qref_registers(values: Any) -> tuple[Register, ...]:
    """Project only the widths and directions that QREF actually represents.

    This is metadata, not an executable dtype conversion. In particular source
    CBit has valid QREF ports although its unrelated wire serializer fails.
    """
    return tuple(
        Register(
            value.name,
            RegisterType("opaque", value.dtype.num_qubits, value.dtype.num_cbits),
            value.shape_symbolic,
            RegisterSide(value.side.value),
        )
        for value in values
    )


def _component_metadata(
    source: Any, *, expanded: bool = False, origin: Any = None
) -> QrefComponent:
    from qualtran.cirq_interop._cirq_to_bloq import CirqGateAsBloq
    from sympy import Expr

    original = source if origin is None else origin
    name = type(original).__name__
    if not expanded:
        name += "_uncompute" if hasattr(source, "uncompute") else ""
        name += "_adjoint" if hasattr(source, "is_adjoint") else ""
    type_name = (
        str(original.gate)
        if isinstance(original, CirqGateAsBloq)
        else type(original).__name__
    )
    cost = source.t_complexity()
    values: dict[str, SymbolicReal | str | None] = {}
    for field in ("t", "clifford", "rotations"):
        value = getattr(cost, field)
        if isinstance(value, Expr):
            values[field] = value
        else:
            # Preserve QREF's own scalar coercion, including NumPy integers
            # which its pinned schema represents as floating-point values.
            normalized = (
                _schema_module()
                .ResourceV1(name=field, type="additive", value=value)
                .value
            )
            assert normalized is None or isinstance(normalized, (int, float, str))
            values[field] = normalized
    return QrefComponent(name, type_name, values)


def qref_from_source(
    source: object,
    *,
    from_callgraph: bool = False,
    decomposition_rules: bool | Iterable[type[object]] = False,
) -> Any:
    """Capture optional live SDK objects, then export only owned metadata.

    Source declared costs and call counts are imported data. This function
    neither executes a quantum operation nor substitutes SDK numerical kernels
    for a native implementation. Returned QREF schemas retain no SDK callbacks.
    """
    from qualtran._infra.bloq import Bloq
    from qualtran._infra.composite_bloq import CompositeBloq
    from qualtran._infra.quantum_graph import BloqInstance

    if not isinstance(source, (Bloq, BloqInstance)):
        raise TypeError(
            "QREF capture requires an explicit source component or instance"
        )
    if from_callgraph:
        if isinstance(source, BloqInstance):
            raise ValueError("BloqInstance cannot drive a callgraph import.")
        graph, totals = source.call_graph()
        identifiers = {node: position for position, node in enumerate(graph.nodes)}
        owned = SymbolicCallGraph(
            identifiers[source],
            {
                (identifiers[a], identifiers[b]): attributes.get("n", 1)
                for a, b, attributes in graph.edges(data=True)
            },
            {identifiers[node]: count for node, count in totals.items()},
        )
        metadata = {
            identifier: _component_metadata(node)
            for node, identifier in identifiers.items()
        }
        registers = {
            identifier: _qref_registers(node.signature)
            for node, identifier in identifiers.items()
        }
        routines = {
            identifier: qref_from_source(node).program.model_dump()
            for node, identifier in identifiers.items()
            if isinstance(node, CompositeBloq)
        }
        return qref_call_graph(
            owned, metadata, registers=registers, component_routines=routines
        )
    return _capture_component_graph(source, decomposition_rules)


def _capture_component_graph(source: Any, rules: bool | Iterable[type[object]]) -> Any:
    from qualtran._infra.bloq import DecomposeNotImplementedError, DecomposeTypeError
    from qualtran._infra.composite_bloq import CompositeBloq
    from qualtran._infra.quantum_graph import BloqInstance, LeftDangle, RightDangle

    keep = None if rules is True else set() if rules is False else set(rules)
    identifiers: dict[object, int] = {}
    entries: dict[int, ComponentLibraryEntry] = {}
    metadata: dict[int, QrefComponent] = {}
    decompositions: dict[int, tuple[ComponentOccurrence, ...]] = {}
    registers: dict[object, Register] = {}

    def capture(node: Any) -> int:
        if node in identifiers:
            return identifiers[node]
        identifier = len(identifiers)
        identifiers[node] = identifier
        graph = node if isinstance(node, CompositeBloq) else None
        if graph is None and (keep is None or type(node) in keep):
            try:
                graph = node.decompose_bloq()
            except (DecomposeNotImplementedError, DecomposeTypeError):
                pass
        view = node if graph is None else graph
        native_registers = _qref_registers(view.signature)
        registers.update(zip(view.signature, native_registers, strict=True))
        metadata[identifier] = _component_metadata(
            view, expanded=graph is not None, origin=node
        )
        declaration = ComponentDeclaration(
            type(node).__name__, registers=native_registers
        )
        if graph is None:
            entries[identifier] = ComponentLibraryEntry(identifier, declaration)
            return identifier
        instances = sorted(graph.bloq_instances, key=lambda child: child.i)
        occurrences = {
            child: ComponentOccurrence(child.i, capture(child.bloq))
            for child in instances
        }

        def port(soquet: Any) -> ComponentPort:
            owner = (
                LibraryBoundary.INPUT
                if soquet.binst is LeftDangle
                else (
                    LibraryBoundary.OUTPUT
                    if soquet.binst is RightDangle
                    else occurrences[soquet.binst]
                )
            )
            return ComponentPort(owner, registers[soquet.reg], tuple(soquet.idx))

        connections = tuple(
            ComponentConnection(port(edge.left), port(edge.right))
            for edge in graph.connections
        )
        entries[identifier] = ComponentLibraryEntry(
            identifier, declaration, connections
        )
        decompositions[identifier] = tuple(occurrences.values())
        return identifier

    original = source.bloq if isinstance(source, BloqInstance) else source
    root = capture(original)
    if isinstance(source, BloqInstance):
        metadata[root] = replace(
            metadata[root], name=f"{metadata[root].type_name}_{source.i}"
        )
    library = ComponentLibrary(entries=tuple(entries[key] for key in sorted(entries)))
    return qref_component_graph(
        library, root, metadata, decomposition_rules=True, decompositions=decompositions
    )
