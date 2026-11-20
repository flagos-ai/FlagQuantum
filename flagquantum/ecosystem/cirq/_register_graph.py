# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source graph interchange follows Qualtran v0.7.0 serialization/bloq.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/serialization/bloq.py

"""Convert a live typed SDK graph to owned Program with explicit lifetimes."""

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Any

from ...compiler.formats.legacy_component_arguments import ComponentArgument
from ...compiler.formats.legacy_component_library import (
    ComponentDeclaration,
    deserialize_component_library,
)
from ...compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from ...compiler.registers.component_graph import _namespace_helpers
from ...compiler.registers.component_rebindings import (
    join_word_function,
    split_word_function,
)
from ...compiler.registers.layout.component_workspace import (
    allocate_word_function,
    free_word_function,
)
from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ._qubit_manager import InteropQubitManager
from ._register_import import import_cirq_register_optree
from .conversion import _cirq


def _leaf_program(
    source: Any, name: str, registers: tuple[Register, ...]
) -> FunctionRecord | ProgramRecord:
    from qualtran._infra.gate_with_registers import get_named_qubits
    from qualtran.bloqs.bookkeeping import Allocate, Free, Join, Split

    if isinstance(source, (Allocate, Free)):
        dtype = registers[0].dtype
        lifetime_factory = (
            allocate_word_function
            if isinstance(source, Allocate)
            else free_word_function
        )
        return lifetime_factory(name, dtype, source.dirty)
    if isinstance(source, (Split, Join)):
        rebinding_factory = (
            split_word_function if isinstance(source, Split) else join_word_function
        )
        dtype = next(reg.dtype for reg in registers if not reg.shape)
        return rebinding_factory(name, dtype)
    cirq = _cirq()
    quregs = get_named_qubits(source.signature)
    incoming = {reg.name: quregs[reg.name] for reg in source.signature.lefts()}
    manager = InteropQubitManager(cirq.GreedyQubitManager(prefix="graph_work"))
    manager.manage_qubits(
        qubit for values in incoming.values() for qubit in values.flat
    )
    operation, outgoing = source.as_cirq_op(qubit_manager=manager, **incoming)
    record = import_cirq_register_optree(
        operation,
        signature=Signature(registers),
        in_quregs=incoming,
        out_quregs=outgoing,
    )
    # The importer returns its entry first. The existing Compiler helper owns
    # deterministic callee renaming and keeps each leaf's local helpers isolated.
    record = replace(
        record,
        functions=(replace(record.functions[0], name=name), *record.functions[1:]),
    )
    return _namespace_helpers(record, name)


def import_register_graph(source: object) -> ProgramRecord:
    """Import the actual typed decomposition without re-inferring work origin.

    This private boundary accepts the optional SDK's live component or stored
    CompositeBloq. It uses its declared flattened graph, unchanged serializers
    and the existing owned library resolver. Allocate/Free and Split/Join map
    directly to native definitions. Other leaves use the existing Cirq importer.
    Missing source decompositions and unsupported leaves retain their errors.
    The returned Program owns all data and retains no foreign object.
    """
    from qualtran import Bloq, CompositeBloq
    from qualtran.serialization.bloq import bloqs_to_proto

    if not isinstance(source, Bloq):
        raise TypeError("Register graph import requires a live typed component")
    graph = (
        source if isinstance(source, CompositeBloq) else source.decompose_bloq()
    ).flatten()
    library = deserialize_component_library(
        bloqs_to_proto(graph, max_depth=1, pred=lambda _: False).SerializeToString()
    )
    # Shallow declarations identify live leaves without importing constructors
    # named by serialized text. Values come only from this caller's live graph.
    leaves: dict[str, list[tuple[ComponentDeclaration, Any]]] = {}
    for leaf in dict.fromkeys(instance.bloq for instance in graph.bloq_instances):
        declaration = (
            deserialize_component_library(
                bloqs_to_proto(leaf, max_depth=0).SerializeToString()
            )
            .entries[0]
            .declaration
        )
        candidates = leaves.setdefault(declaration.name, [])
        if any(existing == declaration for existing, _ in candidates):
            raise ValueError(
                "Distinct live components share one serialized declaration"
            )
        candidates.append((declaration, leaf))

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
        *,
        source_name: str,
    ) -> FunctionRecord | ProgramRecord:
        if registers is None:
            raise ValueError("Register graph leaves require typed declarations")
        constructor_arguments = tuple(
            ComponentArgument(key, value)
            for key, value in arguments.items()
            if not isinstance(value, FunctionRecord)
        )
        matches = [
            leaf
            for declaration, leaf in leaves[source_name]
            if declaration.registers == registers
            and declaration.arguments == constructor_arguments
            and len(constructor_arguments) == len(arguments)
        ]
        if len(matches) != 1:
            raise ValueError(
                "Register graph leaf needs an unambiguous live declaration"
            )
        return _leaf_program(matches[0], name, registers)

    return resolve_component_program(
        library,
        0,
        {
            source_name: partial(factory, source_name=source_name)
            for source_name in leaves
        },
    )
