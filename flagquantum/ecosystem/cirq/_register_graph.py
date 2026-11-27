# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source graph interchange follows Qualtran v0.7.0 serialization/bloq.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/serialization/bloq.py

"""Convert a live typed SDK graph to owned Program with explicit lifetimes."""

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Any

from ...compiler.formats.legacy_component_arguments import (
    ComponentArgument,
    ComponentReference,
)
from ...compiler.formats.legacy_component_library import (
    ComponentDeclaration,
    ComponentLibrary,
    ComponentLibraryEntry,
    ComponentOccurrence,
    deserialize_component_library,
    serialize_component_library,
)
from ...compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    _resolve_component_program,
    resolve_component_program,
)
from ...compiler.registers.component_graph import _namespace_helpers
from ...compiler.registers.component_rebindings import (
    join_word_function,
    split_word_function,
)
from ...compiler.registers.layout.component_partition import partition_component_program
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


def _declaration_key(declaration: ComponentDeclaration) -> bytes:
    # Canonical interchange preserves array dtype, shape and exact values.
    # Dataclass equality cannot compare ndarray-valued constructor arguments.
    return serialize_component_library(
        ComponentLibrary(entries=(ComponentLibraryEntry(0, declaration),))
    )


def _leaf_program(
    source: Any, name: str, registers: tuple[Register, ...]
) -> FunctionRecord | ProgramRecord:
    from qualtran._infra.gate_with_registers import get_named_qubits
    from qualtran.bloqs.bookkeeping.allocate import Allocate
    from qualtran.bloqs.bookkeeping.free import Free
    from qualtran.bloqs.bookkeeping.join import Join
    from qualtran.bloqs.bookkeeping.partition import _PartitionBase
    from qualtran.bloqs.bookkeeping.split import Split

    if isinstance(source, (Allocate, Free)):
        dtype = registers[0].dtype
        lifetime_factory = (
            allocate_word_function
            if isinstance(source, Allocate)
            else free_word_function
        )
        return lifetime_factory(name, dtype, source.dirty)
    if isinstance(source, _PartitionBase):
        parts = registers[1:]
        return partition_component_program(parts, name, reverse=not source.partition)
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
    from qualtran._infra.bloq import Bloq
    from qualtran._infra.composite_bloq import CompositeBloq
    from qualtran._infra.data_types import QMontgomeryUInt
    from qualtran.bloqs.bookkeeping.join import Join
    from qualtran.bloqs.bookkeeping.split import Split
    from qualtran.serialization.bloq import (
        _assign_bloq_an_id,
        _search_for_subbloqs,
        bloqs_to_proto,
    )

    if not isinstance(source, Bloq):
        raise TypeError("Register graph import requires a live typed component")
    graph = (
        source if isinstance(source, CompositeBloq) else source.decompose_bloq()
    ).flatten()
    library = deserialize_component_library(
        bloqs_to_proto(graph, max_depth=1, pred=lambda _: False).SerializeToString()
    )
    # Use the same unchanged discovery as serialization so constructor child
    # IDs belong to this complete live graph, not separately numbered leaves.
    source_ids: dict[Any, tuple[int, bool]] = {}
    _assign_bloq_an_id(graph, source_ids, shallow=True)
    _search_for_subbloqs(graph, source_ids, lambda _: False, 1)
    entries = {entry.identifier: entry for entry in library.entries}
    if {identifier for identifier, _ in source_ids.values()} != set(entries):
        raise ValueError("Live graph discovery differs from serialized definitions")
    leaves: dict[str, list[tuple[ComponentDeclaration, Any]]] = {}
    for leaf, (identifier, _) in source_ids.items():
        if leaf is graph:
            continue
        declaration = entries[identifier].declaration
        candidates = leaves.setdefault(declaration.name, [])
        duplicates = [
            existing_leaf
            for existing, existing_leaf in candidates
            if _declaration_key(existing) == _declaration_key(declaration)
        ]
        if duplicates:
            # The source serializer omits Montgomery modulus. These exact
            # views lower only raw ordered storage through the owned dtype.
            if (
                type(leaf) in (Split, Join)
                and type(leaf.dtype) is QMontgomeryUInt
                and type(duplicates[0]) is type(leaf)
                and type(duplicates[0].dtype) is QMontgomeryUInt
            ):
                continue
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
        matches = []
        for declaration, leaf in leaves[source_name]:
            constructor_arguments = []
            if set(arguments) != {arg.name for arg in declaration.arguments}:
                continue
            for argument in declaration.arguments:
                value = arguments[argument.name]
                expected = argument.value
                if isinstance(expected, ComponentReference):
                    if not isinstance(value, FunctionRecord) or value.name != (
                        f"legacy_component_{expected.identifier}"
                    ):
                        break
                    constructor_arguments.append(argument)
                else:
                    if isinstance(value, FunctionRecord):
                        break
                    constructor_arguments.append(
                        ComponentArgument(argument.name, value)
                    )
            else:
                if _declaration_key(declaration) == _declaration_key(
                    replace(
                        declaration,
                        registers=registers,
                        arguments=tuple(constructor_arguments),
                    )
                ):
                    matches.append(leaf)
        if len(matches) != 1:
            raise ValueError(
                "Register graph leaf needs an unambiguous live declaration"
            )
        return _leaf_program(matches[0], name, registers)

    factories = {
        source_name: partial(factory, source_name=source_name) for source_name in leaves
    }
    if any(not tuple(instance.bloq.signature) for instance in graph.bloq_instances):
        # Protobuf connections omit zero-port occurrences. Keep actual live
        # identities and traversal order instead of reconstructing them from costs.
        execution_order = tuple(
            ComponentOccurrence(instance.i, source_ids[instance.bloq][0])
            for instance, _, _ in graph.iter_bloqsoqs()
        )
        return _resolve_component_program(
            library, 0, factories, execution_orders={0: execution_order}
        )
    return resolve_component_program(library, 0, factories)
