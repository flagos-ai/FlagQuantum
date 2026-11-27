# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Live graph/endpoint semantics follow locked Qualtran v0.7.0 CompositeBloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/composite_bloq.py

"""End live SDK graphs at owned signatures, occurrences and explicit lifetimes."""

from dataclasses import replace
from numbers import Integral
from typing import Any

from ...compiler.formats.legacy_component_library import (
    ComponentConnection,
    ComponentDeclaration,
    ComponentLibraryEntry,
    ComponentOccurrence,
    ComponentPort,
    LibraryBoundary,
)
from ...compiler.formats.legacy_component_resolution import _build_composite
from ...compiler.formats.legacy_registers import deserialize_registers
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
from ...core.ir.program import FunctionRecord, ProgramRecord, verify_program_record
from ...core.ir.program.register_types import (
    classical_montgomery_register_type,
    montgomery_register_type,
)
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_parameters import is_symbolic_value
from ._qubit_manager import InteropQubitManager
from ._register_import import import_cirq_register_optree
from .conversion import _cirq


def _live_registers(source: Any) -> tuple[Register, ...]:
    """Reuse the nominal codec and retain live Montgomery fields it cannot encode."""
    from qualtran._infra.data_types import CMontgomeryUInt, QMontgomeryUInt
    from qualtran.serialization.registers import registers_to_proto

    registers = deserialize_registers(
        registers_to_proto(source.signature).SerializeToString()
    )
    result = []
    for register, original in zip(registers, source.signature, strict=True):
        if isinstance(original.dtype, (QMontgomeryUInt, CMontgomeryUInt)):
            modulus = original.dtype.modulus
            if isinstance(modulus, Integral):
                modulus = int(modulus)
            elif modulus is not None and not is_symbolic_value(modulus):
                raise ValueError("Live Montgomery modulus must be integral or symbolic")
            factory = (
                classical_montgomery_register_type
                if isinstance(original.dtype, CMontgomeryUInt)
                else montgomery_register_type
            )
            register = Register(
                register.name,
                factory(register.bitsize, modulus),
                shape=register.shape_symbolic,
                side=register.side,
            )
        result.append(register)
    return tuple(result)


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
    """Capture actual leaf identities and connections without a constructor wire.

    Only register metadata crosses the existing nominal codec. Physical leaves
    still execute their original SDK callbacks. Shared Compiler owners replay
    the complete live call order and check every typed edge and lifetime.
    No source object or callback is retained in the resulting native Program.
    """
    from qualtran._infra.bloq import Bloq
    from qualtran._infra.composite_bloq import CompositeBloq
    from qualtran._infra.quantum_graph import LeftDangle, RightDangle

    if not isinstance(source, Bloq):
        raise TypeError("Register graph import requires a live typed component")
    graph = (
        source if isinstance(source, CompositeBloq) else source.decompose_bloq()
    ).flatten()
    leaves: dict[Any, int] = {}
    for instance in sorted(graph.bloq_instances, key=lambda item: item.i):
        if instance.bloq not in leaves:
            leaves[instance.bloq] = len(leaves) + 1
    registers = {0: _live_registers(graph)}
    registers.update(
        (identifier, _live_registers(leaf)) for leaf, identifier in leaves.items()
    )
    ports = {
        identifier: {
            (register.name, register.side.value): register for register in signature
        }
        for identifier, signature in registers.items()
    }

    def port(soquet: Any) -> ComponentPort:
        if soquet.binst is LeftDangle:
            owner: ComponentOccurrence | LibraryBoundary = LibraryBoundary.INPUT
            identifier = 0
        elif soquet.binst is RightDangle:
            owner = LibraryBoundary.OUTPUT
            identifier = 0
        else:
            identifier = leaves[soquet.binst.bloq]
            owner = ComponentOccurrence(soquet.binst.i, identifier)
        return ComponentPort(
            owner,
            ports[identifier][(soquet.reg.name, soquet.reg.side.value)],
            tuple(soquet.idx),
        )

    connections = tuple(
        ComponentConnection(port(edge.left), port(edge.right))
        for edge in graph.connections
    )
    entries = {
        identifier: ComponentLibraryEntry(
            identifier,
            ComponentDeclaration(
                f"{type(leaf).__module__}.{type(leaf).__qualname__}",
                registers=registers[identifier],
            ),
        )
        for leaf, identifier in leaves.items()
    }
    entries[0] = ComponentLibraryEntry(
        0,
        ComponentDeclaration(
            "qualtran._infra.composite_bloq.CompositeBloq", registers=registers[0]
        ),
        connections,
    )
    definitions: dict[int, FunctionRecord] = {}
    functions: dict[str, FunctionRecord] = {}
    for leaf, identifier in leaves.items():
        name = f"component_{identifier}_{type(leaf).__name__}"
        captured = _leaf_program(leaf, name, registers[identifier])
        module = (
            captured.functions if isinstance(captured, ProgramRecord) else (captured,)
        )
        for function in module:
            if function.name in functions:
                raise ValueError("Live graph leaf definitions have conflicting names")
            functions[function.name] = function
        definitions[identifier] = functions[name]
    execution_order = tuple(
        ComponentOccurrence(instance.i, leaves[instance.bloq])
        for instance, _, _ in graph.iter_bloqsoqs()
    )
    root = _build_composite(
        "main",
        entries[0],
        entries,
        definitions,
        functions,
        execution_order=execution_order,
    )
    program = ProgramRecord((root, *functions.values()))
    diagnostics = verify_program_record(program)
    if diagnostics:
        raise ValueError(f"Invalid captured register graph: {diagnostics}")
    return program
