# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 qref_interop/bartiq_demo.ipynb.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/qref_interop/bartiq_demo.ipynb

"""Native alias-preparation models for the QREF and Bartiq lesson."""

from typing import Any

from sympy import Integer

from flagquantum.algorithms.primitives.alias_preparation import (
    source_dense_alias_preparation_cost,
)
from flagquantum.algorithms.primitives.alias_sampling import alias_sampling_table
from flagquantum.compiler._component_trace.gate_budget import legacy_gate_budget
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    GateCost,
    estimate_component_resources,
)
from flagquantum.compiler.formats.legacy_component_library import (
    ComponentConnection,
    ComponentDeclaration,
    ComponentLibrary,
    ComponentLibraryEntry,
    ComponentOccurrence,
    ComponentPort,
    LibraryBoundary,
)
from flagquantum.compiler.formats.qref_resources import (
    QrefComponent,
    qref_component,
    qref_component_graph,
)
from flagquantum.compiler.registers.preparation.uniform_preparation_resource_model import (
    source_uniform_preparation_resource_formula,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.value_math import SymbolicInt

from .alias_preparation_catalog import AliasPreparationExample
from .uniform_preparation_catalog import UniformPreparationExample


def alias_interchange_example() -> AliasPreparationExample:
    return AliasPreparationExample(
        alias_sampling_table(tuple(range(24)), precision=1e-5)
    )


def _resources(name: str, cost: ComponentCost) -> QrefComponent:
    return QrefComponent(
        name, name, legacy_gate_budget(estimate_component_resources(cost).gates)
    )


def alias_preparation_qref(example: AliasPreparationExample) -> Any:
    """Project the dense alias call layout and the existing historical budget."""
    table = example.table
    cost = source_dense_alias_preparation_cost(table, legacy_shims=True)
    uniform, lookup, compare = (call.component for call in cost.calls)
    root = "CompositeBloq"
    probabilities = table.mu
    selection = (table.selection_size - 1).bit_length()
    hadamards = ComponentCost("OnEach", probabilities, GateCost(clifford=probabilities))
    swap = ComponentCost("CSwap", 2 * selection + 1, GateCost(cswap=selection))
    calls = (
        ("PrepareUniformSuperposition", uniform, (("target", "selection"),)),
        ("OnEach", hadamards, (("q", "sigma_mu"),)),
        (
            "QROM",
            lookup,
            (("selection", "selection"), ("target0_", "alt"), ("target1_", "keep")),
        ),
        (
            "LessThanEqual",
            compare,
            (("x", "keep"), ("y", "sigma_mu"), ("target", "less_than_equal")),
        ),
        (
            "CSwap",
            swap,
            (("ctrl", "less_than_equal"), ("x", "alt"), ("y", "selection")),
        ),
    )
    registers = {register.name: register for register in example.signature}
    live = {
        name: ComponentPort(LibraryBoundary.INPUT, reg)
        for name, reg in registers.items()
    }
    entries = []
    metadata = {0: _resources(root, cost)}
    edges = []
    for instance, (name, model, bindings) in enumerate(calls):
        definition = instance + 1
        owner = ComponentOccurrence(instance, definition)
        child_registers = tuple(
            Register(port, registers[parent].dtype) for port, parent in bindings
        )
        for child_register, (_, parent) in zip(child_registers, bindings, strict=True):
            port = ComponentPort(owner, child_register)
            edges.append(ComponentConnection(live[parent], port))
            live[parent] = port
        entries.append(
            ComponentLibraryEntry(
                definition, ComponentDeclaration(name, registers=child_registers)
            )
        )
        metadata[definition] = _resources(name, model)
    edges.extend(
        ComponentConnection(live[name], ComponentPort(LibraryBoundary.OUTPUT, register))
        for name, register in registers.items()
    )
    entry = ComponentLibraryEntry(
        0, ComponentDeclaration(root, registers=tuple(registers.values())), tuple(edges)
    )
    return qref_component_graph(
        ComponentLibrary(entries=(entry, *entries)), 0, metadata
    )


def uniform_preparation_qref(count: SymbolicInt) -> Any:
    """Export the pinned uniform-preparation formula on its symbolic port."""
    name = "PrepareUniformSuperposition"
    formula = source_uniform_preparation_resource_formula(count, legacy_shims=True)
    budget = legacy_gate_budget(formula)
    if not isinstance(count, int):
        budget = {
            key: Integer(value) if isinstance(value, int) else value
            for key, value in budget.items()
        }
    return qref_component(
        QrefComponent(name, name, budget),
        tuple(UniformPreparationExample(count).signature),
    )
