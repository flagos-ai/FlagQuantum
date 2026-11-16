# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 call-graph protocol notebook.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/call_graph.ipynb

"""Native declarations and decomposition fixtures for the call-graph protocol."""

from dataclasses import replace

from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    RawCallees,
    SymbolAllocator,
    SymbolicCallGraph,
    component_callee_counts,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    ConjunctionCall,
    ResourceCall,
    generalize_control_values,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler._component_trace.terminal_t_cost import (
    _terminal_resource_callees,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.control.multi_control import multi_and_stages
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import SymbolicInt

BINARY_AND = ConjunctionCall((1, 1))
CNOT = CliffordCall("cx", 2)
MY_FUNCTION = ComponentCost("MyFunc", 0)


def resource_declaration(
    node: ResourceCall, _: SymbolAllocator
) -> RawCallees[ResourceCall]:
    return _terminal_resource_callees(node, None)


def conjunction_graph(
    values: tuple[int, ...], *, generalize_activation: bool = False
) -> SymbolicCallGraph[ResourceCall]:
    root: ResourceCall = ConjunctionCall(values)
    if not generalize_activation:
        return symbolic_call_graph(root, resource_declaration)

    allocator = SymbolAllocator()
    placeholders = (allocator.new_symbol("cv1"), allocator.new_symbol("cv2"))

    def generalize(node: ResourceCall) -> ResourceCall:
        if isinstance(node, ConjunctionCall) and isinstance(node.controls, tuple):
            if len(node.controls) == 2:
                return replace(node, controls=placeholders)
        return node

    return symbolic_call_graph(
        root, resource_declaration, generalizer=generalize, allocator=allocator
    )


def declared_function_graph(count: SymbolicInt) -> SymbolicCallGraph[ResourceCall]:
    """Declare a symbolic number of ANDs and five CNOTs."""
    root: ResourceCall = MY_FUNCTION

    def declare(
        node: ResourceCall, symbols: SymbolAllocator
    ) -> RawCallees[ResourceCall]:
        if node == root:
            return ((BINARY_AND, 2 * count), (CNOT, 5))
        return resource_declaration(node, symbols)

    return symbolic_call_graph(root, declare)


def swap_decomposition_program() -> ProgramRecord:
    bit = quantum_bit_type()
    cnot = bit_gate_function(
        "CNOT",
        (Register("ctrl", bit), Register("target", bit)),
        Circuit(2).cx(0, 1).to_ir(),
    )
    builder = ComponentProgramBuilder(
        Signature((Register("q0", bit), Register("q1", bit))), (cnot,)
    )
    q0, q1 = builder.call_tuple(
        "CNOT", {"ctrl": builder.inputs["q0"], "target": builder.inputs["q1"]}
    )
    q1, q0 = builder.call_tuple("CNOT", {"ctrl": q1, "target": q0})
    q0, q1 = builder.call_tuple("CNOT", {"ctrl": q0, "target": q1})
    return builder.finalize({"q0": q0, "q1": q1})


def conjunction_counts_agree(values: tuple[int, ...]) -> bool:
    """Compare declared and measured-ladder dependencies under a shared policy."""
    root: ResourceCall = ConjunctionCall(values)

    def decompose(_: ResourceCall, __: SymbolAllocator) -> RawCallees[ResourceCall]:
        return tuple(
            (ConjunctionCall(cvs), 1) for _, _, _, cvs in multi_and_stages(values)
        )

    def ignore_cliffords(node: ResourceCall) -> ResourceCall | None:
        return None if isinstance(node, CliffordCall) else node

    generalizers = (ignore_cliffords, generalize_control_values)
    declared = component_callee_counts(
        root, resource_declaration, generalizer=generalizers
    )
    decomposed = component_callee_counts(root, decompose, generalizer=generalizers)
    return declared == decomposed


def protocol_label(node: ResourceCall) -> str:
    if isinstance(node, ConjunctionCall) and isinstance(node.controls, tuple):
        if len(node.controls) == 2:
            left, right = node.controls
            return f"And({left}, {right})"
        return f"MultiAnd({len(node.controls)})"
    if node == CNOT:
        return "CNOT"
    if node == CliffordCall("x", 1):
        return "X"
    if isinstance(node, ComponentCost):
        return node.name
    return str(node)
