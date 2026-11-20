# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 for_testing/interior_alloc.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/for_testing/interior_alloc.py

"""Construct the interior allocation model and its finite typed Program."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping

from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.logical_width import query_logical_width
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler._component_trace.width_schedule import (
    WidthConnection,
    WidthStep,
    estimate_connection_width,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.primitives.swap import swap_bit_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature


def interior_workspace_signature(width: SymbolicInt) -> Signature:
    return Signature.build(x=width, y=width)


def _word_swap(width: SymbolicInt) -> ProgramRecord:
    if not isinstance(width, int):
        return ProgramRecord(
            (
                FunctionRecord(
                    "swap_word",
                    (),
                    registers=tuple(interior_workspace_signature(width)),
                ),
            )
        )
    bit = quantum_bit_type()
    dtype = opaque_register_type(width)
    swap = bit_gate_function(
        "swap_bit",
        (Register("x", bit), Register("y", bit)),
        CircuitIR(2, swap_bit_ir(0, 1)),
    )
    builder = ComponentProgramBuilder(
        interior_workspace_signature(width),
        (swap,),
        name="swap_word",
    )
    x, y = builder.inputs["x"], builder.inputs["y"]
    assert isinstance(x, RegisterBinding) and isinstance(y, RegisterBinding)
    pairs = [
        builder.call_dict("swap_bit", {"x": left, "y": right})
        for left, right in zip(builder.split(x), builder.split(y), strict=True)
    ]
    return builder.finalize(
        {
            name: builder.join([pair[name] for pair in pairs], dtype)
            for name in ("x", "y")
        }
    )


def interior_workspace_fixture(width: SymbolicInt) -> ProgramRecord:
    """Route x into y on the clean-y input subspace; the middle word is freed."""
    dtype = opaque_register_type(width)
    signature = interior_workspace_signature(width)
    builder = ComponentProgramBuilder(signature, _word_swap(width).functions)
    middle = builder.allocate(dtype=dtype)
    first = builder.call_dict("swap_word", {"x": builder.inputs["x"], "y": middle})
    second = builder.call_dict("swap_word", {"x": first["y"], "y": builder.inputs["y"]})
    assert isinstance(second["x"], RegisterBinding)
    builder.free(second["x"])
    return builder.finalize({"x": first["x"], "y": second["y"]})


def interior_call_kinds(record: ProgramRecord) -> tuple[str, ...]:
    functions: dict[str, FunctionRecord] = {fn.name: fn for fn in record.functions}
    names = []
    for call in functions["main"].blocks[0].operations:
        assert call.callee is not None
        callee = functions[call.callee]
        opcode = callee.blocks[0].operations[0].opcode
        names.append(
            "Swap"
            if callee.name == "swap_word"
            else {"allocate_register": "Allocate", "free_register": "Free"}[opcode]
        )
    return tuple(names)


def interior_width_connections(
    width: SymbolicInt, query_child: Callable[[str], SymbolicInt]
) -> tuple[tuple[WidthStep, ...], tuple[WidthConnection, ...]]:
    """Declare the symbolic ownership path of the two serial word swaps."""
    steps = (
        WidthStep("input"),
        WidthStep("allocate", query_child("Allocate")),
        WidthStep("component", query_child("Swap")),
        WidthStep("component", query_child("Swap")),
        WidthStep("free", query_child("Free")),
        WidthStep("output"),
    )
    edges = (
        WidthConnection(0, 2, width),
        WidthConnection(1, 2, width),
        WidthConnection(2, 5, width),
        WidthConnection(2, 3, width),
        WidthConnection(0, 3, width),
        WidthConnection(3, 4, width),
        WidthConnection(3, 5, width),
    )
    return steps, edges


def interior_width_call_graph(width: SymbolicInt) -> SymbolicCallGraph[str]:
    return symbolic_call_graph("Interior allocation", _declared_calls(width))


def _declared_calls(width: SymbolicInt) -> dict[str, dict[str, SymbolicInt]]:
    return {
        "Interior allocation": {"Allocate": 1, "Swap": 2, "Free": 1},
        "Swap": {"TwoBitSwap": width},
    }


def interior_width_model(
    width: SymbolicInt, *, cache: MutableMapping[str, SymbolicInt] | None = None
) -> SymbolicInt:
    """Compose the symbolic decomposition with existing recursive width queries."""
    root = "Interior allocation"
    signatures: dict[str, SymbolicInt] = {
        root: interior_workspace_signature(width).n_qubits(),
        "Allocate": width,
        "Free": width,
        "Swap": 2 * width,
        "TwoBitSwap": 2,
    }
    calls = _declared_calls(width)

    def decomposition(
        node: str, child: Callable[[str], SymbolicInt]
    ) -> SymbolicInt | None:
        if node != root:
            return None
        return estimate_connection_width(*interior_width_connections(width, child))

    return query_logical_width(
        root,
        signature_width=signatures.__getitem__,
        callees=lambda node: tuple(calls.get(node, {}).items()),
        decomposition_width=decomposition,
        cache=cache,
    )
