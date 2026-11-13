# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the frozen Qualtran v0.7.0 addition notebook.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.ipynb

"""Display all ten addition examples using existing native metadata and models.

Run python -m examples.algorithms.addition_catalog. This catalog emits no gates.
Executable addition remains in classical_register_addition and uses append.
"""

from collections.abc import Sequence
from dataclasses import replace
from html import escape

from sympy import symbols

from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.resource_generalization import ResourceCall
from flagquantum.compiler.registers.arithmetic.arithmetic_declarations import (
    arithmetic_call_graph,
)
from flagquantum.core.ir.program.register_types import (
    signed_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.addition_symbols import addition_wire_symbol
from flagquantum.drawer.call_graph import render_call_graph_svg
from flagquantum.drawer.register_graph_table import signature_instance_table
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import MusicalScoreData, register_score_data
from flagquantum.drawer.signature_graph import signature_register_score


def addition_examples() -> tuple[tuple[str, ArithmeticCall, Signature], ...]:
    """Keep the locked notebook's widths, encodings, constants and instance order."""
    n, k = symbols("n k")
    calls = (
        ("add_symb", ArithmeticCall("add", (n, n), ("twos_complement",) * 2)),
        ("add_small", ArithmeticCall("add", (4, 4))),
        ("add_large", ArithmeticCall("add", (64, 64))),
        ("add_diff_size_regs", ArithmeticCall("add", (4, 16))),
        ("add_oop_symb", ArithmeticCall("out_of_place_add", (n,))),
        ("add_oop_small", ArithmeticCall("out_of_place_add", (4,))),
        ("add_oop_large", ArithmeticCall("out_of_place_add", (64,))),
        ("add_k", ArithmeticCall("constant_add", (n,), constant=k)),
        ("add_k_small", ArithmeticCall("constant_add", (4,), constant=2)),
        (
            "add_k_large",
            ArithmeticCall("constant_add", (64,), ("twos_complement",), constant=-23),
        ),
    )
    result = []
    for name, call in calls:
        dtype = (
            signed_register_type
            if call.encodings and call.encodings[0] == "twos_complement"
            else unsigned_register_type
        )
        width = call.widths[0]
        if call.operation == "add":
            signature = Signature(
                (Register("a", dtype(width)), Register("b", dtype(call.widths[1])))
            )
        elif call.operation == "out_of_place_add":
            signature = Signature(
                (
                    Register("a", dtype(width)),
                    Register("b", dtype(width)),
                    Register("c", dtype(width + 1), side=RegisterSide.RIGHT),
                )
            )
        else:
            signature = Signature((Register("x", dtype(width)),))
        result.append((name, call, signature))
    return tuple(result)


def addition_example_graph(call: ArithmeticCall) -> SymbolicCallGraph[ResourceCall]:
    """Show the source notebook's one-level declared call graph."""
    return symbolic_call_graph(call, arithmetic_call_graph(call), max_depth=1)


def addition_signature_svg(
    examples: Sequence[tuple[str, ArithmeticCall, Signature]],
) -> bytes:
    """Render symbolic and concrete signature tables through existing Drawer owners.

    Tables retain register order, direction and dtype widths. The native table
    layout replaces the source musical-score layout without allocating wires.
    """
    nodes = []
    for index, (name, call, signature) in enumerate(examples):
        inputs = tuple(register.name for register in signature.lefts())
        outputs = tuple(register.name for register in signature.rights())
        step = RegisterStep(name, inputs, outputs)
        ports = {register.name: f"p{i}" for i, register in enumerate(signature)}
        table = signature_instance_table(signature, step, ports)
        metadata = ", ".join(
            f"{reg.name}={reg.dtype.encoding}[{reg.bitsize}]" for reg in signature
        )
        if call.constant is not None:
            metadata += f", k={call.constant}"
        table = table.replace(
            "</TABLE>", f'<TR><TD COLSPAN="2">{escape(metadata)}</TD></TR></TABLE>'
        )
        nodes.append(f"n{index} [shape=plain, label=<{table}>];")
    return render_call_graph_svg("digraph signatures { " + " ".join(nodes) + " }")


def addition_signature_data(
    example: tuple[str, ArithmeticCall, Signature],
) -> MusicalScoreData:
    """Use native addition glyphs for a concrete signature's existing score."""
    name, call, signature = example
    title = {
        "add": "Add",
        "out_of_place_add": "OutOfPlaceAdder",
        "constant_add": "AddK",
    }[call.operation]
    score = signature_register_score(signature, title)
    column = score.columns[0]
    column = replace(column, step=replace(column.step, gate=call.operation))
    return register_score_data(
        replace(score, columns=(column,)), wire_symbol=addition_wire_symbol
    )


def main() -> None:
    for name, call, _ in addition_examples():
        print(name, addition_example_graph(call).leaf_counts)
    print("Signature SVG bytes", len(addition_signature_svg(addition_examples())))


if __name__ == "__main__":
    main()
