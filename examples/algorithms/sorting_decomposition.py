# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 arithmetic/sorting.py.

"""Draw complete sorting source graphs through existing native owners."""

from collections.abc import Mapping

from examples.algorithms.comparison_decomposition import symbolic_greater_than_program
from examples.algorithms.subtraction_decomposition import subtraction_symbols
from flagquantum.compiler.registers.comparison.compare_exchange_component import (
    compare_exchange_program,
)
from flagquantum.compiler.registers.comparison.ordering_declaration import (
    ComparatorDeclaration,
)
from flagquantum.compiler.registers.comparison.parallel_comparison_component import (
    parallel_comparison_program,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.sorting.bitonic_merge_component import (
    bitonic_merge_program,
)
from flagquantum.compiler.registers.sorting.bitonic_sort_component import (
    bitonic_sort_program,
)
from flagquantum.compiler.registers.sorting.network_graph import (
    bitonic_merge_graph,
    bitonic_sort_graph,
    parallel_comparison_graph,
)
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.partition_symbols import partition_wire_symbol
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    RegisterWireSymbolSupplier,
    SymbolClass,
    register_score_data,
)
from flagquantum.drawer.swap_symbols import swap_wire_symbol


def symbolic_comparator_program(ports: Signature) -> ProgramRecord:
    """Compose symbolic source comparison/swap calls with one owned history bit."""
    word, bit = ports[0].dtype, quantum_bit_type()
    compare = symbolic_greater_than_program(
        Signature.build_from_dtypes(a=word, b=word, target=bit)
    )
    swap = FunctionRecord(
        "CSwap",
        (),
        registers=tuple(
            Signature.build_from_dtypes(
                ctrl=bit,
                x=opaque_register_type(word.num_qubits),
                y=opaque_register_type(word.num_qubits),
            )
        ),
    )
    builder = ComponentProgramBuilder(ports, (swap,))
    history = builder.allocate(dtype=bit)
    result = builder.call_program(compare, {**builder.inputs, "target": history})
    result = builder.call_dict(
        swap.name, {"ctrl": result["target"], "x": result["a"], "y": result["b"]}
    )
    return builder.finalize({"a": result["x"], "b": result["y"], "out": result["ctrl"]})


def sorting_catalog_program(
    operation: str, ports: Signature, parameters: Mapping[str, object]
) -> ProgramRecord | None:
    """Reuse the shared finite network builders with concrete or symbolic children.

    Symbolic widths retain source-success dependency graphs down to declared
    comparison/swap leaves. Their unresolved physical storage still rejects.
    """
    if operation not in {
        "Comparator",
        "ParallelComparators",
        "BitonicMerge",
        "BitonicSort",
    }:
        return None
    width = ports[0].bitsize
    if type(width) is not int:
        if operation == "Comparator":
            return symbolic_comparator_program(ports)
        comparator = symbolic_comparator_program(ComparatorDeclaration(width).signature)
        if operation == "BitonicSort" and len(ports[0].shape_symbolic) == 1:
            count = ports[0].shape_symbolic[0]
            if type(count) is int:
                return bitonic_sort_graph(count, comparator)
        if operation == "BitonicMerge" and len(ports[0].shape_symbolic) == 1:
            half = ports[0].shape_symbolic[0]
            if type(half) is int:
                return bitonic_merge_graph(half, comparator)
        if operation == "ParallelComparators":
            count, offset = parameters["k"], parameters["offset"]
            if type(count) is int and type(offset) is int:
                return parallel_comparison_graph(count, offset, comparator)
        return None
    if operation == "Comparator":
        return compare_exchange_program(width)
    if operation == "ParallelComparators":
        count, offset = parameters["k"], parameters["offset"]
        assert type(count) is int and type(offset) is int
        return parallel_comparison_program(count, offset, width)
    if operation == "BitonicMerge":
        half = parameters["half_length"]
        assert type(half) is int
        return bitonic_merge_program(half, width)
    count = parameters["k"]
    assert type(count) is int
    return bitonic_sort_program(count, width)


def sorting_symbols(
    record: ProgramRecord, operation: str
) -> RegisterWireSymbolSupplier:
    """Compose source comparison/swap glyphs and default typed sorting labels."""
    functions = {f.name: f for f in record.functions}
    fallback = subtraction_symbols(record)

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        ports = tuple(r.name for r in functions[step.name].registers or ())
        if ports == ("a", "b", "target"):
            if register is None:
                return "Text", {"text": "a>b", "fontsize": 10}
            return "TextBox", {
                "text": (
                    "⨁(a > b)" if register.name == "target" else f"In({register.name})"
                )
            }
        if ports == ("ctrl", "x", "y"):
            return swap_wire_symbol(step, register, index, controlled=True)
        titles = {
            ("a", "b", "out"): "Comparator",
            ("xs", "junk"): "BitonicSort",
            ("xs", "ys", "result", "junk"): "BitonicMerge",
        }
        if ports in titles:
            title = titles[ports]
            if ports == ("xs", "junk") and operation == "BitonicMerge":
                title = "ParallelComparators"
            if register is None:
                return "Text", {
                    "text": title if len(title) <= 10 else title[:8] + "..",
                    "fontsize": 10,
                }
            return partition_wire_symbol(step, register, index)
        return fallback(step, register, index)

    return supply


def sorting_score(
    operation: str, ports: Signature, parameters: Mapping[str, object]
) -> MusicalScoreData | None:
    """Project the full immediate sorting calls into native register drawings."""
    record = sorting_catalog_program(operation, ports, parameters)
    if record is None:
        return None
    return register_score_data(
        component_register_score(record, record.functions[0].name),
        register_wire_symbol=sorting_symbols(record, operation),
    )
