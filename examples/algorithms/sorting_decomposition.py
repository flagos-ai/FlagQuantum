# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 arithmetic/sorting.py.

"""Draw complete sorting source graphs through existing native owners."""

from collections.abc import Mapping

import numpy as np

from examples.algorithms.subtraction_decomposition import subtraction_symbols
from flagquantum.compiler.registers.comparison.compare_exchange_component import (
    compare_exchange_program,
)
from flagquantum.compiler.registers.comparison.parallel_comparison_component import (
    parallel_comparison_program,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.sorting.bitonic_declaration import (
    BitonicMergeDeclaration,
    BitonicSortDeclaration,
)
from flagquantum.compiler.registers.sorting.bitonic_merge_component import (
    bitonic_merge_program,
)
from flagquantum.compiler.registers.sorting.bitonic_sort_component import (
    bitonic_sort_program,
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


def _symbolic_comparator(ports: Signature) -> ProgramRecord:
    word, bit = ports[0].dtype, quantum_bit_type()
    compare = FunctionRecord(
        "GreaterThan",
        (),
        registers=tuple(Signature.build_from_dtypes(a=word, b=word, target=bit)),
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
    builder = ComponentProgramBuilder(ports, (compare, swap))
    history = builder.allocate(dtype=bit)
    result = builder.call_dict(compare.name, {**builder.inputs, "target": history})
    result = builder.call_dict(
        swap.name, {"ctrl": result["target"], "x": result["a"], "y": result["b"]}
    )
    return builder.finalize({"a": result["x"], "b": result["y"], "out": result["ctrl"]})


def _symbolic_sort(ports: Signature, count: int) -> ProgramRecord:
    width = ports[0].bitsize
    half = count // 2
    sort = FunctionRecord(
        "BitonicSort",
        (),
        registers=tuple(BitonicSortDeclaration(half, width).signature),
    )
    merge = FunctionRecord(
        "BitonicMerge",
        (),
        registers=tuple(BitonicMergeDeclaration(half, width).signature),
    )
    builder = ComponentProgramBuilder(ports, (sort, merge))
    xs = np.asarray(builder.inputs["xs"])
    left = builder.call_dict(sort.name, {"xs": xs[:half]})
    right = builder.call_dict(sort.name, {"xs": xs[half:]})
    result = builder.call_dict(merge.name, {"xs": left["xs"], "ys": right["xs"]})
    return builder.finalize(
        {
            "xs": result["result"],
            "junk": np.concatenate(
                (
                    np.asarray(left["junk"]),
                    np.asarray(right["junk"]),
                    np.asarray(result["junk"]),
                )
            ),
        }
    )


def sorting_catalog_program(
    operation: str, ports: Signature, parameters: Mapping[str, object]
) -> ProgramRecord | None:
    """Reuse concrete sorting owners and source symbolic factory declarations.

    Symbolic child definitions retain typed ports with no executable body. The
    symbolic sorting factory has four words and three immediate recursive calls.
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
            return _symbolic_comparator(ports)
        if operation == "BitonicSort" and ports[0].shape == (4,):
            return _symbolic_sort(ports, 4)
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
