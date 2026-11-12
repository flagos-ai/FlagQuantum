# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Examples adapted from Qualtran v0.7.0 simulation/classical_sim.ipynb.

"""Run python -m examples.algorithms.classical_simulation.

Compose existing numerical owners with the real factorized addition graph.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.component_rebindings import (
    join_word_function,
    split_word_function,
)
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.bit_encoding import bit_value_domain
from flagquantum.simulation.classical_addition import register_add_classical_values
from flagquantum.simulation.classical_and import and_classical_values
from flagquantum.simulation.classical_bit_binding import (
    join_classical_value,
    split_classical_value,
)
from flagquantum.simulation.classical_controlled_bits import cnot_classical_values
from flagquantum.simulation.classical_register_codec import (
    decode_register_bits,
    encode_register_bits,
)
from flagquantum.simulation.classical_truth_table import (
    TruthTable,
    classical_truth_table,
    format_classical_truth_table,
)
from flagquantum.simulation.component_classical_state import ComponentClassicalState
from flagquantum.simulation.integer_values import integer_word_domain

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _word(values: Mapping[str, object], name: str) -> int:
    value = values[name]
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} requires an integer basis value")
    return int(value)


def cnot_values(values: Mapping[str, object]) -> dict[str, object]:
    control, target = cnot_classical_values(
        _word(values, "ctrl"), _word(values, "target")
    )
    return {"ctrl": control, "target": target}


def addition_values(width: int, values: Mapping[str, object]) -> dict[str, object]:
    a, b = register_add_classical_values(_word(values, "a"), _word(values, "b"), width)
    return {"a": a, "b": b}


def addition_state(width: int, values: Mapping[str, object]) -> ComponentClassicalState:
    """Evaluate each stored Add call, including checked carry erasure.

    Split/join recipes must match the canonical Compiler definitions. Numerical
    callbacks are bound explicitly to this example's known component names.
    Unknown recipes fail rather than interpreting arbitrary primitive bodies.
    """
    record = addition_component_program(width)

    def action(function: FunctionRecord, inputs: dict[str, object]) -> object:
        if function.name == "add_cx":
            return cnot_values(inputs)
        if function.name in {"add_and", "add_and_adjoint"}:
            controls = np.asarray(inputs["ctrl"], dtype=np.uint8)
            return and_classical_values(
                controls,
                target=_word(inputs, "target") if "target" in inputs else None,
                uncompute=function.name.endswith("adjoint"),
            )
        assert function.registers is not None
        signature = Signature(function.registers)
        left = tuple(signature.lefts())
        right = tuple(signature.rights())
        if len(left) == len(right) == 1:
            if function == split_word_function(function.name, left[0].dtype):
                return {
                    "reg": split_classical_value(
                        inputs["reg"],
                        lambda value: [
                            int(v)
                            for v in encode_register_bits(
                                left[0], np.asarray(value)
                            ).flat
                        ],
                    )
                }
            if function == join_word_function(function.name, right[0].dtype):
                return {
                    "reg": join_classical_value(
                        np.asarray(inputs["reg"]),
                        lambda bits: decode_register_bits(right[0], np.asarray(bits)),
                    )
                }
        raise ValueError(f"unbound numerical addition recipe {function.name}")

    positions = tuple(i for i, _, _ in iter_component_calls(record))
    return ComponentClassicalState(record, positions, values, action)


def addition_trace(
    width: int, values: Mapping[str, object]
) -> tuple[tuple[str, dict[str, object]], ...]:
    """Return each actual call's committed output values, with native SSA labels."""
    state = addition_state(width, values)
    record = addition_component_program(width)
    functions = {f.name: f for f in record.functions}
    trace: list[tuple[str, dict[str, object]]] = []
    for position, _, _ in iter_component_calls(record):
        state.step()
        operation = record.functions[0].blocks[0].operations[position]
        function = functions[operation.callee or ""]
        refs = iter(operation.results)
        outputs: dict[str, object] = {}
        for register in Signature(function.registers or ()).rights():
            array = np.empty(register.shape, dtype=object)
            for index in register.all_idxs():
                array[index] = state.assignments[next(refs)]
            outputs[register.name] = array if register.shape else array[()]
        trace.append((function.name, outputs))
    return tuple(trace)


def draw_addition_trace(width: int, values: Mapping[str, object]) -> "Figure":
    """Render a call/value table; this is not a Graphviz layout replica."""
    from matplotlib import pyplot as plt

    trace = addition_trace(width, values)
    figure, axes = plt.subplots(figsize=(9, max(3, len(trace) * 0.32)))
    axes.axis("off")
    rows = [
        [
            str(i),
            name if name.startswith("add_") else "Word/bit binding",
            ", ".join(
                f"{k}={np.asarray(v, dtype=np.int64).tolist()}"
                for k, v in outputs.items()
            ),
        ]
        for i, (name, outputs) in enumerate(trace)
    ]
    table = axes.table(
        cellText=rows,
        colLabels=["Call", "Component", "Committed outputs"],
        loc="center",
        cellLoc="left",
        colWidths=[0.08, 0.32, 0.6],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    axes.set_title(f"Addition classical values: {dict(values)}")
    figure.tight_layout()
    return figure


def addition_table(
    width: int,
) -> tuple[list[str], list[str], TruthTable]:
    signature = Signature(
        addition_component_program(width).functions[0].registers or ()
    )
    return classical_truth_table(
        signature,
        lambda r: integer_word_domain(int(r.bitsize)),
        lambda v: tuple(addition_values(width, v).values()),
    )


def cnot_table() -> tuple[list[str], list[str], TruthTable]:
    bit = RegisterType("bit", 1)
    signature = Signature([Register("ctrl", bit), Register("target", bit)])
    return classical_truth_table(
        signature,
        lambda r: bit_value_domain(),
        lambda v: tuple(cnot_values(v).values()),
    )


def main() -> None:
    assert cnot_values({"ctrl": 1, "target": 0}) == {"ctrl": 1, "target": 1}
    assert addition_state(8, {"a": 5, "b": 7}).simulate() == {"a": 5, "b": 12}
    print(format_classical_truth_table(*addition_table(2)))
    print("Native factorized addition agrees with its numerical model.")


if __name__ == "__main__":
    main()
