# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix main_count_operations_sampled.cc.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/cmd/main_count_operations_sampled.cc

"""Format branch-dependent counts without owning their execution."""

from collections import Counter
from collections.abc import Mapping

from .kmx_circuit import KmxCircuit
from .kmx_operation_counts import _DEBUG, _ROWS, _RULES

_FORMAL = {"PUSH_CONDITION", "POP_CONDITION", "INFO_TAG"}
_GROUPS = (("Metadata Operations", ("NEG",)), *_RULES[2:])
_ALL = tuple(op for _, opcodes in _GROUPS for op in opcodes)


def format_sampled_counts(
    circuit: KmxCircuit, totals: Mapping[str, int], shots: int, *, groups: bool = False
) -> str:
    """Render the source rows using integer truncation for decimals."""
    physical = Counter(record.opcode for record in circuit)
    lines = ["                     type,         total,      average,       kept%"]

    def add(label: str, opcodes: tuple[str, ...]) -> None:
        possible = sum(physical[opcode] for opcode in opcodes) * shots
        executed = sum(
            physical[opcode] * shots if opcode in _FORMAL else totals.get(opcode, 0)
            for opcode in opcodes
        )
        mean = executed * 10 // max(shots, 1)
        if possible:
            percent = executed * 1000 // possible
            kept = f"{percent // 10:>9}.{percent % 10}%"
        else:
            kept = " " * 9 + "---"
        lines.append(f"{label:>25},{executed:>14},{mean // 10:>11}.{mean % 10},{kept}")

    for label in _ROWS:
        add(label, _DEBUG if label == _DEBUG[0] else (label,))
    if groups:
        for label, opcodes in _GROUPS:
            add(label, opcodes)
        add("Total Operations", _ALL)
    return "\n".join(lines) + "\n"
