# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 HasDuplicates.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/lists/has_duplicates.py
# Changes: explicit wire lifetimes and an operand-preserving cleanup route.

"""Detect adjacent duplicates in a sorted register list."""

from ...core.ir import Instruction
from .bitwise import _validate_wires
from .half_ordering import half_compare_ir, half_compare_outputs
from .multi_control import multi_control_x_ir


def sorted_duplicate_requirements(
    length: int, width: int, *, controlled: bool = False
) -> tuple[int, int]:
    """Return the shared quantum workspace and reusable classical slots."""
    if type(length) is not int or length < 1 or type(width) is not int or width < 1:
        raise ValueError(
            "Duplicate detection requires a nonempty list of nonempty words"
        )
    if type(controlled) is not bool:
        raise ValueError("Duplicate detection controlled mode must be boolean")
    edges = length - 1
    ladder = max(0, edges + controlled - 1)
    return edges * (width + 2) + max(2 if edges else 0, ladder), max(
        width if edges else 0, ladder
    )


def sorted_duplicates_ir(
    words: tuple[tuple[int, ...], ...],
    flag: int,
    work: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    signed: bool = False,
    control: int | None = None,
    source_aliases: bool = False,
) -> tuple[tuple[Instruction, ...], tuple[tuple[int, ...], ...]]:
    """Toggle flag on a duplicate in sorted big-endian words, with clean work.

    The optional control is active on one. Signed words use two's complement.
    The default route retains physical operand order through uncomputation.
    The diagnostic source_aliases route propagates the pinned LT adjoint's
    swapped logical ports, and can violate the And-erasure precondition for
    lists of three or more. It returns the original graph's logical port map.
    An uncontrolled singleton is identity on the default route.
    """
    n = len(words[0]) if words else 0
    q, c = sorted_duplicate_requirements(len(words), n, controlled=control is not None)
    if any(len(word) != n for word in words):
        raise ValueError("Duplicate detection requires equal-width words")
    if type(signed) is not bool or type(source_aliases) is not bool:
        raise ValueError("Duplicate detection modes must be boolean")
    if len(work) != q or len(classical_bits) != c:
        raise ValueError(
            f"Duplicate detection requires {q} work wires and {c} classical slots"
        )
    if len(set(classical_bits)) != c or any(
        type(bit) is not int or bit < 0 for bit in classical_bits
    ):
        raise ValueError(
            "Duplicate detection requires distinct nonnegative classical slots"
        )
    enable = () if control is None else (control,)
    _validate_wires((*enable, flag, *(wire for word in words for wire in word), *work))
    if len(words) == 1 and control is None:
        if source_aliases:
            raise ValueError(
                "Source duplicate detection cannot decompose an uncontrolled singleton"
            )
        return (), words

    edges = len(words) - 1
    history = tuple(work[i : i + n + 2] for i in range(0, edges * (n + 2), n + 2))
    scratch = work[edges * (n + 2) :]
    gates: list[Instruction] = []
    for i, record in enumerate(history):
        gates.extend(
            half_compare_ir(
                words[i],
                words[i + 1],
                record[:-1],
                record[-1],
                scratch[:2],
                relation="lt",
                signed=signed,
            )
        )
    controls = (*enable, *(record[-1] for record in history))
    ladder = len(controls) - 1
    gates.extend(
        multi_control_x_ir(
            controls,
            (1,) * len(controls),
            flag,
            scratch[:ladder],
            classical_bits[:ladder],
        )
    )
    gates.append(
        Instruction("x", (flag,))
        if control is None
        else Instruction("cx", (control, flag))
    )

    outputs = list(words)
    for i in reversed(range(edges)):
        a, b = outputs[i : i + 2]
        record = history[i]
        gates.extend(
            half_compare_ir(
                a,
                b,
                record[:-1],
                record[-1],
                scratch[:2],
                relation="lt",
                signed=signed,
                adjoint=True,
                classical_bits=classical_bits[:n],
            )
        )
        if source_aliases:
            aliases = half_compare_outputs(
                a, b, record[:-1], record[-1], relation="lt", adjoint=True
            )
            outputs[i : i + 2] = [aliases["a"], aliases["b"]]
    return tuple(gates), tuple(outputs)
