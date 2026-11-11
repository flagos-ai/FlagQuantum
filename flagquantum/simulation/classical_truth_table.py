# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 get_classical_truth_table and format_classical_truth_table.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/classical_sim.py

"""Enumerate bound scalar value domains and format classical action tables."""

from collections.abc import Callable, Iterable, Sequence
from itertools import product
from typing import TypeAlias

from ..core.ir.program.registers import Register
from ..core.ir.program.signatures import Signature
from .classical_domains import register_classical_domain

TruthTable: TypeAlias = list[tuple[Sequence[object], Sequence[object]]]


def classical_truth_table(
    signature: Signature,
    domain: Callable[[Register], Iterable[object]],
    evaluate: Callable[[dict[str, object]], Sequence[object]],
) -> tuple[list[str], list[str], TruthTable]:
    """Enumerate scalar inputs in signature order using existing domain owners.

    All input shapes are checked before querying any domain. Output shapes and
    returned sequences are retained without coercion, arity checks or copying.
    The evaluator owns value/phase semantics; it may consume an existing native
    classical graph state. Empty input signatures evaluate once, while an empty
    domain suppresses evaluation. Domain pools and the full table are materialized,
    so storage grows with the Cartesian product, as in the source.

    Args:
        signature: Native ordered input/output register metadata.
        domain: A numerical provider for each input register's scalar values.
        evaluate: A bound numerical action returning outputs in signature order.

    Raises:
        NotImplementedError: An input register has an array shape.
    """
    for register in signature.lefts():
        if register.shape:
            raise NotImplementedError()
    in_names: list[str] = []
    domains = []
    for register in signature.lefts():
        in_names.append(register.name)
        domains.append(domain(register))
    out_names = [register.name for register in signature.rights()]
    table: TruthTable = []
    for inputs in product(*domains):
        values = {name: value for name, value in zip(in_names, inputs, strict=False)}
        table.append((inputs, evaluate(values)))
    return in_names, out_names, table


def format_classical_truth_table(
    in_names: Sequence[str],
    out_names: Sequence[str],
    truth_table: Sequence[tuple[Sequence[object], Sequence[object]]],
) -> str:
    """Preserve source headings, separator length, value formatting and newlines."""
    heading = "  ".join(in_names) + "  |  " + "  ".join(out_names) + "\n"
    heading += "-" * len(heading)
    entries = [
        ", ".join(f"{value}" for value in inputs)
        + " -> "
        + ", ".join(f"{value}" for value in outputs)
        for inputs, outputs in truth_table
    ]
    return "\n".join([heading] + entries)


def register_classical_truth_table(
    signature: Signature,
    evaluate: Callable[[dict[str, object]], Sequence[object]],
) -> tuple[list[str], list[str], TruthTable]:
    """Enumerate existing native dtype domains through the shared table owner.

    Shape refusal, domain acquisition, Cartesian order and evaluator errors keep
    classical_truth_table's behavior. The lower-level entry remains available
    for explicit domains. No stochastic or phase policy is inferred here.
    """
    return classical_truth_table(signature, register_classical_domain, evaluate)
