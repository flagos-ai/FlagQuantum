# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 state_preparation_alias_sampling.py.

"""Concrete and table-free examples for coherent alias preparation."""

from __future__ import annotations

from dataclasses import dataclass

from flagquantum.algorithms.primitives.alias_preparation import (
    alias_black_box_prepare,
    alias_preparation_requirements,
    source_dense_alias_preparation_cost,
    source_sparse_alias_preparation_cost,
)
from flagquantum.algorithms.primitives.alias_sampling import (
    AliasSamplingTable,
    alias_sampling_table,
    sparse_alias_sampling_table,
    sparse_alias_sampling_table_from_dense,
)
from flagquantum.algorithms.primitives.symbolic_alias_preparation import (
    AliasPreparationShape,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.core.ir.program.register_types import opaque_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature


def alias_signature(
    selection: SymbolicInt, mu: SymbolicInt, sparse_index: SymbolicInt | None = None
) -> Signature:
    sparse = () if sparse_index is None else (("sparse_index", sparse_index),)
    return Signature(
        tuple(
            Register(name, opaque_register_type(width))
            for name, width in (
                ("selection", selection),
                ("sigma_mu", mu),
                *sparse,
                ("alt", selection),
                ("keep", mu),
                ("less_than_equal", 1),
            )
        )
    )


@dataclass(frozen=True)
class AliasPreparationExample:
    table: AliasSamplingTable
    adjoint: bool = False

    @property
    def signature(self) -> Signature:
        return alias_signature(
            (self.table.selection_size - 1).bit_length(),
            self.table.mu,
            (
                None
                if self.table.indices is None
                else (len(self.table.keep) - 1).bit_length()
            ),
        )

    @property
    def qubit_count(self) -> int:
        oracle = alias_black_box_prepare(self.table)
        work, _ = alias_preparation_requirements(self.table)
        return oracle.selection_width + oracle.junk_width + work

    def resource_model(self) -> ComponentCost:
        model = (
            source_dense_alias_preparation_cost
            if self.table.indices is None
            else source_sparse_alias_preparation_cost
        )
        return model(self.table)

    def call_graph(self) -> SymbolicCallGraph[str]:
        root = self.resource_model().name
        rows = len(self.table.keep)
        calls = {
            f"PrepareUniformSuperposition({rows})": 1,
            f"OnEach(H, {self.table.mu})": 1,
            "QROM": 1,
            f"LessThanEqual({self.table.mu}, {self.table.mu})": 1,
            f"CSwap({(self.table.selection_size - 1).bit_length()})": 1,
        }
        return symbolic_call_graph(root, {root: calls}, max_depth=1)

    def append_to(self, circuit: Circuit) -> Circuit:
        oracle = alias_black_box_prepare(self.table)
        start = oracle.selection_width
        end = start + oracle.junk_width
        work, slots = alias_preparation_requirements(self.table)
        return oracle.make_on(
            selection=tuple(range(start)),
            junk=tuple(range(start, end)),
            workspace=tuple(range(end, end + work)),
            classical_bits=tuple(range(slots)),
            adjoint=self.adjoint,
        ).append_to(circuit)


def dense_alias_example() -> AliasPreparationExample:
    weights = (1.0, 1.0, 3.0, 2.0)
    return AliasPreparationExample(
        alias_sampling_table(weights, precision=sum(weights) / 32)
    )


def sparse_alias_example(*, from_dense: bool = False) -> AliasPreparationExample:
    if from_dense:
        table = sparse_alias_sampling_table_from_dense(
            (1, 0, 0, 1, 0, 3, 0, 2, 0), precision=1 / 32
        )
    else:
        table = sparse_alias_sampling_table(
            {0: 1, 3: 1, 5: 3, 7: 2}, 9, precision=1 / 32
        )
    return AliasPreparationExample(table)


def symbolic_alias_example(*, sparse: bool = False) -> AliasPreparationShape:
    from sympy import symbols

    rows, support, total, epsilon = symbols(r"L d \lambda \epsilon")
    if sparse:
        return AliasPreparationShape.from_sparse_precision(
            rows, support, total, epsilon
        )
    return AliasPreparationShape.from_precision(rows, total, epsilon)
