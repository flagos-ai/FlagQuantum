# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 optimization/k_xor_sat/kikuchi_adjacency_list.py
# at bcb17bb25360ed68ce3bc084645db3e929799389.

"""Kikuchi neighbor query declarations and typed composition of explicit children."""

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from ..compiler._component_trace.arithmetic_call import ArithmeticCall
from ..compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ..compiler._component_trace.qec_gate_cost import QECGateCost, query_qec_gate_cost
from ..compiler.registers.arithmetic.arithmetic_declarations import (
    arithmetic_call_graph,
)
from ..compiler.registers.comparison.equality import source_equality_resource_formula
from ..compiler.registers.component_builder import ComponentProgramBuilder
from ..compiler.registers.sorting.symmetric_difference_declaration import (
    source_symmetric_difference_resource_formula,
)
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.register_bindings import RegisterBinding
from ..core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ..core.ir.program.registers import Register, RegisterType
from ..core.ir.program.signatures import Signature
from ..core.value_math import SymbolicInt
from .kxor_instance import KXorInstance, example_kxor_instance

_Node = TypeVar("_Node", bound=Hashable)


def _neighbor_signature(
    inst: KXorInstance, ell: SymbolicInt, *, scratch: bool
) -> Signature:
    dtype = unsigned_register_type(inst.index_bitsize)
    registers = [Register("S", dtype, (ell,)), Register("k", dtype, (ell,))]
    if scratch:
        registers.extend(
            (Register("T", dtype, (ell,)), Register("flag", quantum_bit_type()))
        )
    return Signature(registers)


def _neighbor_resource_gates(inst: KXorInstance, ell: SymbolicInt) -> QECGateCost:
    width, rows = inst.index_bitsize, inst.num_unique_constraints
    difference = source_symmetric_difference_resource_formula(ell, inst.k, ell, width)
    equality = source_equality_resource_formula(width)
    increment = ArithmeticCall("controlled_constant_add", (width,), constant=1)
    addition = query_qec_gate_cost(
        increment, declared_calls=arithmetic_call_graph(increment)
    )
    # Xor(QBit()) is one CNOT. Each forward CNOT occurs again in its adjoint.
    local = QECGateCost(
        toffoli=(rows + 1) * width,
        clifford=1 + 3 * rows,
        and_bloq=rows,
        measurement=rows,
    )
    return (
        local.plus(difference, 2 * rows)
        .plus(equality, 2 * rows)
        .plus(addition, 2 * rows)
    )


@dataclass(frozen=True)
class ColumnOfKthNonZeroEntry:
    """Declare the original counted forward neighbor lookup and its adjoint."""

    inst: KXorInstance
    ell: SymbolicInt

    @property
    def signature(self) -> Signature:
        return _neighbor_signature(self.inst, self.ell, scratch=True)

    @property
    def index_dtype(self) -> RegisterType:
        return unsigned_register_type(self.inst.index_bitsize)

    def adjoint(self) -> "ColumnOfKthNonZeroEntry":
        return self

    def resource_gates(self) -> QECGateCost:
        return _neighbor_resource_gates(self.inst, self.ell)

    def source_graph(
        self,
        *,
        root: _Node,
        controlled_xor: _Node,
        bit_xor: _Node,
        difference: _Node,
        inverse_difference: _Node,
        increment: _Node,
        inverse_increment: _Node,
        equality: _Node,
        inverse_equality: _Node,
        conjunction: _Node,
        erase_conjunction: _Node,
        cnot: _Node,
    ) -> SymbolicCallGraph[_Node]:
        rows = self.inst.num_unique_constraints
        calls: dict[_Node, SymbolicInt] = {}
        for child, count in (
            (controlled_xor, rows + 1),
            (bit_xor, 1),
            (difference, rows),
            (inverse_difference, rows),
            (increment, rows),
            (inverse_increment, rows),
            (equality, rows),
            (inverse_equality, rows),
            (conjunction, rows),
            (erase_conjunction, rows),
            (cnot, 2 * rows),
        ):
            calls[child] = calls.get(child, 0) + count
        return symbolic_call_graph(root, {root: calls})


@dataclass(frozen=True)
class IndexOfNonZeroColumn(ColumnOfKthNonZeroEntry):
    """Retain the separate reverse-query identity with identical declared costs.

    The frozen source uses the same signature, self-adjoint contract and direct
    call multiset for both query classes. Their intended classical roles differ.
    """

    def adjoint(self) -> "IndexOfNonZeroColumn":
        return self


@dataclass(frozen=True)
class KikuchiNonZeroIndex:
    """Compose explicitly supplied forward/reverse functions in source order.

    s remains declared metadata. The original body does not consult it.
    Child decompositions are supplied by their own admitted owners.
    """

    inst: KXorInstance
    ell: SymbolicInt
    s: SymbolicInt

    @property
    def signature(self) -> Signature:
        return _neighbor_signature(self.inst, self.ell, scratch=False)

    @property
    def index_dtype(self) -> RegisterType:
        return unsigned_register_type(self.inst.index_bitsize)

    @property
    def forward(self) -> ColumnOfKthNonZeroEntry:
        return ColumnOfKthNonZeroEntry(self.inst, self.ell)

    @property
    def reverse(self) -> IndexOfNonZeroColumn:
        return IndexOfNonZeroColumn(self.inst, self.ell)

    def resource_gates(self) -> QECGateCost:
        return self.forward.resource_gates().plus(self.reverse.resource_gates(), 1)

    def source_graph(
        self, *, root: _Node, forward: _Node, reverse: _Node
    ) -> SymbolicCallGraph[_Node]:
        calls = {forward: 1}
        calls[reverse] = calls.get(reverse, 0) + 1
        return symbolic_call_graph(root, {root: calls})

    def program(
        self, forward: FunctionRecord, reverse: FunctionRecord
    ) -> ProgramRecord:
        """Bind the complete source root wiring to explicit typed child bodies.

        This preserves the source's positional return assignment, including its
        choice to free reverse T and return reverse k. Free records ownership.
        Numeric zero restoration needs independent child execution evidence.
        """
        expected = tuple(self.forward.signature)
        if forward.registers != expected or reverse.registers != expected:
            raise ValueError("Neighbor child signatures must match S/k/T/flag exactly")
        if not isinstance(self.ell, int) or self.ell < 0:
            raise ValueError("Neighbor composition requires concrete nonnegative ell")
        builder = ComponentProgramBuilder(self.signature, (forward, reverse))
        target = np.array(
            [builder.allocate(dtype=self.index_dtype) for _ in range(self.ell)],
            dtype=object,
        )
        flag = builder.allocate(dtype=quantum_bit_type())
        outputs = builder.call_dict(
            forward.name, {**builder.inputs, "T": target, "flag": flag}
        )
        outputs = builder.call_dict(reverse.name, outputs)
        released = outputs["T"]
        assert isinstance(released, np.ndarray)
        for word in released:
            builder.free(word)
        result_flag = outputs["flag"]
        assert isinstance(result_flag, RegisterBinding)
        builder.free(result_flag)
        return builder.finalize({"S": outputs["S"], "k": outputs["k"]})


def example_column_query() -> ColumnOfKthNonZeroEntry:
    return ColumnOfKthNonZeroEntry(example_kxor_instance(), 8)


def symbolic_column_query() -> ColumnOfKthNonZeroEntry:
    from sympy import symbols

    n, m, k, c, _ = symbols("n m k c s", positive=True, integer=True)
    return ColumnOfKthNonZeroEntry(KXorInstance.symbolic(n, m, k), c * k)


def example_kikuchi_nonzero_index() -> KikuchiNonZeroIndex:
    inst = example_kxor_instance()
    return KikuchiNonZeroIndex(inst, 8, inst.brute_force_sparsity(8))


def symbolic_kikuchi_nonzero_index() -> KikuchiNonZeroIndex:
    from sympy import symbols

    n, m, k, c, s = symbols("n m k c s", positive=True, integer=True)
    return KikuchiNonZeroIndex(KXorInstance.symbolic(n, m, k), c * k, s)
