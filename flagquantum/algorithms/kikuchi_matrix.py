# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 optimization/k_xor_sat/kikuchi_adjacency_matrix.py
# at bcb17bb25360ed68ce3bc084645db3e929799389.

"""Kikuchi adjacency entry declarations assembled from shared algorithm owners."""

from collections.abc import Hashable
from dataclasses import dataclass
from functools import cached_property
from typing import TypeVar

from ..compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ..compiler._component_trace.qec_gate_cost import QECGateCost
from ..compiler.registers.sorting.symmetric_difference_declaration import (
    source_symmetric_difference_resource_formula,
)
from ..core.ir.program.register_types import (
    fixed_point_register_type,
    opaque_register_type,
    quantum_bit_type,
    unsigned_register_type,
)
from ..core.ir.program.registers import RegisterType
from ..core.ir.program.signatures import Signature
from ..core.value_math import SymbolicInt
from .kxor_instance import KXorInstance, example_kxor_instance
from .kxor_loading import LoadUniqueScopeIndex, PRGAUniqueConstraintRHS

_Node = TypeVar("_Node", bound=Hashable)


@dataclass(frozen=True)
class KikuchiMatrixEntry:
    """Declare H[S,T] using the frozen difference/index/RHS child chain.

    The source supplies a query graph and omits a physical composite body.
    Its difference and index compute/uncompute calls bypass outer controls.
    The original signature contains two packed index words and a target qubit.
    """

    inst: KXorInstance
    ell: SymbolicInt
    entry_bitsize: SymbolicInt

    @property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            S=opaque_register_type(self.composite_index_bitsize),
            T=opaque_register_type(self.composite_index_bitsize),
            q=quantum_bit_type(),
        )

    @cached_property
    def index_dtype(self) -> RegisterType:
        return unsigned_register_type(self.inst.index_bitsize)

    @cached_property
    def composite_index_bitsize(self) -> SymbolicInt:
        return self.ell * self.inst.index_bitsize

    @cached_property
    def rotation_angle_dtype(self) -> RegisterType:
        return fixed_point_register_type(self.entry_bitsize, self.entry_bitsize)

    @property
    def scope_index(self) -> LoadUniqueScopeIndex:
        return LoadUniqueScopeIndex(self.inst)

    def rhs(self, *, controlled: bool = False) -> PRGAUniqueConstraintRHS:
        """Use the existing RHS owner with the original precision and control."""
        return PRGAUniqueConstraintRHS(self.inst, self.entry_bitsize, controlled)

    def source_graph(
        self,
        *,
        root: _Node,
        difference: _Node,
        inverse_difference: _Node,
        index: _Node,
        inverse_index: _Node,
        rotation: _Node,
    ) -> SymbolicCallGraph[_Node]:
        """Retain both Always-wrapped compute pairs as distinct caller keys."""
        calls: dict[_Node, SymbolicInt] = {}
        for child in (difference, inverse_difference, index, inverse_index, rotation):
            calls[child] = calls.get(child, 0) + 1
        return symbolic_call_graph(root, {root: calls})

    def resource_gates(
        self, *, controlled: bool = False, adjoint: bool = False
    ) -> QECGateCost:
        """Expand the original direct dependencies and control only the RHS."""
        if controlled and adjoint:
            raise NotImplementedError(
                "Locked source controlled-adjoint matrix cost reaches atomic controlled And"
            )
        difference = source_symmetric_difference_resource_formula(
            self.ell, self.ell, self.inst.k, self.inst.index_bitsize
        )
        return (
            QECGateCost()
            .plus(difference, 2)
            .plus(self.scope_index.resource_gates(), 2)
            .plus(self.rhs(controlled=controlled).resource_gates(adjoint=adjoint), 1)
        )


def example_kikuchi_matrix_entry() -> KikuchiMatrixEntry:
    """Return the original ten-variable instance and ell-eight declaration."""
    return KikuchiMatrixEntry(example_kxor_instance(), 8, 3)


def symbolic_kikuchi_matrix_entry() -> KikuchiMatrixEntry:
    """Return the original n/m/k/c instance with ell=c*k and three angle bits."""
    from sympy import symbols

    n, m, k, c = symbols("n m k c", positive=True, integer=True)
    return KikuchiMatrixEntry(KXorInstance.symbolic(n, m, k), c * k, 3)
