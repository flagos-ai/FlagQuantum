# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 optimization/k_xor_sat/kikuchi_block_encoding.py
# at bcb17bb25360ed68ce3bc084645db3e929799389.

"""Kikuchi Hamiltonian declarations and typed sparse-oracle composition."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable
from dataclasses import dataclass
from typing import TypeVar

from ..compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ..compiler._component_trace.qec_gate_cost import QECGateCost
from ..compiler.registers.component_builder import ComponentProgramBuilder
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.register_types import (
    bounded_unsigned_register_type,
    unsigned_register_type,
)
from ..core.ir.program.signatures import Signature
from ..core.value_math import SymbolicInt
from ..core.value_parameters import is_symbolic_value
from .kikuchi_adjacency import KikuchiNonZeroIndex
from .kikuchi_matrix import KikuchiMatrixEntry
from .kxor_instance import KXorInstance, example_kxor_instance
from .primitives.black_box_prepare import PackedPrepareDeclaration
from .primitives.hermitian_sparse_matrix_block_encoding import (
    source_sqrt_entry_signature,
)
from .primitives.identity_prepare import IdentityPrepare
from .primitives.sparse_matrix_block_encoding import (
    _source_sparse_diffusion_resource_formula,
)

_Node = TypeVar("_Node", bound=Hashable)


@dataclass(frozen=True)
class KikuchiEntryOracle:
    """Rename the matrix-entry ports to the shared sparse-entry interface."""

    entry: KikuchiMatrixEntry

    @property
    def system_bitsize(self) -> SymbolicInt:
        return self.entry.composite_index_bitsize

    @property
    def signature(self) -> Signature:
        return source_sqrt_entry_signature(self.system_bitsize)

    @property
    def epsilon(self) -> int:
        return 0

    @property
    def phase_gradient_bitsize(self) -> SymbolicInt:
        return self.entry.entry_bitsize

    def resource_gates(self, *, adjoint: bool = False) -> QECGateCost:
        return self.entry.resource_gates(adjoint=adjoint)

    def source_graph(self, *, root: _Node, entry: _Node) -> SymbolicCallGraph[_Node]:
        return symbolic_call_graph(root, {root: {entry: 1}})

    def program(self, entry: FunctionRecord) -> ProgramRecord:
        """Bind an admitted matrix body and preserve q/i/j output ordering."""
        if entry.registers != tuple(self.entry.signature):
            raise ValueError("Entry body must match the matrix S/T/q signature")
        builder = ComponentProgramBuilder(self.signature, (entry,))
        outputs = builder.call_dict(
            entry.name,
            {
                "S": builder.inputs["i"],
                "T": builder.inputs["j"],
                "q": builder.inputs["q"],
            },
        )
        return builder.finalize(
            {"q": outputs["q"], "i": outputs["S"], "j": outputs["T"]}
        )


@dataclass(frozen=True)
class KikuchiRowColumnOracle:
    """Retain the source packed ordinal/index wrapper and its array boundary."""

    neighbor: KikuchiNonZeroIndex

    @property
    def system_bitsize(self) -> SymbolicInt:
        return self.neighbor.index_dtype.num_qubits * self.neighbor.ell

    @property
    def num_nonzero(self) -> SymbolicInt:
        return self.neighbor.s

    @property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            l=bounded_unsigned_register_type(self.system_bitsize, self.num_nonzero),
            i=unsigned_register_type(self.system_bitsize),
        )

    def resource_gates(self) -> QECGateCost:
        return self.neighbor.resource_gates()

    def source_graph(self, *, root: _Node, neighbor: _Node) -> SymbolicCallGraph[_Node]:
        return symbolic_call_graph(root, {root: {neighbor: 1}})

    def program(self, neighbor: FunctionRecord) -> ProgramRecord:
        """Attempt the literal source call through the shared shape verifier.

        The source passes packed scalars to array-valued neighbor ports. This
        fails shape admission until a separately justified packing contract exists.
        """
        if neighbor.registers != tuple(self.neighbor.signature):
            raise ValueError("Neighbor body must match the array S/k signature")
        builder = ComponentProgramBuilder(self.signature, (neighbor,))
        outputs = builder.call_dict(
            neighbor.name, {"S": builder.inputs["i"], "k": builder.inputs["l"]}
        )
        return builder.finalize({"l": outputs["k"], "i": outputs["S"]})


@dataclass(frozen=True)
class _SparseResourceCall:
    """Keep sparse-query identity and direction until resource folding."""

    component: (
        KikuchiEntryOracle
        | KikuchiRowColumnOracle
        | tuple[SymbolicInt, SymbolicInt]
        | SymbolicInt
    )
    adjoint: bool = False
    controlled: bool = False


@dataclass(frozen=True, init=False)
class KikuchiHamiltonian:
    """Declare the source sparse Hamiltonian using domain-owned query modules."""

    inst: KXorInstance
    ell: SymbolicInt
    entry_bitsize: SymbolicInt
    s: SymbolicInt

    def __init__(
        self,
        inst: KXorInstance,
        ell: SymbolicInt,
        entry_bitsize: SymbolicInt | None = None,
        s: SymbolicInt | None = None,
    ) -> None:
        if entry_bitsize is None:
            if not is_symbolic_value(inst.max_rhs) and inst.max_rhs != 2:
                raise ValueError(
                    "Entries outside range [-2, 2], please specify an explicit entry_bitsize."
                )
            entry_bitsize = 3
        object.__setattr__(self, "inst", inst)
        object.__setattr__(self, "ell", ell)
        object.__setattr__(self, "entry_bitsize", entry_bitsize)
        object.__setattr__(self, "s", inst.num_unique_constraints if s is None else s)

    @property
    def matrix_entry(self) -> KikuchiMatrixEntry:
        return KikuchiMatrixEntry(self.inst, self.ell, self.entry_bitsize)

    @property
    def neighbor_query(self) -> KikuchiNonZeroIndex:
        return KikuchiNonZeroIndex(self.inst, self.ell, self.s)

    @property
    def entry_oracle(self) -> KikuchiEntryOracle:
        return KikuchiEntryOracle(self.matrix_entry)

    @property
    def column_oracle(self) -> KikuchiRowColumnOracle:
        return KikuchiRowColumnOracle(self.neighbor_query)

    @property
    def system_bitsize(self) -> SymbolicInt:
        return self.entry_oracle.system_bitsize

    @property
    def ancilla_bitsize(self) -> SymbolicInt:
        return self.system_bitsize + 2

    @property
    def resource_bitsize(self) -> int:
        return 0

    @property
    def alpha(self) -> SymbolicInt:
        return self.column_oracle.num_nonzero

    @property
    def epsilon(self) -> int:
        return self.entry_oracle.epsilon

    @property
    def signature(self) -> Signature:
        return Signature.build(
            system=self.system_bitsize,
            ancilla=self.ancilla_bitsize,
            resource=self.resource_bitsize,
        )

    @property
    def signal_state(self) -> PackedPrepareDeclaration:
        return PackedPrepareDeclaration(
            IdentityPrepare.from_bitsizes((self.ancilla_bitsize,))
        )

    def __str__(self) -> str:
        return "B[K_l]"

    def source_graph(
        self, *, root: _Node, sparse_encoding: _Node
    ) -> SymbolicCallGraph[_Node]:
        return symbolic_call_graph(root, {root: {sparse_encoding: 1}})

    def source_sparse_calls(
        self, *, controlled: bool = False
    ) -> dict[_SparseResourceCall, int]:
        """Retain the eight Hermitian sparse calls, including both swap widths."""
        if type(controlled) is not bool:
            raise ValueError("Hamiltonian control flag must be boolean")
        diffusion = (self.system_bitsize, self.s)
        return dict(
            Counter(
                (
                    _SparseResourceCall(diffusion),
                    _SparseResourceCall(self.column_oracle),
                    _SparseResourceCall(self.entry_oracle),
                    _SparseResourceCall(self.system_bitsize, controlled=controlled),
                    _SparseResourceCall(1, controlled=controlled),
                    _SparseResourceCall(self.entry_oracle, adjoint=True),
                    _SparseResourceCall(self.column_oracle, adjoint=True),
                    _SparseResourceCall(diffusion, adjoint=True),
                )
            )
        )

    def resource_gates(self, *, controlled: bool = False) -> QECGateCost:
        """Price the source inventory with existing query and diffusion models."""
        cost = QECGateCost()
        for call, count in self.source_sparse_calls(controlled=controlled).items():
            component = call.component
            if isinstance(component, KikuchiEntryOracle):
                child = component.resource_gates(adjoint=call.adjoint)
            elif isinstance(component, KikuchiRowColumnOracle):
                child = component.resource_gates()
            elif isinstance(component, tuple):
                cost = cost.plus(
                    _source_sparse_diffusion_resource_formula(
                        *component, legacy_shims=False
                    ),
                    count,
                )
                continue
            else:
                child = (
                    QECGateCost(cswap=component)
                    if call.controlled
                    else QECGateCost(clifford=component)
                )
            cost = cost.plus(child, count)
        return cost

    def program(self, sparse_encoding: FunctionRecord) -> ProgramRecord:
        """Bind an admitted sparse-encoding body with the original named ports."""
        if sparse_encoding.registers != tuple(self.signature):
            raise ValueError("Sparse body must match the Hamiltonian signature")
        builder = ComponentProgramBuilder(self.signature, (sparse_encoding,))
        return builder.finalize(builder.call_dict(sparse_encoding.name, builder.inputs))


def example_kikuchi_hamiltonian() -> KikuchiHamiltonian:
    return KikuchiHamiltonian(example_kxor_instance(), 8)


def symbolic_kikuchi_hamiltonian() -> KikuchiHamiltonian:
    from sympy import symbols

    n, m, k, c = symbols("n m k c", positive=True, integer=True)
    return KikuchiHamiltonian(KXorInstance.symbolic(n, m, k), c * k)
