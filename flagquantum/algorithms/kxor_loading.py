# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 optimization/k_xor_sat/load_kxor_instance.py at
# bcb17bb25360ed68ce3bc084645db3e929799389.

"""k-XOR scope loading and the original index/RHS oracle declarations.

Scope loading binds the shared measured QROM body. Index and RHS oracles retain
the source's declaration-only execution boundary and explicit child parameters.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

from ..compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ..compiler._component_trace.qec_gate_cost import QECGateCost, query_qec_gate_cost
from ..compiler._component_trace.resource_generalization import (
    ControlledRotationCall,
    PhaseCall,
)
from ..compiler._component_trace.resource_model import estimate_component_resources
from ..compiler.registers.comparison.comparison_resource_model import (
    source_less_than_constant_qec_cost,
)
from ..compiler.registers.comparison.equality import (
    source_equality_constant_resource_formula,
)
from ..compiler.registers.lookup.qrom_table import QromTable
from ..compiler.registers.resources.symbolic_resource_model import (
    source_controlled_add_or_subtract_resource_formula,
)
from ..core.ir.program.register_types import (
    bounded_unsigned_register_type,
    fixed_point_register_type,
    opaque_register_type,
    quantum_bit_type,
)
from ..core.ir.program.registers import Register, RegisterSide, RegisterType
from ..core.ir.program.signatures import Signature
from ..core.value_math import SymbolicInt, ceil, log2
from ..core.value_parameters import is_symbolic_value
from ..simulation.value_dimensions import SymbolicSequenceLength
from .kxor_instance import Constraint, KXorInstance
from .primitives._block_binding import _BoundBlockOperation, _validate_block_work

_Node = TypeVar("_Node", bound=Hashable)


def _index_dtype(inst: KXorInstance) -> RegisterType:
    rows = inst.num_unique_constraints
    return bounded_unsigned_register_type(ceil(log2(rows)), rows)


@dataclass(frozen=True)
class LoadConstraintScopes:
    """Load each sorted scope into a clean big-endian packed output.

    The caller owns the output, clean QROM workspace and feedback slots. Applying
    the same XOR lookup twice clears the output on valid indices.
    """

    inst: KXorInstance

    @cached_property
    def scope_bitsize(self) -> SymbolicInt:
        return self.inst.k * self.inst.index_bitsize

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        return 0

    @cached_property
    def m_dtype(self) -> RegisterType:
        return _index_dtype(self.inst)

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            (
                Register("j", self.m_dtype),
                Register(
                    "U",
                    opaque_register_type(self.scope_bitsize),
                    side=RegisterSide.RIGHT,
                ),
            )
        )

    @cached_property
    def qrom(self) -> QromTable:
        if self.inst.is_symbolic():
            # The original symbolic model has a single packed target. The
            # concrete file uses k independently packed scalar targets.
            return QromTable.from_shape(
                self.inst.num_unique_constraints, self.scope_bitsize
            )
        scopes = self.inst.batched_scopes
        assert isinstance(scopes, tuple) and isinstance(self.inst.k, int)
        assert isinstance(self.inst.index_bitsize, int)
        data = np.array([scope for scope, _ in scopes], dtype=int)
        assert data.shape == (self.inst.num_unique_constraints, self.inst.k)
        return QromTable.from_data(
            *data.T, target_bitsizes=(self.inst.index_bitsize,) * self.inst.k
        )

    def source_graph(self, *, root: _Node, lookup: _Node) -> SymbolicCallGraph[_Node]:
        return symbolic_call_graph(root, {root: {lookup: 1}})

    def make_on(
        self,
        j: tuple[int, ...],
        scope: tuple[int, ...],
        *,
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
    ) -> _BoundBlockOperation:
        """Return the measured lookup operation consumed by circuit.append(op)."""
        if self.inst.is_symbolic():
            raise ValueError("Scope loading requires a concrete instance")
        if len(j) != self.m_dtype.num_qubits or len(scope) != self.scope_bitsize:
            raise ValueError("Scope loader ports must match their declared widths")
        ports = (*j, *scope)
        _validate_block_work(ports, workspace, classical_bits)
        instructions = self.qrom.bind_packed(ports, workspace, classical_bits)
        return _BoundBlockOperation(
            instructions, max((*ports, *workspace), default=-1) + 1
        )


@dataclass(frozen=True)
class LoadUniqueScopeIndex:
    """Declare the source's scope-to-index equality queries.

    Both source ports are THRU, including j despite its RIGHT prose description.
    The source provides a query graph with an unspecified constant placeholder.
    """

    inst: KXorInstance

    @cached_property
    def scope_bitsize(self) -> SymbolicInt:
        return self.inst.k * self.inst.index_bitsize

    @cached_property
    def m_dtype(self) -> RegisterType:
        return _index_dtype(self.inst)

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            j=self.m_dtype, U=opaque_register_type(self.scope_bitsize)
        )

    def resource_gates(self) -> QECGateCost:
        """Expand the source's symbolic equality placeholder using its owner."""
        from sympy import Symbol

        equality = source_equality_constant_resource_formula(
            self.scope_bitsize, Symbol("_c0")
        )
        return QECGateCost().plus(equality, self.inst.num_unique_constraints)

    def source_graph(self, *, root: _Node, equality: _Node) -> SymbolicCallGraph[_Node]:
        return symbolic_call_graph(
            root, {root: {equality: self.inst.num_unique_constraints}}
        )


@dataclass(frozen=True)
class PRGAUniqueConstraintRHS:
    """Declare the original H/Rz/sign-correction and QROM/unload schedule.

    The source supplies a resource graph. Its signature omits the phase-gradient
    register required by its Rz child and has no executable composite body.
    """

    inst: KXorInstance
    angle_bitsize: SymbolicInt
    is_controlled: bool = False

    @property
    def n_ctrl(self) -> int:
        return 1 if self.is_controlled else 0

    @cached_property
    def m_dtype(self) -> RegisterType:
        return _index_dtype(self.inst)

    @cached_property
    def angle_dtype(self) -> RegisterType:
        return fixed_point_register_type(self.angle_bitsize, self.angle_bitsize)

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            ctrl=opaque_register_type(self.n_ctrl), j=self.m_dtype, q=quantum_bit_type()
        )

    @cached_property
    def angle_data(
        self,
    ) -> tuple[
        NDArray[np.float64] | SymbolicSequenceLength,
        tuple[float, ...] | SymbolicSequenceLength,
    ]:
        maximum, scopes = self.inst.max_rhs, self.inst.batched_scopes
        if is_symbolic_value(maximum, scopes):
            length = SymbolicSequenceLength(self.inst.num_unique_constraints)
            return length, length
        assert isinstance(scopes, tuple)
        values = np.array([rhs for _, rhs in scopes])
        assert np.all(values == np.sort(values)), "data must be sorted!"
        angles = np.arccos(np.sqrt(np.abs(values) / maximum))
        rounded: NDArray[np.float64] = np.round(angles * 2**self.angle_bitsize)
        return rounded, tuple(np.sign(values))

    @cached_property
    def amplitude_qrom(self) -> QromTable:
        data, _ = self.angle_data
        if isinstance(data, SymbolicSequenceLength):
            return QromTable.from_shape(
                self.inst.num_unique_constraints,
                self.angle_bitsize,
                num_controls=self.n_ctrl,
            )
        assert isinstance(self.angle_bitsize, int)
        return QromTable.from_data(
            data, target_bitsizes=(self.angle_bitsize,), num_controls=self.n_ctrl
        )

    @cached_property
    def num_negative(self) -> SymbolicInt:
        _, signs = self.angle_data
        if isinstance(signs, SymbolicSequenceLength):
            return self.inst.num_unique_constraints // 2
        assert np.all(signs == np.sort(signs)), "data must be sorted!"
        return int(np.searchsorted(signs, 0))

    def resource_gates(self, *, adjoint: bool = False) -> QECGateCost:
        """Compose original QEC categories independently of physical lookup gates."""
        if type(adjoint) is not bool:
            raise ValueError("RHS model adjoint must be boolean")
        table = self.amplitude_qrom
        lookup = (
            table.resource_formula()
            if table.datasets is None
            else estimate_component_resources(table.resource_model()).gates
        )
        rotation = source_controlled_add_or_subtract_resource_formula(
            self.angle_bitsize, self.angle_bitsize, add_when_control_is_on=not adjoint
        )
        compare = source_less_than_constant_qec_cost(
            self.m_dtype.num_qubits, self.num_negative
        )
        sign = query_qec_gate_cost(
            ControlledRotationCall(PhaseCall("s", adjoint=adjoint), (1,))
        )
        return (
            QECGateCost(clifford=2)
            .plus(lookup, 2)
            .plus(rotation, 1)
            .plus(compare, 1)
            .plus(sign, 1)
        )

    def with_positive_control(self) -> PRGAUniqueConstraintRHS:
        """Return the source's explicitly supplied single-positive specialization."""
        return replace(self, is_controlled=True)

    def source_control_route(
        self, values: tuple[int, ...]
    ) -> Literal["generic", "conjunction", "specialized"]:
        """Retain specialized, conjunction and generic-fallback selection."""
        if not values or any(
            type(value) is not int or value not in (0, 1) for value in values
        ):
            raise ValueError("RHS controls require a nonempty binary activation tuple")
        if len(values) != 1:
            return "generic"
        if self.is_controlled:
            return "conjunction"
        return "specialized" if values == (1,) else "generic"

    def source_graph(
        self,
        *,
        root: _Node,
        lookup: _Node,
        hadamard: _Node,
        rotation: _Node,
        sign_compare: _Node,
        controlled_s: _Node,
        inverse_lookup: _Node,
    ) -> SymbolicCallGraph[_Node]:
        calls: dict[_Node, SymbolicInt] = {}
        for child, count in (
            (lookup, 1),
            (hadamard, 2),
            (rotation, 1),
            (sign_compare, 1),
            (controlled_s, 1),
            (inverse_lookup, 1),
        ):
            calls[child] = calls.get(child, 0) + count
        return symbolic_call_graph(root, {root: calls})


def example_scope_loader() -> LoadConstraintScopes:
    """Return the source's complete six-variable, eight-clause example."""
    constraints = (
        Constraint((0, 1, 2, 3), 1),
        Constraint((0, 1, 4, 5), -1),
        Constraint((1, 2, 4, 5), 1),
        Constraint((0, 3, 4, 5), 1),
        Constraint((2, 3, 4, 5), 1),
        Constraint((0, 1, 2, 3), 1),
        Constraint((0, 3, 4, 5), 1),
        Constraint((2, 3, 4, 5), 1),
    )
    return LoadConstraintScopes(KXorInstance(6, 4, constraints))


def symbolic_scope_loader() -> LoadConstraintScopes:
    """Return the original n/m/k shape-only factory."""
    from sympy import symbols

    n, m, k = symbols("n m k", positive=True, integer=True)
    return LoadConstraintScopes(KXorInstance.symbolic(n, m, k))
