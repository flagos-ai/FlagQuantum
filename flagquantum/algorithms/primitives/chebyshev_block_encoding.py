# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ChebyshevPolynomial.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/chebyshev_polynomial.py

"""Chebyshev transformation of an identity-signal block encoding."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Literal, cast

from ...circuit import _CircuitOperation
from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler._component_trace.symbolic_math import SymbolicInt, SymbolicReal, smax
from ...compiler.registers.preparation.chebyshev_component import (
    chebyshev_call_sequence,
)
from ...compiler.registers.preparation.prepare_reflection import (
    prepare_reflection_ir,
    prepare_reflection_requirements,
    source_prepare_reflection_cost,
    source_symbolic_zero_reflection_resource_formula,
)
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from ...core.ir import Instruction
from ...core.ir.program.signatures import Signature
from ._block_binding import _BoundBlockOperation, _capture_block, _validate_block_work
from ._block_metadata import _BlockMetadata, _packed_block_signature
from .black_box_prepare import BlackBoxPrepare, _identity_prepare


def source_symbolic_chebyshev_resource_formula(
    system: SymbolicInt,
    ancilla: SymbolicInt,
    resource: SymbolicInt,
    order: int,
    forward_child: SymbolicResourceFormula,
    adjoint_child: SymbolicResourceFormula,
    *,
    adjoint: bool = False,
    legacy_shims: bool = False,
) -> SymbolicResourceFormula:
    """Compose the locked fallback when symbolic signal width blocks execution.

    Keep the source's zero-count callee declarations in the peak query. Odd
    orders retain the source direct-graph reflection undercount; no executable
    gate stream or concrete allocation is inferred from this model.
    """
    if type(order) is not int or order < 0 or type(adjoint) is not bool:
        raise ValueError("Chebyshev needs a nonnegative order and boolean direction")
    if type(ancilla) is int:
        raise ValueError("Symbolic fallback requires an unresolved signal width")
    signature = _packed_block_signature(system, ancilla, resource)
    reflection = source_symbolic_zero_reflection_resource_formula(
        ancilla, legacy_shims=legacy_shims
    )
    pairs, tail = divmod(order, 2)
    counts = (pairs, pairs + tail) if adjoint else (pairs + tail, pairs)
    calls = (
        (forward_child, counts[0]),
        (adjoint_child, counts[1]),
        (reflection, 2 * pairs),
    )
    return SymbolicResourceFormula(
        **{
            field.name: sum(
                getattr(child, field.name) * count for child, count in calls
            )
            for field in fields(GateCost)
        },
        logical_qubits=cast(
            SymbolicInt,
            smax(signature.n_qubits(), *(child.logical_qubits for child, _ in calls)),
        ),
    )


@dataclass(frozen=True)
class ChebyshevBlockEncoding:
    """Encode the order-n Chebyshev polynomial of the normalized child block."""

    block_encoding: _BlockMetadata
    order: int

    def __post_init__(self) -> None:
        if type(self.order) is not int or self.order < 0:
            raise ValueError("Chebyshev order must be a nonnegative integer")
        if self.block_encoding.signal_state.prepare is not _identity_prepare:
            raise ValueError("Chebyshev requires identity child signal preparation")

    @property
    def system_bitsize(self) -> int:
        return self.block_encoding.system_bitsize

    @property
    def ancilla_bitsize(self) -> int:
        return self.block_encoding.ancilla_bitsize

    @property
    def resource_bitsize(self) -> int:
        return self.block_encoding.resource_bitsize

    @property
    def signature(self) -> Signature:
        """Declare the source-ordered packed through ports."""
        return _packed_block_signature(
            self.system_bitsize, self.ancilla_bitsize, self.resource_bitsize
        )

    @property
    def alpha(self) -> int:
        return 1

    @property
    def epsilon(self) -> SymbolicReal:
        return self.block_encoding.epsilon * self.order

    @property
    def signal_state(self) -> BlackBoxPrepare:
        return BlackBoxPrepare(
            (("ancilla", self.ancilla_bitsize),), (), _identity_prepare
        )

    def resource_model(
        self,
        forward_child: ComponentCost,
        adjoint_child: ComponentCost,
        *,
        adjoint: bool = False,
        controlled: bool = False,
        reflection_model: Literal["call_graph", "decomposition"] = "call_graph",
    ) -> ComponentCost:
        """Compose directional costs and choose the source reflection count route."""
        if (
            type(adjoint) is not bool
            or type(controlled) is not bool
            or reflection_model not in ("call_graph", "decomposition")
        ):
            raise ValueError("Chebyshev requires boolean flags and a cost route")
        pairs, remainder = divmod(self.order, 2)
        width = (
            self.system_bitsize
            + self.ancilla_bitsize
            + self.resource_bitsize
            + int(controlled)
        )
        forward_count, adjoint_count = pairs + remainder, pairs
        if adjoint:
            forward_count, adjoint_count = adjoint_count, forward_count
        calls = []
        for child, count in (
            (forward_child, forward_count),
            (adjoint_child, adjoint_count),
        ):
            if count:
                if child.qubits < width:
                    raise ValueError(
                        "Chebyshev child cost omits declared through ports"
                    )
                calls.append(CostCall(child, count))
        if self.ancilla_bitsize:
            count = 2 * pairs if reflection_model == "call_graph" else self.order
            if count:
                identity = ComponentCost("PrepareIdentity", self.ancilla_bitsize)
                reflection = source_prepare_reflection_cost(
                    self.ancilla_bitsize, 0, identity, identity, controlled=controlled
                )
                calls.append(
                    CostCall(
                        reflection,
                        count,
                        bystander_qubits=self.system_bitsize + self.resource_bitsize,
                    )
                )
        return ComponentCost("Chebyshev", width, calls=tuple(calls))

    def make_on(
        self,
        bind_child: Callable[
            [Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], bool],
            _CircuitOperation,
        ],
        *,
        system: tuple[int, ...],
        ancilla: tuple[int, ...] = (),
        resource: tuple[int, ...] = (),
        control: int | None = None,
        workspace: tuple[int, ...] = (),
        classical_bits: tuple[int, ...] = (),
        reflection_workspace: tuple[int, ...] = (),
        reflection_classical_bits: tuple[int, ...] = (),
        adjoint: bool = False,
    ) -> _BoundBlockOperation:
        """Bind the alternating calls. The child factory binds any control."""
        if type(adjoint) is not bool:
            raise ValueError("Chebyshev direction must be boolean")
        if (len(system), len(ancilla), len(resource)) != (
            self.system_bitsize,
            self.ancilla_bitsize,
            self.resource_bitsize,
        ):
            raise ValueError("Chebyshev ports must match the declared widths")
        through = (*system, *ancilla, *resource)
        if control is not None:
            through = (*through, control)
        _validate_block_work(through, workspace, classical_bits)
        _validate_block_work(through, reflection_workspace, reflection_classical_bits)
        all_wires = (*through, *workspace, *reflection_workspace)
        if not all_wires:
            raise ValueError(
                "Chebyshev numerical binding needs an existing quantum wire"
            )
        required = max(all_wires) + 1
        reflection = self._bound_reflection(
            ancilla, reflection_workspace, reflection_classical_bits, control
        )
        gates: list[Instruction] = []
        ports = {"system": system, "ancilla": ancilla, "resource": resource}
        allowed = set((*through, *workspace))
        for stage in chebyshev_call_sequence(self.order, adjoint=adjoint):
            if stage == "reflection":
                gates.extend(reflection)
            else:
                child = bind_child(ports, workspace, classical_bits, stage == "inverse")
                gates.extend(_capture_block(child, required, allowed))
        return _BoundBlockOperation(tuple(gates), required)

    def _bound_reflection(
        self,
        ancilla: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        control: int | None,
    ) -> tuple[Instruction, ...]:
        if self.order and ancilla:
            return prepare_reflection_ir(
                ancilla, workspace, classical_bits, global_phase=-1, control=control
            )
        if workspace or classical_bits:
            raise ValueError("Chebyshev reflection has no work when it emits no calls")
        return ()

    def reflection_requirements(self, *, controlled: bool = False) -> tuple[int, int]:
        """Return work and reusable result slots for the sequential reflections."""
        if self.order and self.ancilla_bitsize:
            return prepare_reflection_requirements(
                self.ancilla_bitsize, controlled=controlled
            )
        return 0, 0
