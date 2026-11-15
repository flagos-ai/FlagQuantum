# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ChebyshevPolynomial.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/chebyshev_polynomial.py

"""Chebyshev transformation of an identity-signal block encoding."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from ...circuit import _CircuitOperation
from ...compiler._component_trace.resource_model import ComponentCost, CostCall
from ...compiler._component_trace.symbolic_math import SymbolicReal
from ...compiler.registers.preparation.prepare_reflection import (
    prepare_reflection_ir,
    prepare_reflection_requirements,
    source_uncontrolled_prepare_reflection_cost,
)
from ...core.ir import Instruction
from ._block_binding import _BoundBlockOperation, _capture_block, _validate_block_work
from ._block_metadata import _BlockMetadata
from .black_box_prepare import BlackBoxPrepare, _identity_prepare


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
        reflection_model: Literal["call_graph", "decomposition"] = "call_graph",
    ) -> ComponentCost:
        """Compose directional costs and choose the source reflection count route."""
        if type(adjoint) is not bool or reflection_model not in (
            "call_graph",
            "decomposition",
        ):
            raise ValueError("Chebyshev requires a boolean direction and a cost route")
        pairs, remainder = divmod(self.order, 2)
        width = self.system_bitsize + self.ancilla_bitsize + self.resource_bitsize
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
                reflection = source_uncontrolled_prepare_reflection_cost(
                    self.ancilla_bitsize, 0, identity, identity
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
        stages = reversed(range(self.order)) if adjoint else range(self.order)
        for index in stages:
            if not adjoint:
                gates.extend(reflection)
            direction = bool(index % 2) ^ adjoint
            child = bind_child(ports, workspace, classical_bits, direction)
            gates.extend(_capture_block(child, required, allowed))
            if adjoint:
                gates.extend(reflection)
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
