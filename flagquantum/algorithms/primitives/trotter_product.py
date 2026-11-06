# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 TrotterizedUnitary.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/trotter/trotterized_unitary.py

"""Ordered products of angle-bound unitary terms on shared registers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, TypeAlias

from ...circuit import Circuit
from ...compiler._component_trace.resource_model import ComponentCost, CostCall
from ...compiler.registers.partition import RegisterLayout, unpartition_register
from ...core._composition import inverted_instruction
from ...core.ir import CircuitIR, Instruction
from ...core.parameters import Parameter, ParameterExpression

EvolutionAngle: TypeAlias = int | float | Parameter | ParameterExpression


class AngleBoundTerm(Protocol):
    """A unitary on through ports, with a real angle and a declared logical cost."""

    @property
    def layout(self) -> RegisterLayout: ...

    def emit(
        self, ports: Mapping[str, tuple[int, ...]], angle: EvolutionAngle
    ) -> tuple[Instruction, ...]: ...

    def resource_model(self, angle: EvolutionAngle) -> ComponentCost: ...


@dataclass(frozen=True)
class _BoundEvolution:
    instructions: tuple[Instruction, ...]
    required_qubits: int
    cost: ComponentCost

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits:
            raise ValueError("Evolution binding exceeds circuit width")
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits,
                self.instructions,
                metadata={"batch_size": circuit.bsz},
            ).validate()
        )

    def resource_model(self) -> ComponentCost:
        return self.cost


def _validate_angle(angle: EvolutionAngle) -> None:
    if isinstance(angle, (Parameter, ParameterExpression)):
        return
    if (
        isinstance(angle, bool)
        or not isinstance(angle, (int, float))
        or not isfinite(angle)
    ):
        raise ValueError(
            "Evolution angle must be a finite real scalar or circuit parameter"
        )


def _bind_evolution(
    wires: tuple[int, ...],
    instructions: tuple[Instruction, ...],
    cost: ComponentCost,
    *,
    adjoint: bool,
) -> _BoundEvolution:
    if adjoint:
        instructions = tuple(
            inverted_instruction(gate) for gate in reversed(instructions)
        )
    return _BoundEvolution(instructions, max(wires, default=-1) + 1, cost)


@dataclass(frozen=True)
class TrotterProduct:
    """Apply the supplied term schedule in order on identical named through ports.

    Each paired index and coefficient contributes one call at angle
    `2 * coefficient * timestep`. Pairing stops at the shorter sequence.
    Register tuples are ordered most significant wire first, and all wires
    belong to the caller. Terms may only emit within the declared ports.
    """

    terms: tuple[AngleBoundTerm, ...]
    indices: tuple[int, ...]
    coefficients: tuple[EvolutionAngle, ...]
    timestep: EvolutionAngle

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("A Trotter product requires at least one term")
        if any(term.layout != self.layout for term in self.terms):
            raise ValueError("Trotter terms must have identical register layouts")

    @property
    def layout(self) -> RegisterLayout:
        return self.terms[0].layout

    def _schedule(self) -> tuple[tuple[AngleBoundTerm, EvolutionAngle], ...]:
        calls: list[tuple[AngleBoundTerm, EvolutionAngle]] = []
        for index, coefficient in zip(self.indices, self.coefficients, strict=False):
            _validate_angle(coefficient)
            _validate_angle(self.timestep)
            angle = 2 * coefficient * self.timestep
            _validate_angle(angle)
            calls.append((self.terms[index], angle))
        return tuple(calls)

    def _cost(
        self, schedule: tuple[tuple[AngleBoundTerm, EvolutionAngle], ...]
    ) -> ComponentCost:
        return ComponentCost(
            "TrotterizedUnitary",
            sum(width for _, width in self.layout),
            calls=tuple(
                CostCall(term.resource_model(angle)) for term, angle in schedule
            ),
        )

    def resource_model(self) -> ComponentCost:
        """Count the ordered calls without binding physical wires."""
        return self._cost(self._schedule())

    def make_on(
        self, *, adjoint: bool = False, **ports: tuple[int, ...]
    ) -> _BoundEvolution:
        """Bind the supplied through ports and optionally invert the gate stream."""
        wires = unpartition_register(ports, self.layout)
        schedule = self._schedule()
        instructions = tuple(
            gate for term, angle in schedule for gate in term.emit(ports, angle)
        )
        allowed = set(wires)
        if any(not allowed.issuperset(gate.wires) for gate in instructions):
            raise ValueError("Trotter term emitted outside its declared ports")
        return _bind_evolution(
            wires, instructions, self._cost(schedule), adjoint=adjoint
        )
