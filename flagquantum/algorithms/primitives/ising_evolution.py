# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the periodic Ising unitaries in Qualtran v0.7.0.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/trotter/ising/unitaries.py

"""Transverse-field and periodic nearest-neighbor Ising evolution terms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from ...compiler._component_trace.resource_classification import classify_resource_gate
from ...compiler._component_trace.resource_generalization import RotationCall
from ...compiler._component_trace.resource_model import ComponentCost, GateCost
from ...compiler.registers.layout.partition import RegisterLayout, unpartition_register
from ...core.ir import Instruction
from .trotter_product import (
    EvolutionAngle,
    _bind_evolution,
    _BoundEvolution,
    _validate_angle,
)


def _ising_cost(
    name: str, sites: int, opcode: Literal["rx", "rz"], angle: EvolutionAngle
) -> ComponentCost:
    value = angle if isinstance(angle, (int, float)) else None
    family = classify_resource_gate(RotationCall(opcode, value))
    cliffords = 2 * sites if opcode == "rz" else 0
    return ComponentCost(
        name,
        sites,
        GateCost(
            t=sites if family == "t" else 0,
            clifford=cliffords + (sites if family == "clifford" else 0),
            rotation=sites if family == "rotation" else 0,
        ),
    )


@dataclass(frozen=True)
class IsingXUnitary:
    """Apply an X rotation to each site in register order."""

    nsites: int

    def __post_init__(self) -> None:
        if type(self.nsites) is not int or self.nsites < 1:
            raise ValueError("Ising X requires at least one site")

    @property
    def layout(self) -> RegisterLayout:
        return (("system", self.nsites),)

    def emit(
        self, ports: Mapping[str, tuple[int, ...]], angle: EvolutionAngle
    ) -> tuple[Instruction, ...]:
        _validate_angle(angle)
        wires = unpartition_register(ports, self.layout)
        return tuple(Instruction("rx", (wire,), {"theta": angle}) for wire in wires)

    def resource_model(self, angle: EvolutionAngle) -> ComponentCost:
        _validate_angle(angle)
        return _ising_cost("IsingXUnitary", self.nsites, "rx", angle)

    def make_on(
        self, *, system: tuple[int, ...], angle: EvolutionAngle, adjoint: bool = False
    ) -> _BoundEvolution:
        """Bind the ordered system wires to an appendable operation."""
        return _bind_evolution(
            system,
            self.emit({"system": system}, angle),
            self.resource_model(angle),
            adjoint=adjoint,
        )


@dataclass(frozen=True)
class IsingZZUnitary:
    """Apply CX, Rz, CX on each directed periodic edge in site order."""

    nsites: int

    def __post_init__(self) -> None:
        if type(self.nsites) is not int or self.nsites < 2:
            raise ValueError("Periodic Ising ZZ requires at least two sites")

    @property
    def layout(self) -> RegisterLayout:
        return (("system", self.nsites),)

    def emit(
        self, ports: Mapping[str, tuple[int, ...]], angle: EvolutionAngle
    ) -> tuple[Instruction, ...]:
        _validate_angle(angle)
        wires = unpartition_register(ports, self.layout)
        gates: list[Instruction] = []
        for site, control in enumerate(wires):
            target = wires[(site + 1) % self.nsites]
            cx = Instruction("cx", (control, target))
            gates.extend((cx, Instruction("rz", (target,), {"theta": angle}), cx))
        return tuple(gates)

    def resource_model(self, angle: EvolutionAngle) -> ComponentCost:
        _validate_angle(angle)
        return _ising_cost("IsingZZUnitary", self.nsites, "rz", angle)

    def make_on(
        self, *, system: tuple[int, ...], angle: EvolutionAngle, adjoint: bool = False
    ) -> _BoundEvolution:
        """Bind the ordered system wires to an appendable operation."""
        return _bind_evolution(
            system,
            self.emit({"system": system}, angle),
            self.resource_model(angle),
            adjoint=adjoint,
        )
