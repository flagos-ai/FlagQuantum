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
from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler.registers.layout.partition import RegisterLayout, unpartition_register
from ...core.ir import Instruction
from ...core.ir.program.signatures import Signature
from .trotter_product import (
    EvolutionAngle,
    _bind_evolution,
    _BoundEvolution,
    _validate_angle,
)


def _rotation_call(
    opcode: Literal["rx", "rz"], angle: EvolutionAngle, precision: float = 1e-11
) -> RotationCall:
    _validate_angle(angle)
    value = angle if isinstance(angle, (int, float)) else None
    return RotationCall(opcode, value, precision)


def _ising_cost(name: str, sites: int, rotation_call: RotationCall) -> ComponentCost:
    opcode = rotation_call.opcode
    family = classify_resource_gate(rotation_call)
    rotation = ComponentCost(
        "Rx" if opcode == "rx" else "Rz",
        1,
        GateCost(
            t=int(family == "t"),
            clifford=int(family == "clifford"),
            rotation=int(family == "rotation"),
        ),
    )
    calls: tuple[CostCall, ...] = (CostCall(rotation, sites, sites - 1),)
    if opcode == "rz":
        cnot = ComponentCost("CNOT", 2, GateCost(clifford=1))
        calls = (CostCall(cnot, 2 * sites, sites - 2), *calls)
    return ComponentCost(name, sites, calls=calls)


@dataclass(frozen=True)
class IsingXUnitary:
    """Apply an X rotation to each site in register order."""

    nsites: int
    eps: float = 1e-10

    def __post_init__(self) -> None:
        if type(self.nsites) is not int or self.nsites < 1:
            raise ValueError("Ising X requires at least one site")

    @property
    def signature(self) -> Signature:
        """Return the declared through register for the ordered sites."""
        return Signature.build(system=self.nsites)

    @property
    def layout(self) -> RegisterLayout:
        return (("system", self.nsites),)

    def emit(
        self, ports: Mapping[str, tuple[int, ...]], angle: EvolutionAngle
    ) -> tuple[Instruction, ...]:
        _validate_angle(angle)
        wires = unpartition_register(ports, self.layout)
        return tuple(Instruction("rx", (wire,), {"theta": angle}) for wire in wires)

    def source_rotation(self, angle: EvolutionAngle) -> RotationCall:
        """The source constructs Rx with its gate default precision."""
        return _rotation_call("rx", angle)

    def resource_model(self, angle: EvolutionAngle) -> ComponentCost:
        return _ising_cost("IsingXUnitary", self.nsites, self.source_rotation(angle))

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
    eps: float = 1e-10

    def __post_init__(self) -> None:
        if type(self.nsites) is not int or self.nsites < 2:
            raise ValueError("Periodic Ising ZZ requires at least two sites")

    @property
    def signature(self) -> Signature:
        """Return the declared through register for the ordered sites."""
        return Signature.build(system=self.nsites)

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

    def source_rotation(self, angle: EvolutionAngle) -> RotationCall:
        """The source forwards its declared precision to each Rz."""
        return _rotation_call("rz", angle, self.eps)

    def resource_model(self, angle: EvolutionAngle) -> ComponentCost:
        return _ising_cost("IsingZZUnitary", self.nsites, self.source_rotation(angle))

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
