# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GateCounts and sequential width accounting.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting

"""Evaluate declared component costs independently of emitted instructions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GateCost:
    """Concrete logical cost categories under a declared gate model."""

    t: int = 0
    toffoli: int = 0
    cswap: int = 0
    and_bloq: int = 0
    clifford: int = 0
    rotation: int = 0
    measurement: int = 0

    def plus(self, other: "GateCost", repetitions: int) -> "GateCost":
        return GateCost(
            t=self.t + repetitions * other.t,
            toffoli=self.toffoli + repetitions * other.toffoli,
            cswap=self.cswap + repetitions * other.cswap,
            and_bloq=self.and_bloq + repetitions * other.and_bloq,
            clifford=self.clifford + repetitions * other.clifford,
            rotation=self.rotation + repetitions * other.rotation,
            measurement=self.measurement + repetitions * other.measurement,
        )

    def total_t_count(
        self,
        *,
        ts_per_toffoli: int = 4,
        ts_per_cswap: int = 4,
        ts_per_and: int = 4,
        ts_per_rotation: int = 11,
    ) -> int:
        """Convert logical counts to model T equivalents at the specified rates."""
        return (
            self.t
            + ts_per_toffoli * self.toffoli
            + ts_per_cswap * self.cswap
            + ts_per_and * self.and_bloq
            + ts_per_rotation * self.rotation
        )


@dataclass(frozen=True)
class ComponentCost:
    """Model for one component with exclusive local gates and sequential children.

    The local qubit count covers ports and any local peak between calls.
    Child edges explicitly state how many other qubits are live during the call.
    """

    name: str
    qubits: int
    gates: GateCost = GateCost()
    calls: tuple["CostCall", ...] = ()


@dataclass(frozen=True)
class CostCall:
    """Repeated calls share the same sequential width contribution."""

    component: ComponentCost
    repetitions: int = 1
    bystander_qubits: int = 0


@dataclass(frozen=True)
class ModeledResources:
    gates: GateCost
    peak_logical_qubits: int


def estimate_component_resources(component: ComponentCost) -> ModeledResources:
    """Sum logical gates and take the maximum scheduled width along the call tree."""
    cache: dict[int, ModeledResources] = {}

    def visit(node: ComponentCost) -> ModeledResources:
        key = id(node)
        if key in cache:
            return cache[key]
        if type(node.qubits) is not int or node.qubits < 0:
            raise ValueError("Component width must be a nonnegative integer")
        gates, peak = node.gates, node.qubits
        for call in node.calls:
            if (
                type(call.repetitions) is not int
                or call.repetitions < 0
                or type(call.bystander_qubits) is not int
                or call.bystander_qubits < 0
            ):
                raise ValueError(
                    "Call multiplicity and bystander width must be nonnegative integers"
                )
            if call.repetitions:
                child = visit(call.component)
                gates = gates.plus(child.gates, call.repetitions)
                peak = max(peak, child.peak_logical_qubits + call.bystander_qubits)
        estimate = ModeledResources(gates, peak)
        cache[key] = estimate
        return estimate

    return visit(component)
