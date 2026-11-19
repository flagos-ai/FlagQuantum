"""Generate operator-exact named circuits, then select by measured CX count."""

import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

import torch

import flagquantum as fq
from flagquantum.compiler import estimate_resources
from flagquantum.compiler.resource_estimation import ResourceEstimate

from .model import LAYERS, WIDTH, cost_terms


class Schedule(Enum):
    SEPARATE = "separate parity"
    SORTED = "shared parity, sorted"
    NEAREST = "shared parity, nearest next"


@dataclass(frozen=True)
class Candidate:
    schedule: Schedule
    circuit: fq.Circuit
    resources: ResourceEstimate

    @property
    def cx(self) -> int:
        return self.resources.operation_counts.get("cx", 0)


def groups() -> dict[int, list[tuple[frozenset[int], int]]]:
    """Group all generated terms by a fixed physical parity target."""
    output: dict[int, list[tuple[frozenset[int], int]]] = defaultdict(list)
    for support, coefficient in cost_terms():
        if support:
            wires = tuple(WIDTH - 1 - bit for bit in support)
            output[wires[-1]].append((frozenset(wires[:-1]), coefficient))
    return dict(output)


def ordered(
    terms: list[tuple[frozenset[int], int]],
    schedule: Schedule,
) -> tuple[tuple[frozenset[int], int], ...]:
    if schedule is not Schedule.NEAREST:
        return tuple(sorted(terms, key=lambda row: tuple(sorted(row[0]))))
    pending = terms.copy()
    current: frozenset[int] = frozenset()
    order = []
    while pending:
        choice = min(
            pending, key=lambda row: (len(current ^ row[0]), tuple(sorted(row[0])))
        )
        pending.remove(choice)
        order.append(choice)
        current = choice[0]
    return tuple(order)


def encode(schedule: Schedule) -> Candidate:
    """Retain all nonzero terms, including the original identity phase."""
    circuit = fq.Circuit(WIDTH, dtype=torch.complex128)
    for wire in range(WIDTH):
        circuit.u3(wire, math.pi / 2, 0.0, math.pi)
    all_zero_cost = sum(coefficient for _, coefficient in cost_terms())
    parities = groups()
    for layer in range(LAYERS):
        gamma = fq.Parameter(f"gamma_{layer}")
        beta = fq.Parameter(f"beta_{layer}")
        for wire, terms in sorted(parities.items()):
            current: frozenset[int] = frozenset()
            for target, coefficient in ordered(terms, schedule):
                for control in sorted(current ^ target):
                    circuit.cx(control, wire)
                circuit.u3(wire, 0.0, 0.0, (2 * coefficient) * gamma)
                current = target
                if schedule is Schedule.SEPARATE:
                    for control in sorted(current):
                        circuit.cx(control, wire)
                    current = frozenset()
            for control in sorted(current):
                circuit.cx(control, wire)

        # U3(0,0,2c*gamma) carries exp(+i*c*gamma). The two phases and
        # two X operations below give the same correction on both basis states.
        correction = (-all_zero_cost) * gamma
        circuit.u3(0, 0.0, 0.0, correction).u3(0, math.pi, 0.0, math.pi)
        circuit.u3(0, 0.0, 0.0, correction).u3(0, math.pi, 0.0, math.pi)
        for wire in range(WIDTH):
            circuit.u3(wire, beta, -math.pi / 2, math.pi / 2)
    resources = estimate_resources(circuit)
    if resources.n_qubits != WIDTH or set(resources.operation_counts) != {"cx", "u3"}:
        raise AssertionError("Candidate escaped the 13-wire CX/U3 basis")
    return Candidate(schedule, circuit, resources)


def choose() -> tuple[tuple[Candidate, ...], Candidate]:
    candidates = tuple(encode(schedule) for schedule in Schedule)
    selected = min(
        candidates,
        key=lambda item: (item.cx, item.resources.depth, item.schedule.value),
    )
    return candidates, selected


def bind(candidate: Candidate, parameters: tuple[float, ...]) -> fq.Circuit:
    if len(parameters) != 2 * LAYERS or any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in parameters
    ):
        raise ValueError("Three layers require six finite real angles")
    values = {
        f"{name}_{layer}": parameters[offset + layer]
        for name, offset in (("gamma", 0), ("beta", LAYERS))
        for layer in range(LAYERS)
    }
    return candidate.circuit.bind_parameters(values)
