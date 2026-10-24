"""Gate-bound Pauli locations shared by exact DEM construction and execution."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ...core.ir import CircuitIR
from ...errors import CapabilityError
from ...noise import KrausChannel, NoiseModel, bit_flip_channel, phase_flip_channel
from ..circuits.memory import MemoryCircuit
from ..execution.signatures import (
    _gate_product_signatures,
    _lowered_program,
    _pauli_channel_terms,
)
from .construction import _GATE_BOUND_FAMILIES, _UNPLACEABLE_RULE_GATES, _pauli_fault
from .matrices import Entry
from .model import DemError


@dataclass(frozen=True)
class _CircuitNoiseLocation:
    position: int
    qubits: tuple[int, ...]
    channel: KrausChannel
    terms: tuple[tuple[str, float], ...]
    independent: bool


def _circuit_noise_locations(
    noise: NoiseModel, program: CircuitIR
) -> tuple[_CircuitNoiseLocation, ...]:
    for rule in noise.rules:
        unplaceable = sorted(set(rule.gate_names) & _UNPLACEABLE_RULE_GATES)
        if unplaceable:
            raise CapabilityError(
                f"a gate-bound noise rule cannot name {unplaceable[0]!r}: "
                "readout and preparation faults use PhenomenologicalNoise"
            )
    locations: list[_CircuitNoiseLocation] = []
    for position, instruction in enumerate(program.instructions):
        for channel, qubits in noise.channels_for(instruction):
            independent = channel.name in _GATE_BOUND_FAMILIES
            terms: tuple[tuple[str, float], ...]
            if independent:
                family, probability = _pauli_fault(channel)
                if len(qubits) != 1:
                    raise CapabilityError(
                        f"the {channel.name!r} rule acts on {len(qubits)} qubits; a single-qubit Pauli fault requires one"
                    )
                # Reconstruct exactly as the previous circuit sampler did; its
                # second floating-point conversion and seeded stream are pinned.
                channel = (
                    bit_flip_channel if family == "data" else phase_flip_channel
                )(probability)
                terms = (
                    (("X" if family == "data" else "Z", probability),)
                    if probability
                    else ()
                )
            else:
                try:
                    terms = _pauli_channel_terms(channel)
                except CapabilityError as error:
                    raise CapabilityError(
                        f"gate-bound channels include bit_flip and phase_flip and "
                        f"one-/two-qubit Pauli mixtures: {error}"
                    ) from error
                if len(qubits) != channel.n_qubits or len(set(qubits)) != len(qubits):
                    raise ValueError(
                        "channel targets must be distinct and match its arity"
                    )
            if position == len(program.instructions) - 1:
                raise CapabilityError(
                    f"the {channel.name!r} rule on {instruction.name!r} matches "
                    "the program's last instruction, so the fault would follow "
                    "the program's end rather than a gate in it"
                )
            if terms:
                locations.append(
                    _CircuitNoiseLocation(position, qubits, channel, terms, independent)
                )
    return tuple(locations)


def _circuit_noise_entries(
    circuit: MemoryCircuit, noise: NoiseModel
) -> tuple[tuple[Entry, ...], tuple[DemError, ...]]:
    program = _lowered_program(circuit)
    locations = _circuit_noise_locations(noise, program)
    faults = tuple(
        (location.position, location.qubits, label)
        for location in locations
        for label, _ in location.terms
    )
    signatures = iter(_gate_product_signatures(circuit, program, faults))
    independent: list[Entry] = []
    grouped: list[DemError] = []
    for error_id, location in enumerate(locations):
        outcomes: dict[tuple[tuple[int, ...], tuple[int, ...]], list[float]] = {}
        for _, probability in location.terms:
            signature = next(signatures)
            if not signature[0] and not signature[1]:
                continue
            if location.independent:
                independent.append((probability, *signature))
            else:
                outcomes.setdefault(signature, []).append(probability)
        grouped.extend(
            DemError(math.fsum(weights), *signature, error_id=error_id)
            for signature, weights in outcomes.items()
        )
    return tuple(independent), tuple(grouped)
