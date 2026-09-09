"""Lower an ephemeral selected trace into the existing Core CircuitIR."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ...core.ir import CircuitIR, Instruction, MeasurementNode, ObservableNode
from ...core.parameters import Parameter, bind_parameter_value
from .model import HybridProgram
from .specialize import SpecializedTrace, specialize_program


@dataclass(frozen=True)
class LoweredHybridProgram:
    """A reusable circuit template plus late-bound runtime parameter views."""

    circuit_template: CircuitIR
    bindings: Mapping[str, Any] = field(compare=False, repr=False)
    program_identity: str
    input_signature_identity: str
    selected_structure_identity: str
    cache_event: str
    evicted_cache_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", MappingProxyType(dict(self.bindings)))

    @property
    def cache_key(self) -> str:
        encoded = ":".join(
            (
                self.program_identity,
                self.input_signature_identity,
                self.selected_structure_identity,
            )
        )
        return hashlib.sha256(encoded.encode("ascii")).hexdigest()

    def bind(self) -> CircuitIR:
        """Bind parameter slots while preserving tensor objects and autograd edges."""

        instructions = tuple(
            replace(
                instruction,
                params=bind_parameter_value(instruction.params, self.bindings),
            )
            for instruction in self.circuit_template.instructions
        )
        return replace(self.circuit_template, instructions=instructions)


class CircuitStructureCache:
    """Bounded LRU cache whose hit and eviction behavior is returned to callers."""

    def __init__(self, max_entries: int = 64) -> None:
        if int(max_entries) <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[str, CircuitIR] = OrderedDict()

    def resolve(
        self, key: str, candidate: CircuitIR
    ) -> tuple[CircuitIR, str, str | None]:
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            return cached, "hit", None
        evicted = None
        event = "miss"
        if len(self._entries) >= self.max_entries:
            evicted, _ = self._entries.popitem(last=False)
            event = "miss_evicted"
        self._entries[key] = candidate
        return candidate, event, evicted

    def __len__(self) -> int:
        return len(self._entries)


def _template(
    trace: SpecializedTrace, *, circuit_dtype: str
) -> tuple[CircuitIR, dict[str, Any]]:
    instructions = []
    bindings: dict[str, Any] = {}
    slot_index = 0
    for gate in trace.gates:
        params: dict[str, Any] = {}
        if gate.parameter is not None:
            slot = f"hybrid_p{slot_index}"
            slot_index += 1
            params["theta"] = Parameter(slot)
            bindings[slot] = gate.parameter
        instructions.append(Instruction(gate.name, gate.wires, params=params))
    observables = tuple(
        ObservableNode(pauli, (wire,)) for pauli, wire in trace.observables
    )
    term_count = len(trace.observables)
    measurements = tuple(
        MeasurementNode(
            "expectation_ps",
            (wire,),
            metadata={
                "fq_output_index": 0,
                "fq_output_kind": "expectation",
                "fq_output_name": None,
                "fq_output_term": term_index,
                "fq_output_terms": term_count,
                "fq_coefficient": 1.0,
                "x": (wire,) if pauli == "x" else (),
                "y": (wire,) if pauli == "y" else (),
                "z": (wire,) if pauli == "z" else (),
            },
        )
        for term_index, (pauli, wire) in enumerate(trace.observables)
    )
    referenced_wires = [
        wire for instruction in instructions for wire in instruction.wires
    ] + [wire for observable in observables for wire in observable.wires]
    if not referenced_wires:
        raise ValueError("selected trace does not reference a quantum wire")
    circuit = CircuitIR(
        n_wires=max(referenced_wires) + 1,
        instructions=tuple(instructions),
        observables=observables,
        measurements=measurements,
        dtype=circuit_dtype,
    )
    return circuit, bindings


def lower_trace(
    trace: SpecializedTrace,
    *,
    circuit_dtype: str = "complex64",
    cache: CircuitStructureCache | None = None,
) -> LoweredHybridProgram:
    """Lower a selected trace to a parameterized CircuitIR template."""

    candidate, bindings = _template(trace, circuit_dtype=circuit_dtype)
    structure_identity = candidate.content_hash
    key_material = ":".join(
        (
            trace.program_identity,
            trace.input_signature_identity,
            structure_identity,
        )
    )
    key = hashlib.sha256(key_material.encode("ascii")).hexdigest()
    if cache is None:
        template, event, evicted = candidate, "disabled", None
    else:
        template, event, evicted = cache.resolve(key, candidate)
    return LoweredHybridProgram(
        circuit_template=template,
        bindings=bindings,
        program_identity=trace.program_identity,
        input_signature_identity=trace.input_signature_identity,
        selected_structure_identity=structure_identity,
        cache_event=event,
        evicted_cache_key=evicted,
    )


def specialize_and_lower(
    program: HybridProgram,
    inputs: Sequence[Any],
    *,
    circuit_dtype: str = "complex64",
    max_unrolled_iterations: int = 10_000,
    cache: CircuitStructureCache | None = None,
) -> LoweredHybridProgram:
    """Specialize one path and lower it without numerical execution."""

    trace = specialize_program(
        program, inputs, max_unrolled_iterations=max_unrolled_iterations
    )
    return lower_trace(trace, circuit_dtype=circuit_dtype, cache=cache)
