"""Assign private classical scratch storage after fresh register-SSA binding."""

from collections import deque
from dataclasses import replace
from heapq import heappop, heappush

from ....core.classical_control import (
    instruction_condition_clauses,
    validate_classical_control,
)
from ....core.classical_dataflow import parse_classical_xor
from ..register_binding import BoundRegisterProgram


def bind_classical_scratch(
    bound: BoundRegisterProgram, slots: tuple[int, ...]
) -> BoundRegisterProgram:
    """Reuse explicit scratch slots without changing quantum operations or leases.

    This is an opt-in storage binding, not a transparent optimizer for observable
    classical results. Exported classical outputs are refused. Every logical bit
    retains one conservative interval across redefinitions and conditional writes.
    A slot is reusable only after its last read/write. FIFO allocation respects
    supplied order and leaves each physical slot's latest assigned outcome.
    Insufficient storage or malformed bindings fail before returning a result.
    """
    if (
        not isinstance(slots, tuple)
        or any(type(slot) is not int or slot < 0 for slot in slots)
        or len(set(slots)) != len(slots)
    ):
        raise ValueError(
            "Classical scratch requires distinct nonnegative integer slots"
        )
    if any(bound.classical_outputs):
        raise ValueError("Classical scratch binding cannot rewrite exported results")
    circuit = bound.circuit
    validate_classical_control(circuit.instructions, n_qubits=circuit.n_wires)
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for position, gate in enumerate(circuit.instructions):
        clauses = instruction_condition_clauses(gate)
        reads = {bit for clause in clauses for bit, _ in clause}
        if gate.name == "classical_xor":
            assignment = parse_classical_xor(gate)
            reads.update(assignment.sources)
            if clauses:
                reads.add(assignment.target)
        if "classical_bit" in gate.metadata:
            target = gate.metadata["classical_bit"]
            first.setdefault(target, position)
            last[target] = position
        for bit in reads:
            last[bit] = position
    available = deque(slots)
    active: list[tuple[int, int]] = []
    mapping: dict[int, int] = {}
    for bit, position in sorted(first.items(), key=lambda item: item[1]):
        while active and active[0][0] < position:
            _, old = heappop(active)
            available.append(mapping[old])
        if not available:
            raise ValueError("Insufficient classical scratch storage for live values")
        mapping[bit] = available.popleft()
        heappush(active, (last[bit], bit))
    rewritten = []
    for gate in circuit.instructions:
        metadata = dict(gate.metadata)
        if "classical_bit" in metadata:
            metadata["classical_bit"] = mapping[metadata["classical_bit"]]
        clauses = instruction_condition_clauses(gate)
        if clauses:
            mapped = tuple(
                sorted(
                    (
                        tuple(sorted((mapping[bit], value) for bit, value in clause))
                        for clause in clauses
                    ),
                    key=lambda clause: (len(clause), clause),
                )
            )
            if "conditions" in metadata:
                metadata["conditions"] = mapped[0]
            else:
                metadata["condition_clauses"] = mapped
        params = gate.params
        if gate.name == "classical_xor":
            params = {
                **params,
                "sources": tuple(
                    mapping[bit] for bit in parse_classical_xor(gate).sources
                ),
            }
        rewritten.append(replace(gate, params=params, metadata=metadata))
    validate_classical_control(rewritten, n_qubits=circuit.n_wires)
    return replace(bound, circuit=replace(circuit, instructions=tuple(rewritten)))
