"""Exact Pauli-fault responses of deterministic Clifford measurement parities.

Propagate each requested parity backwards through the measurement channel, not
a sampled trajectory. Integer Pauli masks replace amplitudes; the existing
engine's local conjugation rules remain the authority for every Clifford gate.
"""

from __future__ import annotations

from functools import lru_cache
from numbers import Integral

from ...core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ...core.operator_schema import canonical_opcode, get_operator_schema
from ...errors import CapabilityError
from ...noise import KrausChannel
from .engine import (
    _ADJOINT_GATES,
    _GATE_NAMES,
    _conjugate_step,
    _declares_channel,
    _frame_weights,
)


@lru_cache(maxsize=256)
def _local_image(gate: str, letters: str) -> tuple[str, int]:
    return _conjugate_step(gate, tuple(range(len(letters))), letters, 0)


def _reverse_gate(
    gate: str, wires: tuple[int, ...], x: int, z: int, phase: int
) -> tuple[int, int, int]:
    letters = "".join("IXZY"[((x >> q) & 1) + 2 * ((z >> q) & 1)] for q in wires)
    image, power = _local_image(_ADJOINT_GATES.get(gate, gate), letters)
    for wire, letter in zip(wires, image, strict=True):
        bit = 1 << wire
        x = (x & ~bit) | (bit if letter in "XY" else 0)
        z = (z & ~bit) | (bit if letter in "ZY" else 0)
    return x, z, (phase + power) % 4


def _operations(program: CircuitIR) -> tuple[tuple[str, tuple[int, ...], int], ...]:
    operations: list[tuple[str, tuple[int, ...], int]] = []
    measurement = 0
    for index, instruction in enumerate(program.instructions):
        opcode = canonical_opcode(instruction.name)
        schema = get_operator_schema(opcode)
        if (
            _declares_channel(instruction, schema)
            or instruction.matrix is not None
            or "conditions" in instruction.metadata
            or "condition_clauses" in instruction.metadata
        ):
            raise CapabilityError(
                f"instruction {index} {opcode!r}: Clifford fault signatures "
                "require unconditional gates without channels or custom matrices"
            )
        gate = _GATE_NAMES.get(opcode)
        if gate is not None:
            operations.append((gate, instruction.wires, -1))
        elif opcode in ("measure", "reset") and len(instruction.wires) == 1:
            operations.append((opcode, instruction.wires, measurement))
            measurement += opcode == "measure"
        else:
            raise CapabilityError(
                f"instruction {index} {opcode!r} is not a supported Clifford "
                "gate, single-qubit measurement or reset"
            )
    return tuple(operations)


def _check_index(index: int, size: int, label: str) -> None:
    if isinstance(index, bool) or not isinstance(index, Integral):
        raise TypeError(f"{label} index must be an integer")
    if not 0 <= index < size:
        raise ValueError(f"{label} index {index} is outside 0..{size - 1}")


def _parity_mask(indices: tuple[int, ...], size: int, label: str) -> int:
    mask = 0
    for index in indices:
        _check_index(index, size, label)
        mask ^= 1 << int(index)
    return mask


def _clifford_fault_signatures(
    program: CircuitIR,
    *,
    parities: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    faults: tuple[tuple[int, int, str], ...],
) -> tuple[int, ...]:
    """Return a parity bitmask per post-instruction Pauli fault.

    A parity names recorded measurements by occurrence, followed by terminal Z
    readout qubits. A fault names an instruction position, a qubit and X/Y/Z.
    Initial states and resets are |0>. Baseline parity signs are retained, so a
    deterministic odd parity still reads odd when a fault commutes with it.
    No QEC layouts, optional engine, random sampling or amplitude store is used.

    Storage is proportional to instructions, faults and output bitmasks; work
    is one reverse circuit walk per parity plus its fault responses. A parity
    that is not deterministic is refused exactly, including randomness erased
    by a later reset or cancelled by comparing two recorded measurements.
    """
    for _, _, pauli in faults:
        if pauli not in ("X", "Y", "Z"):
            raise ValueError(f"fault Pauli must be X, Y or Z, got {pauli!r}")
    return _clifford_product_signatures(
        program,
        parities=parities,
        faults=tuple((position, (qubit,), pauli) for position, qubit, pauli in faults),
    )


def _pauli_channel_terms(channel: KrausChannel) -> tuple[tuple[str, float], ...]:
    """Classify one-/two-qubit channels using the engine's numerical authority."""
    if channel.n_qubits not in (1, 2):
        raise CapabilityError(
            "QEC circuit noise supports one- or two-qubit Pauli channels"
        )
    qubits = tuple(range(channel.n_qubits))
    labels, weights = _frame_weights(
        Instruction(channel.name, qubits, matrix=channel), 0, qubits
    )
    return tuple(
        (label, weight) for label, weight in zip(labels, weights, strict=True) if weight
    )


def _clifford_product_signatures(
    program: CircuitIR,
    *,
    parities: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    faults: tuple[tuple[int, tuple[int, ...], str], ...],
) -> tuple[int, ...]:
    """Propagate simultaneous Pauli products, preserving deterministic baselines."""
    ir = ensure_circuit_ir(program)
    if ir.measurements:
        raise CapabilityError(
            "Clifford fault signatures do not read output request nodes"
        )
    operations = _operations(ir)
    count = sum(gate == "measure" for gate, _, _ in operations)
    by_position: dict[int, list[tuple[int, int, int]]] = {}
    for index, (position, qubits, letters) in enumerate(faults):
        _check_index(position, len(operations), "fault instruction")
        if not qubits or len(qubits) != len(letters) or len(set(qubits)) != len(qubits):
            raise ValueError("a Pauli product needs one letter per distinct qubit")
        fault_x = fault_z = 0
        for qubit, letter in zip(qubits, letters, strict=True):
            _check_index(qubit, ir.n_wires, "fault qubit")
            if letter not in "IXYZ":
                raise ValueError(f"fault Pauli must be I, X, Y or Z, got {letter!r}")
            if letter in "XY":
                fault_x |= 1 << int(qubit)
            if letter in "ZY":
                fault_z |= 1 << int(qubit)
        by_position.setdefault(position, []).append((index, fault_x, fault_z))
    signatures = [0] * len(faults)
    for parity_index, (recorded, terminal) in enumerate(parities):
        selected = _parity_mask(recorded, count, "measurement")
        x, z, phase = 0, _parity_mask(terminal, ir.n_wires, "terminal wire"), 0
        parity_bit = 1 << parity_index
        for position in range(len(operations) - 1, -1, -1):
            for index, fault_x, fault_z in by_position.get(position, ()):
                anticommutes = (
                    (x & fault_z).bit_count() ^ (z & fault_x).bit_count()
                ) & 1
                if anticommutes:
                    signatures[index] |= parity_bit
            gate, wires, measurement = operations[position]
            if gate in ("measure", "reset"):
                bit = 1 << wires[0]
                if x & bit:
                    raise ValueError(
                        f"parity {parity_index} is not deterministic across "
                        "trajectories in the Clifford measurement circuit"
                    )
                if gate == "reset":
                    z &= ~bit
                elif (selected >> measurement) & 1:
                    z ^= bit
            else:
                x, z, phase = _reverse_gate(gate, wires, x, z, phase)
        if x:
            raise ValueError(
                f"parity {parity_index} is not deterministic across trajectories "
                "in the Clifford initial state"
            )
        assert phase % 2 == 0
        if phase == 2:
            for index in range(len(signatures)):
                signatures[index] ^= parity_bit
    return tuple(signatures)
