"""Validate and capture native bound blocks without numerical execution."""

from dataclasses import dataclass

from ...circuit import Circuit, _CircuitOperation
from ...compiler.registers.bits.bitwise import _validate_wires
from ...core.ir import CircuitIR, Instruction


def _validate_block_work(
    through: tuple[int, ...], work: tuple[int, ...], bits: tuple[int, ...]
) -> None:
    if through or work:
        _validate_wires((*through, *work))
    if len(set(bits)) != len(bits) or any(
        type(bit) is not int or bit < 0 for bit in bits
    ):
        raise ValueError("Block result slots must be distinct nonnegative integers")


def _capture_block(
    operation: _CircuitOperation, required: int, allowed: set[int]
) -> tuple[Instruction, ...]:
    captured = Circuit(required).append(operation).to_ir().validate()
    if any(
        wire not in allowed for gate in captured.instructions for wire in gate.wires
    ):
        raise ValueError("Block child used another child's quantum partition")
    return captured.instructions


@dataclass(frozen=True)
class _BoundBlockOperation:
    instructions: tuple[Instruction, ...]
    required_qubits: int

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits:
            raise ValueError("Block binding exceeds circuit width")
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits,
                self.instructions,
                metadata={"batch_size": circuit.bsz},
            ).validate()
        )
