"""Bind preserved physical circuit assets through the existing owned IR."""

from pathlib import Path

from flagquantum.compiler.artifacts.reversible_phase import (
    ReversiblePhaseOperation,
    ReversiblePhaseProgram,
)
from flagquantum.compiler.formats.kmx_circuit import KmxCircuit, read_kmx_circuit
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import Instruction

ASSETS = Path(__file__).with_name("data") / "circuit_assets"


def asset(name: str) -> KmxCircuit:
    return read_kmx_circuit((ASSETS / name).read_text())


def program(circuit: KmxCircuit) -> ReversiblePhaseProgram:
    return ReversiblePhaseProgram(
        circuit.qubits,
        circuit.bits,
        tuple(
            ReversiblePhaseOperation(
                record.opcode.lower(),
                record.qubits,
                record.result_bit,
                record.condition_bit,
                record.angle,
            )
            for record in circuit
        ),
        tuple(range(circuit.bits)),
    )


def instructions(
    circuit: KmxCircuit, classical_value: int = 0
) -> tuple[Instruction, ...]:
    setup = tuple(
        ClassicalXor((), bit, constant=bool((classical_value >> bit) & 1)).instruction()
        for bit in range(circuit.bits)
    )
    return (*setup, *bind_reversible_phase_execution(program(circuit)))
