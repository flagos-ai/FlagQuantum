"""Optional Qiskit Aer execution adapters kept outside core runtime."""

from __future__ import annotations

from typing import Any

import torch

from ...runtime.dynamic import export_dynamic_qasm3
from ...runtime.dynamic._conditions import classical_width as _classical_width
from ...runtime.dynamic._conditions import (
    instruction_conditions as _instruction_conditions,
)
from ...runtime.dynamic.circuit import DynamicCircuit
from ...runtime.dynamic.result import DynamicExecutionResult
from .models import QiskitDependencyError


def _aer_types() -> tuple[Any, Any, Any, Any]:
    try:
        from qiskit import ClassicalRegister, QuantumCircuit, qasm3
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise QiskitDependencyError(
            "Qiskit Aer execution requires the 'qiskit' optional dependency; "
            "install it with `pip install 'flagquantum[qiskit]'`."
        ) from exc
    return QuantumCircuit, ClassicalRegister, qasm3, AerSimulator


def run_qiskit_aer_dynamic(
    circuit: DynamicCircuit,
    *,
    shots: int,
    seed: int | None = None,
) -> DynamicExecutionResult:
    """Execute a dynamic FlagQuantum circuit on local Qiskit Aer."""

    (
        quantum_circuit_type,
        _classical_register_type,
        _qasm3,
        aer_simulator_type,
    ) = _aer_types()
    width = _classical_width(circuit)
    qiskit_circuit = quantum_circuit_type(circuit.n_wires, width + circuit.n_wires)

    def apply(instruction: Any) -> None:
        method = getattr(qiskit_circuit, instruction.name, None)
        if method is None:
            raise ValueError(
                f"Qiskit Aer dynamic gate is unsupported: {instruction.name}"
            )
        params = tuple(instruction.params.values())
        method(*params, *instruction.wires)

    for instruction in circuit._instructions:
        conditions = _instruction_conditions(instruction)
        if instruction.name == "measure":
            qiskit_circuit.measure(
                instruction.wires[0],
                int(instruction.metadata["classical_bit"]),
            )
        elif instruction.name == "reset":
            qiskit_circuit.reset(instruction.wires[0])
        elif conditions:
            if len(conditions) != 1:
                raise ValueError(
                    "Qiskit Aer adapter currently requires one condition bit"
                )
            bit, expected = conditions[0]
            with qiskit_circuit.if_test((qiskit_circuit.clbits[bit], bool(expected))):
                apply(instruction)
        else:
            apply(instruction)
    for wire in range(circuit.n_wires):
        qiskit_circuit.measure(wire, width + wire)

    memory = (
        aer_simulator_type()
        .run(
            qiskit_circuit,
            shots=int(shots),
            memory=True,
            seed_simulator=seed,
        )
        .result()
        .get_memory(qiskit_circuit)
    )
    rows = [[int(char) for char in item.replace(" ", "")[::-1]] for item in memory]
    classical = torch.tensor(
        [row[:width] for row in rows],
        dtype=torch.int64,
    )
    samples = torch.tensor(
        [row[width : width + circuit.n_wires] for row in rows],
        dtype=torch.int64,
    )
    return DynamicExecutionResult(
        samples=samples,
        classical_bits=classical,
        final_states=torch.empty(0),
        shots=int(shots),
        seed=seed,
        execution_semantics="qiskit_aer_dynamic_shots",
        provider_metadata={"provider": "qiskit-aer", "transport": "python_circuit"},
    )


def run_qiskit_aer_qasm3_round_trip(
    circuit: DynamicCircuit,
    *,
    shots: int,
    seed: int | None = None,
) -> DynamicExecutionResult:
    """Export OpenQASM 3, import it with Qiskit, then execute it on Aer."""

    (
        _quantum_circuit_type,
        classical_register_type,
        qasm3,
        aer_simulator_type,
    ) = _aer_types()
    source = export_dynamic_qasm3(circuit)
    qiskit_circuit = qasm3.loads(source)
    width = _classical_width(circuit)
    final = classical_register_type(circuit.n_wires, "final")
    qiskit_circuit.add_register(final)
    for wire in range(circuit.n_wires):
        qiskit_circuit.measure(wire, final[wire])
    memory = (
        aer_simulator_type()
        .run(
            qiskit_circuit,
            shots=int(shots),
            memory=True,
            seed_simulator=seed,
        )
        .result()
        .get_memory(qiskit_circuit)
    )
    rows = [[int(char) for char in item.replace(" ", "")[::-1]] for item in memory]
    classical = torch.tensor([row[:width] for row in rows], dtype=torch.int64)
    samples = torch.tensor(
        [row[width : width + circuit.n_wires] for row in rows],
        dtype=torch.int64,
    )
    return DynamicExecutionResult(
        samples=samples,
        classical_bits=classical,
        final_states=torch.empty(0),
        shots=int(shots),
        seed=seed,
        execution_semantics="qiskit_aer_openqasm3_round_trip",
        provider_metadata={
            "provider": "qiskit-aer",
            "transport": "openqasm3",
            "qasm": source,
        },
    )


__all__ = ("run_qiskit_aer_dynamic", "run_qiskit_aer_qasm3_round_trip")
