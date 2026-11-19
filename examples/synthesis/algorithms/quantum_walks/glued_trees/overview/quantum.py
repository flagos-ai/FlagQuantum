"""Build checked Pauli evolution with the full original source time budgets."""

import math
from dataclasses import dataclass
from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.pauli_evolution import Term, load_operator
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, allocate, qfunc, suzuki_trotter

from .model import decompose, fresh_matrix

REPETITIONS = 10
HAMILTONIAN_20 = load_operator(Path(__file__).with_name("glued_trees_20_terms.json"))
MATRIX_6, EDGES_6 = fresh_matrix()
TERMS_6, EXPANSION_ERROR = decompose(MATRIX_6)


def make_operator(terms: tuple[Term, ...]) -> fq.Observable:
    axes = {"X": fq.X, "Y": fq.Y, "Z": fq.Z}
    value: fq.Observable | None = None
    for coefficient, support in terms:
        pauli = fq.I()
        for bit, axis in support:
            pauli = pauli @ axes[axis](bit)
        weighted = coefficient * pauli
        value = weighted if value is None else value + weighted
    if value is None:
        raise ValueError("source Hamiltonian requires Pauli terms")
    return value


FIRST = make_operator(TERMS_6[:1024])
LAST = make_operator(TERMS_6[1024:])


@qfunc
def six_qubit_slice(duration: float, state: Output[QArray[QBit]]) -> None:
    allocate(6, state)
    suzuki_trotter(FIRST, duration / 10, 1, 1, state)
    suzuki_trotter(LAST, duration / 10, 1, 1, state)


@qfunc
def twenty_qubit_walk(duration: float, state: Output[QArray[QBit]]) -> None:
    allocate(20, state)
    suzuki_trotter(HAMILTONIAN_20, duration, 1, 10, state)


@dataclass(frozen=True)
class State:
    amplitudes: torch.Tensor
    operations: int
    max_physical_width: int
    work_qubits: int


def evolve(qubits: int, duration: float) -> State:
    if qubits not in (6, 20) or not math.isfinite(duration):
        raise ValueError("this example models finite times at six or twenty qubits")
    entrypoint = six_qubit_slice if qubits == 6 else twenty_qubit_walk
    program = synth.synthesize(
        entrypoint,
        classical_inputs={"duration": duration},
        max_qubits=qubits,
        max_workspace=0,
        max_operations=100000,
        memory_limit_bytes=512 * 1024**2,
    )
    wires = program.registers["state"]
    if wires != tuple(range(qubits)):
        raise ValueError(
            "this example requires the source basis word in named mode order"
        )
    if qubits == 20:
        return State(
            fq.run(program).statevector().flatten(),
            len(program.circuit.instructions),
            qubits,
            0,
        )
    # One coherent execution: the public Circuit composes all ten slices first.
    block = fq.Circuit.from_ir(program.circuit, dtype=torch.complex128)
    circuit = fq.Circuit(qubits, dtype=torch.complex128)
    for _ in range(REPETITIONS):
        circuit.compose(block)
    return State(
        fq.run(circuit).statevector().flatten(),
        len(circuit.to_ir().instructions),
        qubits,
        0,
    )
