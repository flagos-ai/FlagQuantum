"""Use native precision for PREPARE and public SELECT for each walk step."""

import math

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import (
    SelectPauliLCU,
)
from flagquantum.synth import Output, QArray, QBit, prepare_amplitudes, qfunc

from .model import ALPHA, DTYPE, WEIGHTS, WORDS, Expansion

HAMILTONIAN_FLAGS = 2
EVOLUTION_FLAGS = 6


@qfunc
def prepare(values: list[float], output: Output[QArray[QBit]]) -> None:
    prepare_amplitudes(values, 0, output)


def preparation(values: tuple[float, ...]) -> fq.Circuit:
    """Prepare a source-sized real vector with double-precision lowering."""
    width = len(values).bit_length() - 1
    program = synth.synthesize(
        prepare,
        classical_inputs={"values": values},
        max_qubits=width,
        max_workspace=0,
    )
    return fq.Circuit.from_ir(program.circuit, dtype=DTYPE)


def _hamiltonian_preparation() -> fq.Circuit:
    return preparation(tuple(math.sqrt(weight / ALPHA) for weight in WEIGHTS))


def hamiltonian_block(data: tuple[float, ...]) -> fq.Circuit:
    """Apply the source Pauli block encoding to a prepared two-qubit input."""
    selectors, system = (0, 1), (2, 3)
    weights = _hamiltonian_preparation()
    circuit = fq.Circuit(4, dtype=DTYPE).compose(preparation(data), qubits=system)
    circuit.compose(weights, qubits=selectors)
    circuit.append(
        SelectPauliLCU(2, 2, WORDS).make_on(selection=selectors, target=system)
    )
    circuit.compose(weights.adjoint(), qubits=selectors)
    return circuit


def controlled_walk() -> fq.Circuit:
    """Append one controlled Hermitian block encoding followed by reflection."""
    selectors, system, control = (0, 1), (2, 3), 4
    weights = _hamiltonian_preparation()
    circuit = fq.Circuit(5, dtype=DTYPE).compose(weights, qubits=selectors)
    circuit.append(
        SelectPauliLCU(2, 2, WORDS, control_val=1).make_on(
            selection=selectors, target=system, control=(control,)
        )
    )
    circuit.compose(weights.adjoint(), qubits=selectors)
    # The control-zero block is identity. The control-one block reflects about |00>.
    reflection = torch.diag(torch.tensor([1, 1, 1, 1, 1, -1, -1, -1], dtype=DTYPE))
    return circuit.any(control, *selectors, matrix=reflection)


def evolution_block(data: tuple[float, ...], poly: Expansion) -> fq.Circuit:
    """Select all source walk powers, with a complex phase for each coefficient."""
    slots = 1 << EVOLUTION_FLAGS
    probabilities = tuple(abs(value) / poly.scale for value in poly.coefficients)
    weights = preparation(
        tuple(map(math.sqrt, probabilities)) + (0.0,) * (slots - len(probabilities))
    )
    phases = tuple(value / abs(value) if value else 1 for value in poly.coefficients)
    phases += (1.0,) * (slots - len(phases))
    selectors = tuple(range(EVOLUTION_FLAGS))
    system = (8, 9)
    circuit = (
        fq.Circuit(10, dtype=DTYPE)
        .compose(preparation(data), qubits=system)
        .compose(weights, qubits=selectors)
    )
    controlled = controlled_walk()
    for power, control in enumerate(reversed(selectors)):
        for _ in range(1 << power):
            circuit.compose(controlled, qubits=(6, 7, *system, control))
    circuit.any(*selectors, matrix=torch.diag(torch.tensor(phases, dtype=DTYPE)))
    circuit.compose(weights.adjoint(), qubits=selectors)
    return circuit
