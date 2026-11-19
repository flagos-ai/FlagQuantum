"""Compose every logical clock's cat noise into bounded native qubit channels."""

import math
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

import flagquantum as fq
from flagquantum.noise import KrausChannel, NoiseModel, noisy_density_matrix

from .noise import Parameters

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
PAULIS = np.array(
    [np.eye(2), [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
    dtype=np.complex128,
)


@dataclass(frozen=True)
class Execution:
    density: torch.Tensor
    clocks: int
    source_instructions: int
    one_qubit_noise_applications: int
    merged_channels: int


def rotation(name: str) -> RealArray:
    matrices = {
        "i": np.eye(2, dtype=np.complex128),
        "h": np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2),
        "x": PAULIS[1],
        "z": PAULIS[3],
        "s": np.diag([1, 1j]),
        "sdg": np.diag([1, -1j]),
        "t": np.diag([1, np.exp(1j * np.pi / 4)]),
        "tdg": np.diag([1, np.exp(-1j * np.pi / 4)]),
    }
    matrix = matrices[name]
    return np.array(
        [
            [np.trace(a @ matrix @ b @ matrix.conj().T).real / 2 for b in PAULIS[1:]]
            for a in PAULIS[1:]
        ],
        dtype=np.float64,
    )


def kraus(transfer: RealArray, name: str) -> KrausChannel:
    """Convert an unital Bloch transfer to its exact two-dimensional Choi channel."""
    choi = np.kron(PAULIS[0].T, PAULIS[0])
    for index, source in enumerate(PAULIS[1:]):
        output = sum(
            (transfer[row, index] * PAULIS[row + 1] for row in range(3)),
            np.zeros((2, 2), dtype=np.complex128),
        )
        choi += np.kron(source.T, output)
    choi /= 2
    values, vectors = np.linalg.eigh(choi)
    if float(values.min()) < -1e-11:
        raise ValueError("The composed cat channel is not completely positive")
    operators = tuple(
        torch.from_numpy(
            np.sqrt(max(float(value), 0.0)) * vectors[:, index].reshape(2, 2).T.copy()
        )
        for index, value in enumerate(values)
        if value > 0
    )
    effect = sum(
        (operator.mH @ operator for operator in operators),
        torch.zeros((2, 2), dtype=torch.complex128),
    )
    if float((effect - torch.eye(2, dtype=torch.complex128)).abs().max()) > 1e-10:
        raise ValueError("The composed cat channel lost trace preservation")
    return KrausChannel(name, operators)


def execute(circuit: fq.Circuit, parameters: Parameters) -> Execution:
    """Keep all ASAP clocks, including preparation, readout and idle-wire correction."""
    width = int(circuit.n_qubits)
    instructions = circuit.to_ir().instructions
    layers: list[list[tuple[str, tuple[int, ...]]]] = []
    ready = [0] * width
    for instruction in instructions:
        name, wires = instruction.name, instruction.wires
        if name not in ("h", "x", "z", "s", "sdg", "t", "tdg", "cx", "ccx"):
            raise ValueError(
                "The logical cat execution requires discrete native instructions"
            )
        layer = max(ready[wire] for wire in wires)
        while len(layers) <= layer:
            layers.append([])
        layers[layer].append((name, wires))
        for wire in wires:
            ready[wire] = layer + 1
    # Every preparation starts in |0>; an identity followed by the p0 channel is exact.
    layers = [[("i", (wire,)) for wire in range(width)], *layers, [("i", (0,))]]
    probabilities = parameters.pauli_probabilities
    diagonal = np.diag(
        [
            1 - 2 * (probabilities[2] + probabilities[3]),
            1 - 2 * (probabilities[1] + probabilities[3]),
            1 - 2 * (probabilities[1] + probabilities[2]),
        ]
    )
    pending = [np.eye(3, dtype=np.float64) for _ in range(width)]
    rotations = {
        name: rotation(name) for name in ("i", "h", "x", "z", "s", "sdg", "t", "tdg")
    }
    native = fq.Circuit(width, dtype=torch.complex128)
    noise = NoiseModel()
    merged = 0

    def flush() -> None:
        nonlocal merged
        for wire in range(width):
            label = f"cat_clock_block_{merged}"
            native.any(wire, matrix=torch.eye(2, dtype=torch.complex128), name=label)
            noise.add(label, kraus(pending[wire], label))
            pending[wire] = np.eye(3, dtype=np.float64)
            merged += 1

    for layer in layers:
        entangling = [(name, wires) for name, wires in layer if len(wires) > 1]
        if entangling:
            flush()
            for name, wires in entangling:
                native.gate(name, wires)
        active = {wires[0]: name for name, wires in layer if len(wires) == 1}
        for wire in range(width):
            pending[wire] = diagonal @ rotations[active.get(wire, "i")] @ pending[wire]
    flush()
    density = noisy_density_matrix(native, noise, dtype=torch.complex128).reshape(
        1 << width, 1 << width
    )
    if abs(complex(density.trace()) - 1) > 1e-9:
        raise ValueError("The complete native cat simulation lost total probability")
    return Execution(
        density, len(layers), len(instructions), width * len(layers), merged
    )
