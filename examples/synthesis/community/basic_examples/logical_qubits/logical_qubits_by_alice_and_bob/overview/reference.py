"""Direct unmerged original-clock density oracle independent of native channels."""

import math

import numpy as np
from numpy.typing import NDArray

import flagquantum as fq

from .noise import Parameters

ComplexArray = NDArray[np.complex128]


def direct(circuit: fq.Circuit, parameters: Parameters) -> ComplexArray:
    """Execute every H/T/CX/CCX and all idle/preparation/readout error locations."""
    width = int(circuit.n_qubits)
    size = 1 << width
    indices = np.arange(size)
    clocks = [0] * width
    events: dict[int, list[tuple[str, tuple[int, ...]]]] = {}
    for instruction in circuit.to_ir().instructions:
        moment = max(clocks[wire] for wire in instruction.wires)
        events.setdefault(moment + 1, []).append((instruction.name, instruction.wires))
        for wire in instruction.wires:
            clocks[wire] = moment + 1
    final = max(clocks) + 1
    events[0] = []
    events[final] = []
    density = np.zeros((size, size), dtype=np.complex128)
    density[0, 0] = 1
    probabilities = parameters.pauli_probabilities
    bit, phase = (
        probabilities[1] + probabilities[2],
        probabilities[2] + probabilities[3],
    )
    signs = [1 - 2 * ((indices >> (width - wire - 1)) & 1) for wire in range(width)]
    flips = [indices ^ (1 << (width - wire - 1)) for wire in range(width)]
    phase_masks = [1 - phase + phase * sign[:, None] * sign[None, :] for sign in signs]

    def left_h(value: ComplexArray, wire: int) -> ComplexArray:
        shaped = value.reshape(1 << wire, 2, 1 << (width - wire - 1), size)
        result = np.stack(
            (shaped[:, 0] + shaped[:, 1], shaped[:, 0] - shaped[:, 1]), axis=1
        )
        return result.reshape(size, size) / math.sqrt(2)

    for moment in range(final + 1):
        for name, wires in events.get(moment, []):
            if name == "h":
                density = left_h(density, wires[0])
                density = left_h(density.conj().T, wires[0]).conj().T
            elif name in ("x", "cx", "ccx"):
                permutation = indices.copy()
                selected = np.ones(size, dtype=bool)
                for wire in wires[:-1]:
                    selected &= ((indices >> (width - wire - 1)) & 1) != 0
                permutation[selected] ^= 1 << (width - wires[-1] - 1)
                density = density[permutation][:, permutation]
            elif name in ("t", "tdg", "s", "sdg", "z"):
                angles = {
                    "t": np.pi / 4,
                    "tdg": -np.pi / 4,
                    "s": np.pi / 2,
                    "sdg": -np.pi / 2,
                    "z": np.pi,
                }
                diagonal = np.exp(1j * angles[name] * (1 - signs[wires[0]]) / 2)
                density = diagonal[:, None] * density * diagonal.conj()[None, :]
            else:
                raise ValueError(
                    "The independent oracle requires discrete cat instructions"
                )
        for wire in range(width):
            density = (1 - bit) * density + bit * density[flips[wire]][:, flips[wire]]
            density *= phase_masks[wire]
    return density
