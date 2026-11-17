"""Compose exact arithmetic, controlled rotations, and walk blocks."""

import math
from collections.abc import Sequence

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    allocate,
    inplace_prepare_amplitudes,
    qfunc,
)

from .model import COMPLEX

TREE_THETA = (
    2 * math.acos(0.3),
    2 * math.acos(0.5 / 4),
    2 * math.acos(0.7 / 4),
    2 * math.acos(0.7 / 4 - 0.3 / 2) - 2 * math.acos(0.5 / 4),
)
CYCLIC_THETA = (2 * math.atan(math.sqrt(0.6 / 0.4)), 2 * math.atan(1.0))
CHEB_AMPLITUDES = (1 / 4, 1 / 2, 1 / 2, 1 / 4, 1 / 2, 1 / 4, 1 / 4, 0.0)


def controlled(
    dest: fq.Circuit,
    source: fq.Circuit,
    data: Sequence[int],
    condition: Sequence[tuple[int, int]],
) -> None:
    """Compose a public controlled block at the requested basis values."""
    if not condition:
        dest.compose(source, qubits=data)
        return
    controls = tuple(wire for wire, _ in condition)
    negative = tuple(wire for wire, value in condition if value == 0)
    for wire in negative:
        dest.x(wire)
    size = source.n_qubits
    wrapped = source.control(len(controls), tuple(range(size, size + len(controls))))
    dest.compose(wrapped, qubits=(*data, *controls))
    for wire in reversed(negative):
        dest.x(wire)


def word_condition(wires: Sequence[int], word: int) -> tuple[tuple[int, int], ...]:
    return tuple((wire, (word >> index) & 1) for index, wire in enumerate(wires))


def rotation(angle: float) -> fq.Circuit:
    return fq.Circuit(1, dtype=COMPLEX).ry(0, angle)


def modular_increment(bits: int) -> fq.Circuit:
    """Implement modular increment in least-significant-wire order."""
    circuit = fq.Circuit(bits, dtype=COMPLEX)
    flip = fq.Circuit(1, dtype=COMPLEX).x(0)
    for index in range(bits - 1, 0, -1):
        controlled(
            circuit,
            flip,
            (index,),
            word_condition(tuple(range(index)), (1 << index) - 1),
        )
    return circuit.x(0)


def tree_encoding(bits: int) -> fq.Circuit:
    """Use the source selector, rotation and reversible multiply-by-two network."""
    labels = (2, 1, 0)
    data, spill = 3, 4
    descending = tuple(range(5, 5 + bits))
    vertices = tuple(reversed(descending))
    circuit = fq.Circuit(5 + bits, dtype=COMPLEX)
    for wire in labels:
        circuit.h(wire)
    first, internal, leaf, root = TREE_THETA
    controlled(circuit, rotation(first), (data,), ((labels[2], 0),))
    controlled(
        circuit, rotation(internal), (data,), ((labels[2], 1), (vertices[-1], 0))
    )
    controlled(circuit, rotation(leaf), (data,), ((labels[2], 1), (vertices[-1], 1)))
    controlled(
        circuit, rotation(root), (data,), ((labels[2], 1), *word_condition(vertices, 0))
    )

    doubling = fq.Circuit(bits + 1, dtype=COMPLEX).swap(bits - 1, bits)
    for index in range(bits - 1, 0, -1):
        doubling.swap(index, index - 1)
    increment = modular_increment(bits)
    controlled(circuit, doubling, (*vertices, spill), ((labels[2], 0), (labels[1], 0)))
    controlled(circuit, increment, vertices, word_condition(labels, 1))
    controlled(circuit, increment.adjoint(), vertices, word_condition(labels, 3))
    controlled(
        circuit,
        doubling.adjoint(),
        (*vertices, spill),
        ((labels[2], 0), (labels[1], 1)),
    )
    for wire in labels:
        circuit.h(wire)
    return circuit


def cyclic_encoding() -> fq.Circuit:
    """Wrap the register swap in the source controlled adders and state selector."""
    block_desc = (0, 1, 2)
    state_desc = (3, 4, 5)
    block = tuple(reversed(block_desc))
    state = tuple(reversed(state_desc))
    selector = fq.Circuit(3, dtype=COMPLEX).ry(2, CYCLIC_THETA[0])
    selector.cry(2, 1, theta=CYCLIC_THETA[1]).ccx(2, 1, 0)
    adder = fq.Circuit(6, dtype=COMPLEX)
    controlled(adder, modular_increment(3), block, ((state[0], 1),))
    controlled(adder, modular_increment(2), block[1:], ((state[1], 1),))
    adder.cx(state[2], block[2])
    circuit = fq.Circuit(6, dtype=COMPLEX).compose(selector, qubits=block_desc)
    circuit.compose(adder)
    for a, b in zip(block, state, strict=True):
        circuit.swap(a, b)
    return circuit.compose(adder.adjoint()).compose(
        selector.adjoint(), qubits=block_desc
    )


@qfunc
def cheb_preparation(amplitudes: list[float], state: Output[QArray[QBit]]) -> None:
    allocate(3, state)
    inplace_prepare_amplitudes(amplitudes, 0.0, state)


def reflect_zero(circuit: fq.Circuit, wires: tuple[int, ...]) -> None:
    """Multiply by minus one precisely on the all-zero selected wires."""
    for wire in wires:
        circuit.x(wire)
    phase = fq.Circuit(1, dtype=COMPLEX).phase(0, theta=math.pi)
    controlled(circuit, phase, (wires[-1],), tuple((wire, 1) for wire in wires[:-1]))
    for wire in reversed(wires):
        circuit.x(wire)


def cheb_circuit() -> fq.Circuit:
    program = synth.synthesize(
        cheb_preparation, classical_inputs={"amplitudes": CHEB_AMPLITUDES}
    )
    circuit = (
        fq.Circuit(7, dtype=COMPLEX).compose(program.circuit, qubits=(4, 5, 6)).h(0)
    )
    for _ in range(2):
        circuit.z(0)
        reflect_zero(circuit, (1, 2, 3))
        circuit.compose(cyclic_encoding(), qubits=(1, 2, 3, 4, 5, 6))
    # Each source i·Z adds a factor i. Two applications give global -1.
    return circuit.x(0).phase(0, theta=math.pi).x(0).phase(0, theta=math.pi).h(0)


def discriminant_walk(bits: int, *, marked: bool) -> fq.Circuit:
    """Implement the exact source H/SWAP structure and marked rotations."""
    ancilla = 0
    block = tuple(range(1, bits + 1))
    state = tuple(range(bits + 1, 2 * bits + 1))
    circuit = fq.Circuit(2 * bits + 1, dtype=COMPLEX)
    # Conjugating Z on the initially clean ancilla reflects the zero block.
    circuit.z(ancilla)
    reflect_zero(circuit, block)
    rotations = fq.Circuit(circuit.n_qubits, dtype=COMPLEX)
    if marked:
        for target in block:
            controlled(rotations, rotation(1.5708), (target,), word_condition(state, 0))
        circuit.compose(rotations)
    for target in (*block, *state):
        circuit.h(target)
    for x, y in zip(block, state, strict=True):
        circuit.swap(x, y)
    return circuit.compose(rotations.adjoint()) if marked else circuit
