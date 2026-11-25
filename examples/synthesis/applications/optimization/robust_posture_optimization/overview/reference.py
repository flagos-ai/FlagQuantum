"""Independent literal Circuit rotations, including absolute phase and source bases."""

import torch

import flagquantum as fq

from .quantum import Basis


def reference_circuit(parameters: tuple[float, ...], basis: Basis) -> fq.Circuit:
    circuit = fq.Circuit(2, dtype=torch.complex128)
    # Native q[0] is the low bit of the named array word.
    for wire, offset in ((1, 0), (0, 3), (0, 0)):
        circuit.rz(wire, theta=parameters[offset + 2])
        circuit.ry(wire, theta=parameters[offset + 1])
        circuit.rx(wire, theta=parameters[offset])
    for wire in (1, 0):
        if basis == "Y":
            circuit.s(wire)
        if basis in ("X", "Y"):
            circuit.h(wire)
    return circuit
