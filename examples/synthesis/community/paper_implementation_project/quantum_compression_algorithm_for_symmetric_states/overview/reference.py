"""Independent Circuit using dense three-bit factors and explicit Toffolis."""

import math

import torch

import flagquantum as fq

from .quantum import Scenario


def reference(scenario: Scenario) -> fq.Circuit:
    width = 5 if scenario == "uniform" else 3
    circuit = fq.Circuit(width, dtype=torch.complex128)
    if scenario == "one_excitation":
        state = torch.tensor(
            [0, 1, 1, 0, 1, 0, 0, 0], dtype=torch.complex128
        ) / math.sqrt(3)
        eye = torch.eye(8, dtype=torch.complex128)
        delta = eye[:, 0] - state
        circuit.any(
            0,
            1,
            2,
            matrix=eye - 2 * torch.outer(delta, delta.conj()) / delta.vdot(delta),
        )
    else:
        for wire in range(width):
            circuit.h(wire)
        if scenario == "nonsymmetric":
            circuit.y(width - 1)
    circuit.cx(width - 2, width - 1).ch(width - 1, width - 2)
    for source in range(2, width):
        n = source + 1
        physical = width - 1 - source
        for b in range(source - 1):
            c = math.sqrt((n - b - 2) / n)
            s = math.sqrt((b + 2) / n)
            matrix = torch.eye(8, dtype=torch.complex128)
            matrix[2, 2], matrix[5, 5] = c, -c
            matrix[2, 5] = matrix[5, 2] = s
            circuit.any(physical, width - 2 - b, width - 1 - b, matrix=matrix)
        matrix = torch.eye(8, dtype=torch.complex128)
        matrix[1, 1] = matrix[4, 4] = math.sqrt(1 - 1 / n)
        matrix[1, 4], matrix[4, 1] = 1 / math.sqrt(n), -1 / math.sqrt(n)
        circuit.any(physical, physical + 1, width - 1, matrix=matrix)
        circuit.cx(physical, physical + 1)
    for source in range(2, width):
        physical = width - 1 - source
        controls = [
            width - 1 - bit
            for bit in range((source + 1).bit_length())
            if (source + 1) >> bit & 1
        ]
        for control in controls:
            circuit.cx(physical, control)
        if len(controls) == 1:
            circuit.cx(controls[0], physical)
        else:
            circuit.ccx(*controls, physical)
    return circuit
