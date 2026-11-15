# Copyright 2020 The Cirq Developers
# SPDX-License-Identifier: Apache-2.0
# Specializes the single-control A/B/C construction used by the frozen
# Qualtran v0.7.0 Hadamard notebook with Cirq 1.7.
# https://github.com/quantumlib/Cirq/blob/v1.7.0/cirq-core/cirq/transformers/analytical_decompositions/controlled_gate_decomposition.py

"""Native teaching circuit for the source controlled-H specialty example."""

from math import pi

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.core.ir import Instruction
from flagquantum.simulation.unitary import get_unitary


def controlled_hadamard_specialty() -> FieldOperation:
    """Bind the six A/B/C construction leaves to control 0 and target 1."""
    # For H the determinant phase is pi/2 and the Euler angles are
    # alpha = -pi, beta = 0, theta = pi/2 in the source construction.
    dtype = torch.complex128
    c = get_unitary(Circuit(1, dtype=dtype).rz(0, -pi / 2))
    b = get_unitary(Circuit(1, dtype=dtype).rz(0, -pi / 2).ry(0, pi / 4))
    a = get_unitary(Circuit(1, dtype=dtype).ry(0, -pi / 4).rz(0, pi))
    cx = Instruction("cx", (0, 1))
    return FieldOperation(
        2,
        (
            Instruction("phase", (0,), params={"theta": pi / 2}),
            Instruction("any", (1,), matrix=c),
            cx,
            Instruction("any", (1,), matrix=b),
            cx,
            Instruction("any", (1,), matrix=a),
        ),
    )
