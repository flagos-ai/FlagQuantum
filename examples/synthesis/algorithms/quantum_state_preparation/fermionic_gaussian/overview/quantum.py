"""Prepare the source Slater orbital and two-mode paired vacuum in native qfuncs."""

import cmath
import math

import torch

from examples.synthesis.applications.physical_systems.fermi_hubbard_model_1d.fermi_hubbard_1d.overview.model import (
    factor,
)
from examples.synthesis.applications.physical_systems.fermi_hubbard_model_1d.fermi_hubbard_1d.overview.quantum import (
    build_program as build_slater_program,
)
from flagquantum import synth
from flagquantum.synth import CX, Output, QArray, QBit, R, X, allocate, phase, qfunc

from .model import periodic_hamiltonian


@qfunc
def prepare_pair(
    angle: float, relative_phase: float, parity: int, q: Output[QArray[QBit]]
) -> None:
    allocate(2, q)
    R(angle, math.pi / 2, q[0])
    phase(q[0], relative_phase)
    CX(q[0], q[1])
    if parity == 1:
        X(q[0])


def slater_program() -> tuple[synth.Program, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(periodic_hamiltonian())
    orbital = eigenvectors[:, :1]
    return build_slater_program(factor(orbital), 0, 0.0, 0.0), eigenvalues


def pairing_program(vector: torch.Tensor) -> synth.Program:
    if vector.shape != (4,) or not torch.isfinite(vector).all():
        raise ValueError("expect four finite Fock amplitudes")
    weights = vector.abs().square()
    even, odd = float(weights[0] + weights[3]), float(weights[1] + weights[2])
    if min(even, odd) > 1e-24 or abs(even + odd - 1) > 1e-12:
        raise ValueError("expected a normalized vacuum with definite parity")
    parity = int(odd > even)
    low, high = (1, 2) if parity else (0, 3)
    a, b = complex(vector[low]), complex(vector[high])
    angle = 2 * math.atan2(abs(b), abs(a))
    relative = (
        cmath.phase(b) - cmath.phase(a) if abs(a) > 1e-14 and abs(b) > 1e-14 else 0.0
    )
    return synth.synthesize(
        prepare_pair,
        classical_inputs={"angle": angle, "relative_phase": relative, "parity": parity},
        max_qubits=2,
        max_workspace=0,
        max_operations=64,
    )
