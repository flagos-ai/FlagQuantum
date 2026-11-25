"""Independent fixed-particle CI energy actions in spatial-orbital integrals."""

import numpy as np
import torch

import flagquantum as fq
from flagquantum.algorithms.fermion.lowering import lift_stabilizer_state
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.core.algebra.pauli import PauliOperator

from . import model, variational


def energy(molecule: model.Model, full_state: torch.Tensor) -> float:
    """Map interleaved JW occupations to PySCF alpha/beta strings, retaining signs."""
    from pyscf.fci import cistring, direct_spin1

    strings = tuple(int(value) for value in cistring.make_strings(range(8), 5))
    ci = np.zeros((56, 56), dtype=np.complex128)
    for alpha_index, alpha in enumerate(strings):
        for beta_index, beta in enumerate(strings):
            word, parity = 0, 0
            for orbital in range(8):
                word |= ((alpha >> orbital) & 1) << (15 - 2 * orbital)
                word |= ((beta >> orbital) & 1) << (14 - 2 * orbital)
                if (beta >> orbital) & 1:
                    parity += (alpha >> (orbital + 1)).bit_count()
            ci[alpha_index, beta_index] = (-1) ** parity * complex(full_state[word])
    if abs(float(np.linalg.norm(ci)) - float(full_state.norm())) > 1e-12:
        raise ValueError(
            "The native state escaped the original five-alpha/five-beta space"
        )
    integrals = direct_spin1.absorb_h1e(
        molecule.one_body.numpy(), molecule.two_body.numpy(), 8, (5, 5), 0.5
    )
    # PySCF's real contraction supports real CI vectors; apply linearity explicitly.
    action = direct_spin1.contract_2e(integrals, ci.real.copy(), 8, (5, 5))
    action = action + 1j * direct_spin1.contract_2e(
        integrals, ci.imag.copy(), 8, (5, 5)
    )
    value = np.vdot(ci, action) + molecule.nuclear_energy * np.vdot(ci, ci)
    if abs(value.imag) > 1e-10:
        raise ValueError("The independent Hermitian energy has an imaginary residual")
    return float(value.real)


def check_energies(
    molecule: model.Model, generators: tuple[PauliOperator, ...]
) -> tuple[float, ...]:
    errors = []
    observable = to_observable(molecule.reduction.operator)
    for scale in (0.0, 0.17, -0.23):
        angles = tuple(scale * (index + 1) / 30 for index in range(30))
        program = variational.circuit(molecule, generators, angles)
        full = lift_stabilizer_state(molecule.reduction, program.state().reshape(-1))
        expected = energy(molecule, full)
        result = fq.run(program, outputs=[fq.expectation(observable)])
        actual = float(torch.as_tensor(result.expectation()).sum().real)
        errors.append(abs(actual - expected))
    if max(errors) > 1e-9:
        raise ValueError(
            "Native full-Hamiltonian energies disagree with independent CI contractions"
        )
    return tuple(errors)
