"""Compare actual reduced H2 UCCSD training with the full JW workflow."""

import pytest
import torch

from examples.algorithms.fermion_fcidump_training import (
    DEFAULT_FCIDUMP,
    _encoded_ansatz,
    train_adam,
)
from examples.algorithms.fermion_fcidump_training import (
    energy as full_energy,
)
from examples.algorithms.fermion_h2_tapered_uccsd import (
    _reduce_h2,
    build_ansatz,
    prepare_h2,
)
from examples.algorithms.fermion_h2_tapered_uccsd import (
    energy as reduced_energy,
)
from flagquantum.algorithms.chemistry import excitation_operator
from flagquantum.algorithms.chemistry_domain.problem import fermion_hamiltonian
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.ecosystem.chemistry.fcidump import read_fcidump

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("angles", ((0.1, -0.07, 0.2), (-0.2, 0.31, -0.17)))
def test_tapered_h2_matches_full_jw_state_energy_gradient_and_update(
    angles: tuple[float, float, float],
) -> None:
    hamiltonian, generators = prepare_h2()
    observable = to_observable(hamiltonian.operator)
    problem = read_fcidump(DEFAULT_FCIDUMP, max_orbitals=2)
    full_observable = to_observable(
        encode_fermion(fermion_hamiltonian(problem), n_modes=4).operator
    )

    def four_qubit(values: torch.Tensor) -> torch.Tensor:
        return full_energy(values, full_observable)

    def two_qubit(values: torch.Tensor) -> torch.Tensor:
        return reduced_energy(values, hamiltonian, generators, observable)

    full_angles = torch.tensor(angles, dtype=torch.float32, requires_grad=True)
    tapered_angles = full_angles.detach().clone().requires_grad_()
    before = full_angles.detach().clone()
    with torch.no_grad():
        full_state = _encoded_ansatz(full_angles, "jordan_wigner").state().reshape(-1)
        circuit = build_ansatz(tapered_angles, hamiltonian, generators)
        assert circuit.n_qubits == 2
        reduced_state = circuit.state().reshape(-1)
        lifted = hamiltonian.mapping.lift_state(reduced_state)
        # Independent occupations, ordered by the two virtual spin bits.
        direct = torch.zeros_like(full_state)
        direct[[0b1100, 0b1001, 0b0110, 0b0011]] = reduced_state
        torch.testing.assert_close(lifted, direct, atol=0, rtol=0)
        torch.testing.assert_close(lifted, full_state, atol=1e-12, rtol=0)

    full_value, tapered_value = four_qubit(full_angles), two_qubit(tapered_angles)
    torch.testing.assert_close(tapered_value, full_value, atol=1e-12, rtol=0)
    (full_gradient,) = torch.autograd.grad(full_value, full_angles)
    (tapered_gradient,) = torch.autograd.grad(tapered_value, tapered_angles)
    assert full_gradient.abs().min().item() > 1e-3
    torch.testing.assert_close(tapered_gradient, full_gradient, atol=1e-7, rtol=0)

    train_adam(full_angles, four_qubit, steps=1)
    train_adam(tapered_angles, two_qubit, steps=1)
    assert (full_angles - before).norm().item() > 1e-2
    torch.testing.assert_close(tapered_angles, full_angles, atol=1e-7, rtol=0)
    with torch.no_grad():
        torch.testing.assert_close(
            two_qubit(tapered_angles), four_qubit(full_angles), atol=1e-8, rtol=0
        )


def test_tapering_refuses_an_excitation_that_changes_spin_parity() -> None:
    with pytest.raises(
        ValueError, match="does not preserve the requested parity sector"
    ):
        _reduce_h2(1j * excitation_operator((0,), (3,)))
