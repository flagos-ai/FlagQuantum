"""Match the full-register H2 UCCSD states and one Adam update across encodings."""

import pytest
import torch

from examples.algorithms.fermion_fcidump_training import (
    DEFAULT_FCIDUMP,
    _encoded_ansatz,
    energy,
)
from flagquantum.algorithms.chemistry.models.problem import fermion_hamiltonian
from flagquantum.algorithms.fermion.encoding import Encoding, encode_fermion
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.ecosystem.chemistry.fcidump import read_fcidump

pytestmark = pytest.mark.integration


def _encoded_index(index: int, encoding: Encoding) -> int:
    a, b, c, d = ((index >> q) & 1 for q in (3, 2, 1, 0))
    if encoding == "parity":
        bits = (a, a ^ b, a ^ b ^ c, a ^ b ^ c ^ d)
    else:
        bits = (a, a ^ b, c, a ^ b ^ c ^ d)
    return sum(bit << (3 - q) for q, bit in enumerate(bits))


@pytest.mark.parametrize("encoding", ("parity", "bravyi_kitaev"))
@pytest.mark.parametrize("angles", ((0.1, -0.07, 0.2), (-0.2, 0.31, -0.17)))
def test_encoded_h2_uses_same_state_energy_gradient_and_adam_step(
    encoding: Encoding, angles: tuple[float, float, float]
) -> None:
    problem = read_fcidump(DEFAULT_FCIDUMP, max_orbitals=2)
    fermion = fermion_hamiltonian(problem)
    reference = torch.tensor(angles, dtype=torch.float32, requires_grad=True)
    candidate = reference.detach().clone().requires_grad_()
    permutation = [_encoded_index(index, encoding) for index in range(16)]
    reference_state = _encoded_ansatz(reference, "jordan_wigner").state().reshape(-1)
    encoded_state = _encoded_ansatz(candidate, encoding).state().reshape(-1)
    torch.testing.assert_close(
        encoded_state[permutation], reference_state, rtol=0, atol=1e-12
    )

    jw = to_observable(encode_fermion(fermion, n_modes=4).operator)
    mapped = to_observable(
        encode_fermion(fermion, n_modes=4, encoding=encoding).operator
    )
    before = reference.detach().clone()
    jw_value = energy(reference, jw)
    encoded_value = energy(candidate, mapped, encoding=encoding)
    torch.testing.assert_close(encoded_value, jw_value, rtol=0, atol=1e-12)

    torch.autograd.backward((jw_value, encoded_value))
    assert reference.grad is not None and candidate.grad is not None
    assert reference.grad.abs().min().item() > 1e-3
    torch.testing.assert_close(candidate.grad, reference.grad, rtol=0, atol=1e-7)

    torch.optim.Adam([reference], lr=0.05).step()
    torch.optim.Adam([candidate], lr=0.05).step()
    assert (reference - before).norm().item() > 1e-2
    torch.testing.assert_close(candidate, reference, rtol=0, atol=1e-7)
    with torch.no_grad():
        torch.testing.assert_close(
            energy(candidate, mapped, encoding=encoding),
            energy(reference, jw),
            rtol=0,
            atol=1e-8,
        )
