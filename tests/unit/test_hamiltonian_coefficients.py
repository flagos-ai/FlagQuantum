"""Complex coefficient inputs share real-energy semantics and preserve gradients."""

import math
import warnings

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms import Hamiltonian, pauli_term
from flagquantum.simulation.mps.state import MPSState

pytestmark = pytest.mark.unit


def test_mps_chain_combines_complex_coefficients_without_losing_gradients() -> None:
    local = torch.tensor([0.6, 0.8], dtype=torch.complex128)
    state = torch.kron(local, local)
    mps = MPSState.from_statevector(state, 2)
    coefficient = torch.tensor(0.7 + 0.2j, dtype=torch.complex128, requires_grad=True)
    hamiltonian = Hamiltonian(
        [
            pauli_term(coefficient, "Z", (0,)),
            pauli_term(0.2 + 0.3j, "Z", (0,)),
            pauli_term(coefficient, "ZZ", (0, 1)),
        ]
    )
    actual = hamiltonian.expectation(mps)
    expected = sum(term.expectation(state) for term in hamiltonian.terms)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual_gradient = torch.autograd.grad(actual.sum(), coefficient)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), coefficient)[0]
    torch.testing.assert_close(
        actual_gradient, expected_gradient, atol=1e-12, rtol=1e-12
    )


@pytest.mark.parametrize("representation", ["circuit", "mps", "statevector", "density"])
@pytest.mark.parametrize("tensor_coefficient", [False, True])
def test_complex_pauli_coefficient_matches_dense_real_expectation(
    representation: str, tensor_coefficient: bool
) -> None:
    state = torch.tensor([0.6, 0.8], dtype=torch.complex128)
    target: fq.Circuit | MPSState | torch.Tensor
    if representation == "circuit":
        target = fq.Circuit(1, dtype=torch.complex128).ry(0, theta=2 * math.acos(0.6))
    elif representation == "mps":
        target = MPSState.from_statevector(state, 1)
    elif representation == "density":
        target = torch.outer(state, state.conj())
    else:
        target = state
    coefficient: complex | torch.Tensor = complex(1 + 2**-40, 0.7)
    if tensor_coefficient:
        coefficient = torch.tensor(
            coefficient, dtype=torch.complex128, requires_grad=True
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = pauli_term(coefficient, "Z", (0,)).expectation(target)
    pauli_expectation = (state.conj() * torch.tensor([1, -1]) * state).sum()
    expected = (
        torch.as_tensor(coefficient, dtype=torch.complex128) * pauli_expectation
    ).real.reshape(1)
    torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)
    assert result.dtype == torch.float64
    if isinstance(coefficient, torch.Tensor):
        result.sum().backward()
        torch.testing.assert_close(coefficient.grad, pauli_expectation)


@pytest.mark.parametrize("real_dtype", (torch.float32, torch.float64))
def test_hamiltonian_adjoint_matches_autograd_value_and_gradient(real_dtype) -> None:
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    theta = torch.tensor(0.31, dtype=real_dtype, requires_grad=True)
    phi = torch.tensor(-0.27, dtype=real_dtype, requires_grad=True)
    circuit = (
        fq.Circuit(3, dtype=complex_dtype)
        .ry(0, theta)
        .rx(1, phi)
        .cx(0, 2)
        .rzz(1, 2, theta)
    )
    hamiltonian = Hamiltonian(
        (
            pauli_term(0.4, "I", (0,)),
            pauli_term(0.7, "ZZ", (0, 2)),
            pauli_term(-0.2, "Z", (1,)),
            pauli_term(0.3, "ZZ", (1, 2)),
        )
    )
    expected_value = hamiltonian.expectation(circuit).squeeze()
    expected_gradient = torch.autograd.grad(
        expected_value, (theta, phi), retain_graph=True
    )

    actual_value = hamiltonian.expectation(circuit, differentiation="adjoint")
    actual_gradient = torch.autograd.grad(actual_value, (theta, phi))

    tolerance = 2e-5 if real_dtype == torch.float32 else 1e-10
    torch.testing.assert_close(
        actual_value, expected_value, atol=tolerance, rtol=tolerance
    )
    for actual, expected in zip(actual_gradient, expected_gradient, strict=True):
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("real_dtype", (torch.float32, torch.float64))
def test_hamiltonian_adjoint_cpu_direct_matches_gather_rollback(
    real_dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    hamiltonian = Hamiltonian(
        (
            pauli_term(0.4, "I", (0,)),
            pauli_term(0.7, "ZZ", (0, 2)),
            pauli_term(-0.2, "Z", (1,)),
            pauli_term(0.3, "ZZ", (1, 2)),
        )
    )

    def value_and_gradient(cpu_direct: bool):
        monkeypatch.setenv(
            "FQ_STATEVECTOR_ADJOINT_CPU_DIRECT", "1" if cpu_direct else "0"
        )
        parameters = torch.tensor([0.31, -0.27], dtype=real_dtype, requires_grad=True)
        circuit = (
            fq.Circuit(3, dtype=complex_dtype)
            .ry(0, parameters[0])
            .rx(1, parameters[1])
            .cx(0, 2)
            .rzz(1, 2, parameters[0])
        )
        value = hamiltonian.expectation(circuit, differentiation="adjoint")
        return value.detach(), torch.autograd.grad(value, parameters)[0]

    rollback_value, rollback_gradient = value_and_gradient(False)
    direct_value, direct_gradient = value_and_gradient(True)
    tolerance = 2e-5 if real_dtype == torch.float32 else 1e-10
    torch.testing.assert_close(
        direct_value, rollback_value, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        direct_gradient, rollback_gradient, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize(
    ("hamiltonian", "error", "match"),
    [
        (Hamiltonian((pauli_term(1.0, "X", (0,)),)), ValueError, "only Z and ZZ"),
        (
            Hamiltonian((pauli_term(1.0 + 0.1j, "Z", (0,)),)),
            ValueError,
            "real Hamiltonian",
        ),
        (
            Hamiltonian(
                (pauli_term(torch.tensor(1.0, requires_grad=True), "Z", (0,)),)
            ),
            ValueError,
            "constant scalar",
        ),
    ],
)
def test_hamiltonian_adjoint_rejects_unsupported_observables(
    hamiltonian, error, match
) -> None:
    theta = torch.tensor(0.2, requires_grad=True)
    circuit = fq.Circuit(1).ry(0, theta)

    with pytest.raises(error, match=match):
        hamiltonian.expectation(circuit, differentiation="adjoint")
