"""Correctness and refusal boundaries for the SV-011 prototype."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from flagquantum.kernels.triton.statevector_pauli_rotation import (
    apply_complex64_local_pauli_rotation_2q,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu, pytest.mark.triton]


def _pauli_matrix(kind: str, *, device: torch.device) -> torch.Tensor:
    one = torch.tensor(1.0, device=device, dtype=torch.complex64)
    zero = torch.tensor(0.0, device=device, dtype=torch.complex64)
    imaginary = torch.tensor(1.0j, device=device, dtype=torch.complex64)
    matrices = {
        "X": torch.stack((torch.stack((zero, one)), torch.stack((one, zero)))),
        "Y": torch.stack(
            (torch.stack((zero, -imaginary)), torch.stack((imaginary, zero)))
        ),
        "Z": torch.stack((torch.stack((one, zero)), torch.stack((zero, -one)))),
    }
    return torch.kron(matrices[kind[0]], matrices[kind[1]])


def _reference(
    state: torch.Tensor,
    angles: torch.Tensor,
    *,
    qubits: tuple[int, int],
    pauli: str,
) -> torch.Tensor:
    from flagquantum.simulation.statevector.operations import _apply_matrix_layout

    cosine = torch.cos(angles / 2)
    sine = torch.sin(angles / 2)
    pauli_matrix = _pauli_matrix(pauli, device=state.device)
    identity = torch.eye(4, device=state.device, dtype=torch.complex64)
    matrices = (
        cosine[:, None, None] * identity - 1.0j * sine[:, None, None] * pauli_matrix
    )
    return _apply_matrix_layout(
        state,
        matrices,
        qubits,
        state.shape[1].bit_length() - 1,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("pauli", ("XX", "YY", "ZZ"))
@pytest.mark.parametrize("qubits", ((0, 1), (0, 11), (7, 2)))
def test_local_pauli_rotation_matches_matrix_reference(
    pauli: str,
    qubits: tuple[int, int],
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(261_011)
    state = torch.randn(
        3,
        1 << 12,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
    )
    angles = torch.tensor((0.17, -0.43, 1.21), device="cuda")

    actual = apply_complex64_local_pauli_rotation_2q(
        state,
        torch.complex(torch.cos(angles / 2), torch.sin(angles / 2)),
        qubits=qubits,
        pauli=pauli,
    )
    expected = _reference(state, angles, qubits=qubits, pauli=pauli)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_local_pauli_rotation_rejects_non_cuda_state() -> None:
    state = torch.randn(1, 16, dtype=torch.complex64)

    with pytest.raises(ValueError, match="contiguous CUDA complex64"):
        apply_complex64_local_pauli_rotation_2q(
            state,
            torch.ones(1, dtype=torch.complex64),
            qubits=(0, 1),
            pauli="XX",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_local_pauli_rotation_rejects_unsupported_contracts() -> None:
    state = torch.randn(2, 16, device="cuda", dtype=torch.complex64)

    with pytest.raises(ValueError, match="XX, YY, or ZZ"):
        apply_complex64_local_pauli_rotation_2q(
            state,
            torch.ones(1, device="cuda", dtype=torch.complex64),
            qubits=(0, 1),
            pauli="XY",
        )
    with pytest.raises(ValueError, match="two distinct"):
        apply_complex64_local_pauli_rotation_2q(
            state,
            torch.ones(1, device="cuda", dtype=torch.complex64),
            qubits=(1, 1),
            pauli="XX",
        )
    with pytest.raises(ValueError, match="one or batch values"):
        apply_complex64_local_pauli_rotation_2q(
            state,
            torch.ones(3, device="cuda", dtype=torch.complex64),
            qubits=(0, 1),
            pauli="XX",
        )
    with pytest.raises(ValueError, match="output must be distinct"):
        apply_complex64_local_pauli_rotation_2q(
            state,
            torch.ones(1, device="cuda", dtype=torch.complex64),
            qubits=(0, 1),
            pauli="XX",
            output=state,
        )
