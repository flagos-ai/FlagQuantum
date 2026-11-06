"""Verify source mixture order, complex operators, and storage wire mapping."""

import numpy as np
import pytest
import torch

from flagquantum.core.ir import CircuitIR
from flagquantum.qec.resources.distillation_channels import (
    noisy_pauli_rotation,
    storage_error,
)
from flagquantum.simulation.density_matrix import density_matrix_from_ir

pytestmark = pytest.mark.unit


def test_rotation_channel_executes_through_native_ir() -> None:
    op = noisy_pauli_rotation("X", 0, 0, 0, (0,))
    rho = density_matrix_from_ir(CircuitIR(1, (op,)).validate(), dtype=torch.complex128)
    expected = torch.tensor([np.cos(np.pi / 8), 1j * np.sin(np.pi / 8)])
    torch.testing.assert_close(rho[0], expected[:, None] @ expected.conj()[None, :])


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("word", ["ZIZZ", "YX", "I"])
@pytest.mark.parametrize("errors", [(0, 0, 0), (0.1, 0.2, 0.3)])
def test_rotation_branches_match_source(
    word: str, errors: tuple[float, float, float]
) -> None:
    cirq = pytest.importorskip("cirq")
    pytest.importorskip("qualtran")
    from qualtran.surface_code.t_factory_utils import NoisyPauliRotation

    op = noisy_pauli_rotation(word, *errors, tuple(range(len(word))))
    source = cirq.kraus(NoisyPauliRotation(word, *errors))
    np.testing.assert_allclose(np.asarray(torch.stack(op.matrix)), source, atol=1e-15)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("kind", ["X", "Z"])
def test_storage_channels_match_source(kind: str) -> None:
    cirq = pytest.importorskip("cirq")
    pytest.importorskip("qualtran")
    from qualtran.surface_code.t_factory_utils import storage_error as source_storage

    wires = (2, 0, 1)
    probabilities = (0.17, 0.0, 0.31)
    source = source_storage(
        kind, probabilities, tuple(cirq.LineQubit(i) for i in wires)
    )
    for op, upstream in zip(
        storage_error(kind, probabilities, wires), source, strict=True
    ):
        assert op.wires == tuple(q.x for q in upstream.qubits)
        # Cirq keeps both zero-probability Pauli slots in its asymmetric channel.
        nonzero = [matrix for matrix in cirq.kraus(upstream) if np.any(matrix)]
        actual = [matrix.numpy() for matrix in op.matrix if bool(torch.any(matrix))]
        np.testing.assert_allclose(actual, nonzero, atol=1e-15)


def test_rejects_invalid_channels() -> None:
    with pytest.raises(ValueError, match="Pauli word"):
        noisy_pauli_rotation("A", 0, 0, 0, (0,))
    with pytest.raises(ValueError, match="mixture"):
        noisy_pauli_rotation("Z", 0.5, 0.5, 0.5, (0,))
    with pytest.raises(ValueError, match="X or Z"):
        storage_error("Y", (0.1,), (0,))
