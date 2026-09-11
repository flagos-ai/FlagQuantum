"""Identity energy offsets must preserve the selected numerical precision."""

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms import pauli_term
from flagquantum.simulation.mps.state import MPSState

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("representation", ["circuit", "mps", "statevector", "density"])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_identity_term_preserves_state_precision_and_coefficient_gradient(
    representation: str, dtype: torch.dtype
) -> None:
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    coefficient = torch.tensor(1.0 + 2**-40, dtype=real_dtype, requires_grad=True)
    target: fq.Circuit | MPSState | torch.Tensor
    if representation == "circuit":
        target = fq.Circuit(1, dtype=dtype)
    elif representation == "mps":
        target = MPSState.zero(1, dtype=dtype)
    elif representation == "density":
        target = torch.eye(2, dtype=dtype) / 2
    else:
        target = torch.tensor([1.0, 0.0], dtype=dtype)

    result = pauli_term(coefficient, "I", (0,)).expectation(target)
    assert result.dtype == real_dtype
    torch.testing.assert_close(result, coefficient.reshape(1), atol=0, rtol=0)
    result.sum().backward()
    torch.testing.assert_close(coefficient.grad, torch.ones_like(coefficient))
