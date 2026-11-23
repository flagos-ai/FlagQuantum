"""Fixed-order complex trace values and gradients across reduction boundaries."""

import pytest
import torch

from flagquantum.simulation.density_reduction import _pairwise_complex_trace

pytestmark = pytest.mark.unit
np = pytest.importorskip("numpy")


@pytest.mark.parametrize(
    "size", (0, 1, 2, 3, 4, 5, 7, 16, 31, 32, 33, 63, 64, 65, 127, 129)
)
@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_complete_complex_trace_matches_pairwise_reference(
    size: int, dtype: torch.dtype
) -> None:
    generator = torch.Generator().manual_seed(286)
    matrix = torch.randn(3, size, size, generator=generator, dtype=dtype)
    # Distinct binary scales expose reassociation beyond ordinary allclose tests.
    scales = torch.logspace(-8, 8, size, dtype=matrix.real.dtype)
    matrix = (matrix * scales).requires_grad_()
    actual = _pairwise_complex_trace(matrix)
    expected = np.trace(matrix.detach().numpy(), axis1=-2, axis2=-1)
    np.testing.assert_array_equal(actual.detach().numpy(), expected)
    actual_gradient = torch.autograd.grad(
        actual.real.sum() + 2 * actual.imag.sum(), matrix
    )[0]
    diagonal = torch.eye(size, dtype=dtype).expand(3, -1, -1)
    torch.testing.assert_close(actual_gradient, diagonal * (1 + 2j), atol=0, rtol=0)
