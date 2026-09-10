import pytest
import torch

from flagquantum.simulation.mps.observables import (
    mps_heisenberg_local_scan,
    mps_z_zz_local_scan,
)

pytestmark = pytest.mark.unit


def _site(bit: int) -> torch.Tensor:
    tensor = torch.zeros((1, 1, 2, 1), dtype=torch.complex128)
    tensor[0, 0, bit, 0] = 1
    return tensor


def test_local_z_zz_scan_tracks_single_and_adjacent_channels() -> None:
    base = torch.ones((1, 1, 1), dtype=torch.complex128)
    outputs = mps_z_zz_local_scan(
        (base, torch.zeros_like(base), torch.zeros_like(base), torch.zeros_like(base)),
        (_site(0), _site(1)),
        (0, 1),
        {0: (0,)},
        {1: (1,)},
        compiled=False,
    )

    torch.testing.assert_close(outputs[2].real, torch.ones_like(base.real))
    torch.testing.assert_close(outputs[3].real, -torch.ones_like(base.real))


def test_local_heisenberg_scan_matches_product_state_energy() -> None:
    base = torch.ones((1, 1, 1), dtype=torch.complex128)
    zeros = tuple(torch.zeros_like(base) for _ in range(4))
    coefficients = (
        torch.tensor([1.0, 2.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([3.0]),
    )

    outputs = mps_heisenberg_local_scan(
        (base, *zeros),
        (_site(0), _site(0)),
        (0, 1),
        coefficients,
    )

    torch.testing.assert_close(
        outputs[-1].real,
        torch.full_like(base.real, 6.0),
    )


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_heisenberg_scan_preserves_bell_correlations_and_gradients(
    dtype: torch.dtype,
) -> None:
    left = torch.zeros((1, 1, 2, 2), dtype=dtype)
    right = torch.zeros((1, 2, 2, 1), dtype=dtype)
    left[0, 0, 0, 0] = left[0, 0, 1, 1] = 2**-0.5
    right[0, 0, 0, 0] = right[0, 1, 1, 0] = 1
    base = torch.ones((1, 1, 1), dtype=dtype)
    real_dtype = base.real.dtype
    field = torch.tensor([0.3, -0.7], dtype=real_dtype, requires_grad=True)
    x = torch.tensor([2.0], dtype=real_dtype, requires_grad=True)
    y = torch.tensor([-3.0], dtype=real_dtype, requires_grad=True)
    z = torch.tensor([4.0], dtype=real_dtype, requires_grad=True)

    outputs = mps_heisenberg_local_scan(
        (base, *(torch.zeros_like(base) for _ in range(4))),
        (left, right),
        (0, 1),
        (field, x, y, z),
    )
    energy = outputs[-1].real.sum()
    energy.backward()

    torch.testing.assert_close(energy, torch.tensor(9.0, dtype=real_dtype))
    torch.testing.assert_close(field.grad, torch.zeros_like(field))
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    torch.testing.assert_close(y.grad, -torch.ones_like(y))
    torch.testing.assert_close(z.grad, torch.ones_like(z))
