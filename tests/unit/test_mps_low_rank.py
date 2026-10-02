import pytest
import torch

import flagquantum.simulation.mps.two_site_dispatch as two_site_dispatch
from flagquantum.simulation.mps.low_rank import (
    fixed_rank_range_qr,
    fixed_rank_two_site_range_qr,
)

pytestmark = pytest.mark.unit


def test_fixed_rank_range_qr_is_deterministic_and_differentiable():
    matrix = torch.randn(2, 12, 10, dtype=torch.complex64, requires_grad=True)
    left, right = fixed_rank_range_qr(matrix, 4, power_iterations=1)
    second_left, second_right = fixed_rank_range_qr(matrix, 4, power_iterations=1)

    torch.testing.assert_close(left, second_left)
    torch.testing.assert_close(right, second_right)
    assert left.shape == (2, 12, 4)
    assert right.shape == (2, 4, 10)
    torch.real((left @ right).sum()).backward()
    assert matrix.grad is not None
    assert torch.isfinite(matrix.grad).all()


def test_power_iteration_improves_or_preserves_reconstruction_error():
    torch.manual_seed(5)
    matrix = torch.randn(1, 16, 14, dtype=torch.complex64)
    plain = fixed_rank_range_qr(matrix, 5, power_iterations=0)
    refined = fixed_rank_range_qr(matrix, 5, power_iterations=1)
    plain_error = torch.linalg.vector_norm(matrix - plain[0] @ plain[1])
    refined_error = torch.linalg.vector_norm(matrix - refined[0] @ refined[1])
    assert refined_error <= plain_error + 1e-5


def test_fixed_rank_two_site_range_qr_shapes_and_gradient():
    left = torch.randn(1, 4, 2, 5, dtype=torch.complex64, requires_grad=True)
    right = torch.randn(1, 5, 2, 3, dtype=torch.complex64, requires_grad=True)
    gate = torch.randn(1, 4, 4, dtype=torch.complex64, requires_grad=True)
    left_out, right_out = fixed_rank_two_site_range_qr(left, gate, right, 3)
    assert left_out.shape == (1, 4, 2, 3)
    assert right_out.shape == (1, 3, 2, 3)
    reconstruction = left_out.reshape(1, 8, 3) @ right_out.reshape(1, 3, 6)
    torch.real(reconstruction.sum()).backward()
    for value in (left, gate, right):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fixed_rank_two_site_opt_in_routes_inference_but_preserves_training_fallback(
    monkeypatch,
) -> None:
    catalog_routes = []
    require_cataloged_kernel = two_site_dispatch._require_mps_projected_two_site_kernel

    def capture_catalog_route(*, device_type, dtype):
        implementation = require_cataloged_kernel(
            device_type=device_type,
            dtype=dtype,
        )
        catalog_routes.append(implementation.implementation_id)
        return implementation

    monkeypatch.setattr(
        two_site_dispatch,
        "_require_mps_projected_two_site_kernel",
        capture_catalog_route,
    )
    left = torch.randn(2, 3, 2, 5, device="cuda", dtype=torch.complex64)
    gate = torch.randn(2, 4, 4, device="cuda", dtype=torch.complex64)
    right = torch.randn(2, 5, 2, 4, device="cuda", dtype=torch.complex64)

    default_left, default_right = fixed_rank_two_site_range_qr(left, gate, right, 4)

    assert default_left.shape == (2, 3, 2, 4)
    assert default_right.shape == (2, 4, 2, 4)
    assert catalog_routes == []

    monkeypatch.setenv("FQ_TRITON_MPS_PROJECTED_TWO_SITE", "1")
    left_out, right_out = fixed_rank_two_site_range_qr(left, gate, right, 4)

    assert left_out.shape == (2, 3, 2, 4)
    assert right_out.shape == (2, 4, 2, 4)
    assert catalog_routes == ["FQKI-TRITON-MPS-002-A"]

    training_inputs = tuple(
        value.detach().clone().requires_grad_(True) for value in (left, gate, right)
    )
    training_left, training_right = fixed_rank_two_site_range_qr(
        *training_inputs,
        4,
    )
    torch.real(
        (training_left.reshape(2, 6, 4) @ training_right.reshape(2, 4, 8)).sum()
    ).backward()

    assert catalog_routes == ["FQKI-TRITON-MPS-002-A"]
    assert all(value.grad is not None for value in training_inputs)
