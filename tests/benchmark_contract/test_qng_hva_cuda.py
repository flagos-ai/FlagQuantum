"""Real CUDA QNG oracle check, separate from optional CPU comparison imports."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.integration]


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="actual CUDA/Triton route required"
)
def test_real_fused_tangents_metric_and_update_match_oracle() -> None:
    pytest.importorskip("triton")
    # NumPy is a required oracle prerequisite on the prepared GPU runner.
    # A missing oracle dependency must fail this lane rather than skip it.
    from benchmarks.qng_hva_ablation import verify
    from benchmarks.qng_hva_geometry import Geometry, Workload

    workload = Workload(4, 2, "complex64", "cuda")
    parameters = (
        (
            0.07
            * torch.randn(
                workload.count,
                generator=torch.Generator().manual_seed(29),
                dtype=torch.float64,
            )
        )
        .float()
        .cuda()
        .requires_grad_()
    )
    assert verify(Geometry("fused_forward", workload), parameters, 0.001, 0.005)[
        "passed"
    ]
