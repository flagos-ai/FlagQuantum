from __future__ import annotations

import pytest
import torch

import flagquantum.training as fqt
from flagquantum.models import HybridQuantumClassifier


@pytest.mark.gpu
@pytest.mark.distributed_accel
def test_same_classifier_runs_single_gpu_training() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    fqt.seed_everything(481)
    model = HybridQuantumClassifier().to("cuda")
    inputs = torch.tensor([[-0.8, -0.2], [0.7, 0.4]], device="cuda")
    targets = torch.tensor([-1.0, 1.0], device="cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    losses = []
    for _ in range(2):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    assert all(torch.isfinite(torch.tensor(losses)))
    assert next(model.parameters()).is_cuda
