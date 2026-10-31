"""Seeded measurements retain the sampling tensor's full device identity."""

from types import SimpleNamespace

import pytest
import torch

from flagquantum.runtime.measurements import _generator
from flagquantum.simulation.tensor_network.models import (
    TensorNetworkContractionPlan,
    TensorNetworkNode,
)
from flagquantum.simulation.tensor_network.state import TensorNetworkState

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("tensor_network", [False, True])
def test_generator_uses_sampling_device_and_index(monkeypatch, tensor_network):
    device = torch.device("cuda:3")
    if tensor_network:
        node = TensorNetworkNode(tensor=SimpleNamespace(device=device), labels=(0,))
        target = TensorNetworkState(
            TensorNetworkContractionPlan(
                n_qubits=1,
                bsz=1,
                nodes=(node,),
                output_labels=(0,),
                path=(),
            )
        )
    else:
        target = SimpleNamespace(device=device)
    selected = []
    seeded = []

    def create_generator(*, device):
        selected.append(device)
        return SimpleNamespace(manual_seed=seeded.append)

    monkeypatch.setattr(torch, "Generator", create_generator)
    assert _generator(target, {"seed": 13}) is not None
    assert selected == [device]
    assert seeded == [13]


def test_unseeded_measurement_keeps_default_rng_policy():
    assert _generator(SimpleNamespace(device="cpu"), {}) is None
