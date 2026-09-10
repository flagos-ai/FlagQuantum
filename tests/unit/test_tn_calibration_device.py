"""Device identity contracts for tensor-network memory calibration."""

from unittest.mock import Mock

import pytest
import torch

from flagquantum.compute import PlatformDevice
from flagquantum.runtime.executors.tensor_network import execution
from flagquantum.runtime.planner.tn_calibration import build_tn_working_set_calibration
from flagquantum.simulation.tensor_network.models import (
    TensorNetworkNode,
    TensorNetworkSlicingPlan,
)


@pytest.mark.parametrize("calibrated", (False, True))
@pytest.mark.parametrize("device_available", (False, True))
def test_cuda_preflight_uses_discovered_device_only_for_calibration(
    monkeypatch: pytest.MonkeyPatch, calibrated: bool, device_available: bool
) -> None:
    reference = Mock(spec=torch.Tensor)
    reference.device = torch.device("cuda:1")
    reference.is_cuda = True
    reference.element_size.return_value = 8
    nodes = (TensorNetworkNode(reference, (0,)),)
    slicing = TensorNetworkSlicingPlan((), (), 1, 1, 1, 128, peak_bytes=1024)
    runtime = Mock()
    runtime.discover.return_value = (
        PlatformDevice("cuda", 0, "Other accelerator", "test"),
        PlatformDevice("cuda", 1, "Selected accelerator", "test", device_available),
    )
    provider = Mock(return_value=runtime)
    monkeypatch.setattr(execution, "get_platform_runtime", provider)
    monkeypatch.setattr(execution, "_zero_for_output", lambda *args: torch.zeros(1))
    calibration = (
        build_tn_working_set_calibration(
            (
                {
                    "predicted_working_set_bytes": size,
                    "cuda_peak_allocated_bytes": size,
                    "cuda_peak_reserved_bytes": size + 256,
                }
                for size in (1024, 2048, 4096)
            ),
            accelerator_name="Selected accelerator",
            complex_bytes=8,
            world_size=1,
            topology_class="test",
        )
        if calibrated
        else None
    )
    kwargs = dict(
        max_working_set_bytes=None,
        working_set_safety_factor=1.25,
        working_set_policy=None,
        memory_calibration=calibration,
        world_size=1,
    )
    if calibrated and not device_available:
        with pytest.raises(ValueError, match="calibration device is unavailable"):
            execution._sparse_working_set_preflight(slicing, nodes, (), **kwargs)
    else:
        summary = execution._sparse_working_set_preflight(slicing, nodes, (), **kwargs)
        assert summary["budget_satisfied"] is True
        assert summary["predicted_working_set_bytes"] >= 1024
    if calibrated:
        provider.assert_called_once_with("cuda")
        runtime.discover.assert_called_once_with()
    else:
        provider.assert_not_called()
