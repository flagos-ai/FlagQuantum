"""Parameter synchronization on Torch builds with either tensor gather API."""

import pytest
import torch
import torch.distributed as dist

from flagquantum.runtime.executors.mps.training_engine import (
    _broadcast_parameters,
    _parameter_broadcast_buckets,
)
from flagquantum.runtime.executors.tensor_network.distributed_optimizer import (
    execute_rank_owned_tn_sgd_step,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("modern_api", [True, False])
@pytest.mark.parametrize("path", ["mps", "tensor_network"])
def test_parameter_sync_uses_available_tensor_gather(
    monkeypatch: pytest.MonkeyPatch, modern_api: bool, path: str
) -> None:
    calls: list[object] = []
    group = object()

    def gather(
        output: torch.Tensor, local: torch.Tensor, *, group: object = None
    ) -> None:
        calls.append(group)
        torch.testing.assert_close(local, torch.tensor([0.9, 1.8]))
        output.copy_(torch.tensor([0.9, 1.8, 3.0, 4.0]))

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the unavailable or nonpreferred gather was called")

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda **kwargs: 2)
    monkeypatch.setattr(dist, "get_rank", lambda **kwargs: 0)
    monkeypatch.setattr(
        dist, "all_gather_into_tensor", forbidden if modern_api else gather
    )
    if modern_api:
        monkeypatch.setattr(dist, "all_gather_single", gather, raising=False)
    else:
        monkeypatch.delattr(dist, "all_gather_single", raising=False)
    parameters = (torch.tensor([0.9, 1.8]), torch.tensor([0.0, 0.0]))
    if path == "mps":
        buckets = _parameter_broadcast_buckets(
            parameters, (0, 1), max_bucket_bytes=1024, world_size=2
        )
        count, byte_count = _broadcast_parameters(parameters, buckets)
        assert count == 1
        assert byte_count == 16
        assert calls == [None]
    else:
        parameters[0].copy_(torch.tensor([1.0, 2.0]))
        result = execute_rank_owned_tn_sgd_step(
            parameters,
            (torch.tensor([1.0, 2.0]), torch.zeros(2)),
            learning_rate=0.1,
            gradient_reduction="owner_reduce",
            process_group=group,
        )
        assert calls == [group]
        assert result.owned_parameter_indices == (0,)
        assert result.gradient_distribution_semantics == "sharded_across_ranks"
    torch.testing.assert_close(parameters[0], torch.tensor([0.9, 1.8]))
    torch.testing.assert_close(parameters[1], torch.tensor([3.0, 4.0]))
