"""MPS site transport preserves complex values on real-only collectives."""

from dataclasses import replace

import pytest
import torch

from flagquantum.runtime.distributed.context import TorchDistributedContext
from flagquantum.runtime.executors.mps import distributed_state

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("conjugate", [False, True])
def test_site_broadcast_round_trip_on_real_only_transport(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, conjugate: bool
) -> None:
    source = (
        torch.complex(
            torch.arange(12, dtype=torch.float64).reshape(1, 2, 2, 3),
            torch.ones(1, 2, 2, 3, dtype=torch.float64),
        )
        .to(dtype)
        .transpose(2, 3)
    )
    if conjugate:
        source = source.conj()
    payloads: list[torch.Tensor] = []
    receiving = False

    def broadcast(value: torch.Tensor, *, src: int) -> None:
        assert src == 0
        assert not value.is_complex(), "transport does not support complex scalars"
        if receiving:
            payload = payloads.pop(0)
            assert value.dtype == payload.dtype
            assert value.shape == payload.shape
            value.copy_(payload)
        else:
            payloads.append(value.clone())

    monkeypatch.setattr(distributed_state.dist, "broadcast", broadcast)
    context = TorchDistributedContext(
        rank=0,
        world_size=2,
        local_rank=0,
        backend="gloo",
        device=torch.device("cpu"),
        initialized=True,
    )
    sent = distributed_state._broadcast_mps_site_tensor(
        source, src=0, context=context, wire=0
    )
    receiving = True
    received = distributed_state._broadcast_mps_site_tensor(
        None, src=0, context=replace(context, rank=1, local_rank=1), wire=0
    )
    assert not payloads
    assert received.dtype == dtype
    torch.testing.assert_close(sent, source, rtol=0, atol=0)
    torch.testing.assert_close(received, source, rtol=0, atol=0)
