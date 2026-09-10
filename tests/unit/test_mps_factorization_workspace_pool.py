from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch

from flagquantum.runtime.executors.mps.factorization import (
    FactorizationWorkspacePool,
    MPSFactorizationMemoryError,
)


def test_factorization_workspace_pool_reuses_released_shape():
    pool = FactorizationWorkspacePool(maximum_bytes=1024)
    first = pool.acquire(
        role="left",
        shape=(2, 4),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    pointer = first.tensor.data_ptr()
    pool.release(first)
    second = pool.acquire(
        role="left",
        shape=(2, 4),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert second.tensor.data_ptr() == pointer
    assert pool.stats() == {
        "allocation_count": 1,
        "reuse_count": 1,
        "reserved_bytes": 32,
        "entry_count": 1,
        "maximum_bytes": 1024,
    }


def test_factorization_workspace_pool_fails_closed_at_bound():
    pool = FactorizationWorkspacePool(maximum_bytes=31)

    with pytest.raises(MPSFactorizationMemoryError, match="pool exhausted"):
        pool.acquire(
            role="left",
            shape=(2, 4),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )


def test_factorization_workspace_pool_rejects_double_release():
    pool = FactorizationWorkspacePool(maximum_bytes=1024)
    entry = pool.acquire(
        role="left",
        shape=(1,),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    pool.release(entry)

    with pytest.raises(RuntimeError, match="already released"):
        pool.release(entry)


def test_cuda_workspace_records_completion_before_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:1")
    tensor = Mock(device=device)
    tensor.numel.return_value = 8
    tensor.element_size.return_value = 4
    stream = Mock()
    event = Mock()
    create_event = Mock(return_value=event)
    monkeypatch.setattr(torch, "empty", Mock(return_value=tensor))
    monkeypatch.setattr(torch.cuda, "Event", create_event)
    monkeypatch.setattr(torch.cuda, "device", Mock(return_value=nullcontext()))
    monkeypatch.setattr(torch.cuda, "current_stream", Mock(return_value=stream))
    pool = FactorizationWorkspacePool(maximum_bytes=1024)
    entry = pool.acquire(role="left", shape=(2, 4), dtype=torch.float32, device=device)

    def record(active_stream: object) -> None:
        assert active_stream is stream
        assert entry.in_use

    event.record.side_effect = record
    pool.release(entry)
    assert not entry.in_use
    create_event.assert_called_once_with()
    event.record.assert_called_once_with(stream)

    reused = pool.acquire(role="left", shape=(2, 4), dtype=torch.float32, device=device)
    assert reused is entry
    assert reused.in_use
    stream.wait_event.assert_called_once_with(event)
    assert pool.stats()["allocation_count"] == 1
