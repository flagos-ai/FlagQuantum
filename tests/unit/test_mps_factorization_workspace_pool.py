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
