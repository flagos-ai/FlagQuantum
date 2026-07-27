import pytest
import torch

from flagquantum.runtime.distributed import engine as distributed

pytestmark = pytest.mark.unit


def test_static_sender_rejects_shape_mismatch_before_transport(monkeypatch):
    monkeypatch.setattr(
        distributed,
        "_run_batched_p2p",
        lambda *args, **kwargs: pytest.fail("transport ran"),
    )
    with pytest.raises(RuntimeError, match="shape mismatch before send"):
        distributed._send_tensor_static_p2p(
            torch.zeros(2), dst=1, sequence=7, expected_shape=(3,), shape_generation=1
        )


def test_descriptor_cache_clear_is_explicit():
    distributed._MPS_STATIC_DESCRIPTOR_CACHE[(1, 2, 0)] = (3, (4,), torch.float32)
    assert distributed.mps_static_descriptor_cache_entries() == 1
    distributed.clear_mps_static_descriptor_cache()
    assert distributed.mps_static_descriptor_cache_entries() == 0
