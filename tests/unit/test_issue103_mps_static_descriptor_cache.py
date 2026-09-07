import pytest
import torch

from flagquantum.runtime.backends.mps import transport as mps_transport

pytestmark = pytest.mark.unit


def test_static_sender_rejects_shape_mismatch_before_transport(monkeypatch):
    monkeypatch.setattr(
        mps_transport,
        "_run_batched_p2p",
        lambda *args, **kwargs: pytest.fail("transport ran"),
    )
    with pytest.raises(RuntimeError, match="shape mismatch before send"):
        mps_transport._send_tensor_static_p2p(
            torch.zeros(2), dst=1, sequence=7, expected_shape=(3,), shape_generation=1
        )


def test_descriptor_cache_clear_is_explicit():
    mps_transport._MPS_STATIC_DESCRIPTOR_CACHE[(1, 2, 0)] = (
        3,
        (4,),
        torch.float32,
    )
    assert mps_transport.mps_static_descriptor_cache_entries() == 1
    mps_transport.clear_mps_static_descriptor_cache()
    assert mps_transport.mps_static_descriptor_cache_entries() == 0


def test_static_receiver_receives_cold_descriptor_before_payload(monkeypatch):
    mps_transport.clear_mps_static_descriptor_cache()
    dispatches = []

    monkeypatch.setattr(
        mps_transport.dist,
        "P2POp",
        lambda _operation, tensor, *_args, **_kwargs: tensor,
    )

    def run_batched(operations, _device, *, diagnostic):
        dispatches.append((operations, diagnostic))
        if "descriptor" in diagnostic:
            operations[0].copy_(torch.tensor([7, 2, 3]))
        else:
            operations[0].copy_(torch.arange(6, dtype=torch.float32))

    monkeypatch.setattr(mps_transport, "_run_batched_p2p", run_batched)

    received = mps_transport._recv_tensor_static_p2p(
        src=1,
        reference=torch.zeros(2, 3),
        sequence=7,
        expected_shape=(2, 3),
        shape_generation=11,
    )

    assert len(dispatches) == 2
    assert all(len(operations) == 1 for operations, _ in dispatches)
    assert "static_receive_cold_descriptor" in dispatches[0][1]
    assert "static_receive_cold_payload" in dispatches[1][1]
    assert torch.equal(received, torch.arange(6, dtype=torch.float32).reshape(2, 3))


def test_static_receiver_drains_payload_before_descriptor_failure(monkeypatch):
    mps_transport.clear_mps_static_descriptor_cache()
    dispatches = []

    monkeypatch.setattr(
        mps_transport.dist,
        "P2POp",
        lambda _operation, tensor, *_args, **_kwargs: tensor,
    )

    def run_batched(operations, _device, *, diagnostic):
        dispatches.append((operations, diagnostic))
        if "descriptor" in diagnostic:
            operations[0].copy_(torch.tensor([8, 2, 3]))
        else:
            operations[0].fill_(1)

    monkeypatch.setattr(mps_transport, "_run_batched_p2p", run_batched)

    with pytest.raises(RuntimeError, match="payload drained before failure"):
        mps_transport._recv_tensor_static_p2p(
            src=1,
            reference=torch.zeros(2, 3),
            sequence=7,
            expected_shape=(2, 3),
            shape_generation=11,
        )

    assert len(dispatches) == 2
    assert torch.equal(dispatches[1][0][0], torch.ones(6))


def test_static_receiver_drains_actual_payload_size_before_shape_failure(monkeypatch):
    mps_transport.clear_mps_static_descriptor_cache()
    payload_sizes = []

    monkeypatch.setattr(
        mps_transport.dist,
        "P2POp",
        lambda _operation, tensor, *_args, **_kwargs: tensor,
    )

    def run_batched(operations, _device, *, diagnostic):
        if "descriptor" in diagnostic:
            operations[0].copy_(torch.tensor([7, 5]))
        else:
            payload_sizes.append(operations[0].numel())
            operations[0].copy_(torch.arange(5, dtype=torch.float32))

    monkeypatch.setattr(mps_transport, "_run_batched_p2p", run_batched)

    with pytest.raises(
        RuntimeError,
        match=r"expected sequence 7 and shape \(3,\).*received sequence 7 and shape \(5,\)",
    ):
        mps_transport._recv_tensor_static_p2p(
            src=1,
            reference=torch.zeros(3),
            sequence=7,
            expected_shape=(3,),
            shape_generation=11,
        )

    assert payload_sizes == [5]
