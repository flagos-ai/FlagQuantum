import torch

from flagquantum.runtime.executors.mps import transport as mps_transport


def test_packed_sender_reuses_payload_buffer_without_changing_values(monkeypatch):
    mps_transport._MPS_P2P_BUFFER_POOL.clear()
    mps_transport.reset_mps_p2p_stats()
    payloads = []

    monkeypatch.setattr(
        mps_transport.dist,
        "P2POp",
        lambda _operation, tensor, *_args, **_kwargs: tensor,
    )

    def capture(operations, _device, *, diagnostic):
        assert "batch_send" in diagnostic
        payloads.append(operations[1].clone())

    monkeypatch.setattr(mps_transport, "_run_batched_p2p", capture)
    values = (torch.arange(6.0).reshape(2, 3), torch.arange(4.0))

    mps_transport._send_tensor_batch_p2p(values, dst=1, sequences=(10, 11))
    first_pointer = mps_transport._MPS_P2P_BUFFER_POOL[
        ("cpu", None, torch.float32, 10)
    ].data_ptr()
    mps_transport._send_tensor_batch_p2p(values, dst=1, sequences=(12, 13))

    expected = torch.cat((torch.arange(6.0), torch.arange(4.0)))
    assert all(torch.equal(payload, expected) for payload in payloads)
    assert (
        mps_transport._MPS_P2P_BUFFER_POOL[("cpu", None, torch.float32, 10)].data_ptr()
        == first_pointer
    )
    stats = mps_transport.mps_p2p_stats()
    assert stats["buffer_pool_misses"] == 1
    assert stats["buffer_pool_hits"] == 1
