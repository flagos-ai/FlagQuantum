"""Asynchronous MPS shape and payload transfers without a process group."""

from unittest.mock import Mock

import pytest
import torch

from flagquantum.runtime.executors.mps import transport

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("missing_index", [0, 1])
def test_async_send_rejects_missing_request(
    monkeypatch: pytest.MonkeyPatch, missing_index: int
) -> None:
    requests: list[Mock | None] = [Mock(), Mock()]
    requests[missing_index] = None
    monkeypatch.setattr(transport.dist, "isend", Mock(side_effect=requests))
    with pytest.raises(RuntimeError, match="MPS asynchronous send.*peer 2"):
        transport._send_tensor_async_p2p(torch.ones(2, 3), dst=2)


@pytest.mark.parametrize("missing_phase", ["shape", "payload"])
def test_async_receive_rejects_missing_request(
    monkeypatch: pytest.MonkeyPatch, missing_phase: str
) -> None:
    def receive(tensor: torch.Tensor, *, src: int) -> Mock | None:
        phase = "shape" if tensor.dtype == torch.int64 else "payload"
        if phase == missing_phase:
            return None
        tensor.copy_(torch.tensor([2, 3]))
        return Mock()

    monkeypatch.setattr(transport.dist, "irecv", receive)
    with pytest.raises(
        RuntimeError, match=f"MPS asynchronous {missing_phase} receive.*peer 2"
    ):
        transport._recv_tensor_async_p2p(src=2, reference=torch.empty(2, 3))


def test_async_transport_preserves_shape_payload_and_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = torch.arange(6.0).reshape(3, 2).t()
    payloads: list[torch.Tensor] = []
    requests: list[Mock] = []

    def send(tensor: torch.Tensor, *, dst: int) -> Mock:
        assert dst == 2
        payloads.append(tensor.clone())
        request = Mock()
        requests.append(request)
        return request

    def receive(tensor: torch.Tensor, *, src: int) -> Mock:
        assert src == 2
        tensor.copy_(payloads.pop(0))
        request = Mock()
        requests.append(request)
        return request

    monkeypatch.setattr(transport.dist, "isend", send)
    monkeypatch.setattr(transport.dist, "irecv", receive)
    transport._send_tensor_async_p2p(expected, dst=2)
    actual = transport._recv_tensor_async_p2p(src=2, reference=expected)
    torch.testing.assert_close(actual, expected)
    assert not payloads
    assert len(requests) == 4
    for request in requests:
        request.wait.assert_called_once_with()
