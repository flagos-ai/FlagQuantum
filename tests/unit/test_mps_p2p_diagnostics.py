import datetime
from contextlib import nullcontext

import pytest
import torch

from flagquantum.runtime.executors.mps import transport


@pytest.mark.parametrize("fail_work", [False, True])
def test_overlap_joins_communication_after_callback(
    monkeypatch: pytest.MonkeyPatch, fail_work: bool
) -> None:
    events: list[str] = []

    class Stream:
        def __init__(self, name: str) -> None:
            self.name = name

        def wait_stream(self, other: object) -> None:
            events.append(f"{self.name}:join")

    class Work:
        def wait(self, *, timeout: datetime.timedelta) -> bool:
            events.append("wait")
            return True

    p2p_stream = Stream("p2p")
    current_stream = Stream("current")
    monkeypatch.setattr(transport, "_MPS_P2P_STREAMS", {("cuda", 0): p2p_stream})
    monkeypatch.setattr(transport, "_MPS_P2P_STATS", transport.mps_p2p_stats())
    monkeypatch.setattr(
        transport.torch.cuda, "current_stream", lambda device: current_stream
    )
    monkeypatch.setattr(transport.torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(
        transport.dist, "batch_isend_irecv", lambda operations: [Work(), Work()]
    )

    def local_work() -> None:
        events.append("work")
        if fail_work:
            raise ValueError("local computation failed")

    expected_error = pytest.raises(ValueError, match="local computation failed")
    with expected_error if fail_work else nullcontext():
        transport._run_batched_p2p_with_overlap(
            [],
            torch.device("cuda:0"),
            diagnostic="test overlap",
            overlap_work=local_work,
        )
    assert events == ["p2p:join", "work", "wait", "wait", "current:join"]


@pytest.mark.parametrize("seconds", ["nan", "inf", "-inf", "0", "-1"])
def test_mps_p2p_rejects_invalid_timeout(
    monkeypatch: pytest.MonkeyPatch, seconds: str
) -> None:
    monkeypatch.setenv("FLAGQUANTUM_MPS_P2P_TIMEOUT_SECONDS", seconds)
    with pytest.raises(ValueError, match="must be positive and finite"):
        transport._p2p_timeout()
    with pytest.raises(ValueError, match="must be positive and finite"):
        transport._run_batched_p2p_with_overlap(
            [],
            torch.device("cuda:0"),
            diagnostic="invalid configuration",
            overlap_work=lambda: pytest.fail("invalid timeout must stop before work"),
        )


class _TimedOutWork:
    def wait(self, *, timeout):
        assert timeout == datetime.timedelta(seconds=0.25)
        return False


def test_mps_p2p_timeout_contains_actionable_context(monkeypatch):
    monkeypatch.setenv("FLAGQUANTUM_MPS_P2P_TIMEOUT_SECONDS", "0.25")
    monkeypatch.setattr(
        transport.dist, "batch_isend_irecv", lambda operations: [_TimedOutWork()]
    )
    monkeypatch.setattr(transport.dist, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError) as caught:
        transport._wait_batched_p2p(
            [object()],
            diagnostic="direction=receive,peer=3,sequence=19,tensor_shape=(2, 4)",
        )

    message = str(caught.value)
    assert "MPS P2P timeout" in message
    assert "peer=3" in message
    assert "sequence=19" in message
    assert "tensor_shape=(2, 4)" in message
    assert "process_group=uninitialized" in message
