import datetime

import pytest

from flagquantum.runtime.executors.mps import transport


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
