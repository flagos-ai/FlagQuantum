"""Application feedback, failure commitment and lifetime over offline messaging."""

from __future__ import annotations

import itertools
import statistics
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from examples.realtime.device_call import majority_decode
from examples.realtime.feedback_loop import SCHEMA, feedback_step, run_shot
from flagquantum import realtime as rt

pytestmark = pytest.mark.integration


@contextmanager
def _open_feedback(
    handler: Callable[[tuple[object, ...]], object] = majority_decode,
) -> Iterator[rt.RealtimeSession]:
    service = rt.FunctionTableService(
        [rt.FunctionEntry("majority_decode", SCHEMA, handler)]
    )
    session = service.open_session("host_call")
    assert session is not None
    driver = rt.DeviceCallChannelDriver(
        {"feedback": rt.loopback_channel("feedback", session)}
    )
    try:
        with rt.RealtimeSession(driver) as realtime:
            realtime.open_channel(0, "feedback")
            yield realtime
    finally:
        session.stop()


def test_feedback_example_runs_without_installed_numerical_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, "-S", "-m", "examples.realtime.feedback_loop"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert "shot 0: initial=0, corrected states=(0, 0, 0)" in result.stdout
    assert "shot 1: initial=1, corrected states=(0, 0, 0)" in result.stdout
    assert "FlagQuantum offline feedback loop check passed" in result.stdout


def test_both_states_and_every_readout_error_pattern_have_an_explicit_limit() -> None:
    with _open_feedback() as realtime:
        for state in (0, 1):
            for request_id, errors in enumerate(itertools.product((0, 1), repeat=5)):
                corrected = feedback_step(
                    realtime, state, request_id=request_id, readout_errors=errors
                )
                # Three wrong observations reverse the vote: the model exposes
                # its failure boundary instead of claiming arbitrary correction.
                assert corrected == (0 if sum(errors) <= 2 else 1)


def test_shots_initialize_their_own_state_on_the_same_open_channel() -> None:
    with _open_feedback() as realtime:
        channel = realtime.channel(0)
        for initial_state in (1, 0, 1):
            assert run_shot(
                realtime, initial_state=initial_state, first_request_id=0
            ) == (0, 0, 0)
            assert realtime.channel(0) is channel


def test_feedback_action_comes_from_the_response() -> None:
    with _open_feedback(lambda arguments: (0,)) as realtime:
        assert run_shot(realtime, initial_state=1, first_request_id=0) == (0, 0, 1)
        assert feedback_step(realtime, 1, request_id=3) == 1


def test_another_handler_replaces_the_vote_without_changing_the_consumer() -> None:
    def median_readout(arguments: tuple[object, ...]) -> tuple[int]:
        samples = arguments[0]
        assert isinstance(samples, tuple)
        return (int(statistics.median(samples)),)

    for handler in (majority_decode, median_readout):
        with _open_feedback(handler) as realtime:
            assert run_shot(realtime, initial_state=1, first_request_id=17) == (0, 0, 0)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("declared", "failed with status 7"),
        ("exception", "failed with status -3"),
        ("non_binary", "command must be a binary integer"),
    ],
)
def test_a_failed_response_does_not_commit_feedback_and_closes_the_channel(
    failure: str,
    message: str,
) -> None:
    def refuse(arguments: tuple[object, ...]) -> tuple[int]:
        if failure == "declared":
            raise rt.RealtimeHandlerError(7, "readout unavailable")
        if failure == "exception":
            raise RuntimeError("handler failed")
        return (2,)

    state = 1
    with (
        pytest.raises(rt.RealtimeCallError, match=message),
        _open_feedback(refuse) as realtime,
    ):
        channel = realtime.channel(0)
        state = feedback_step(realtime, state, request_id=17)
    assert state == 1
    assert not channel.is_open
    assert realtime.device_ids == ()
    with pytest.raises(rt.RealtimeCallError, match="session is closed"):
        feedback_step(realtime, state, request_id=18)


def test_a_stale_response_cannot_commit_feedback() -> None:
    class StaleResponse(rt.LoopbackTransport):
        def await_response(self) -> bytes:
            response, result = rt.read_response(super().await_response())
            return rt.frame_response(
                rt.RPCResponse(
                    response.status,
                    response.result_len,
                    response.request_id + 1,
                    response.ptp_timestamp,
                ),
                result,
                slot_size=rt.DEFAULT_SLOT_SIZE,
            )

    service = rt.FunctionTableService(
        [rt.FunctionEntry("majority_decode", SCHEMA, majority_decode)]
    )
    session = service.open_session("host_call")
    assert session is not None
    transport = StaleResponse(session, slot_size=rt.DEFAULT_SLOT_SIZE)
    driver = rt.DeviceCallChannelDriver(
        {
            "stale": lambda: rt.DeviceCallChannel(
                "stale", transport, slot_size=rt.DEFAULT_SLOT_SIZE
            )
        }
    )
    state = 1
    try:
        with (
            pytest.raises(rt.RealtimeProtocolError, match="response echoes"),
            rt.RealtimeSession(driver) as realtime,
        ):
            channel = realtime.open_channel(0, "stale")
            state = feedback_step(realtime, state, request_id=17)
        assert state == 1
        assert not channel.is_open
    finally:
        session.stop()


def test_request_ids_correlate_responses_without_deduplicating_actions() -> None:
    samples_seen = []

    def record(arguments: tuple[object, ...]) -> tuple[int]:
        samples_seen.append(arguments[0])
        return majority_decode(arguments)

    with _open_feedback(record) as realtime:
        assert feedback_step(realtime, 0, request_id=17) == 0
        assert feedback_step(realtime, 1, request_id=17) == 0
    assert samples_seen == [(0, 0, 0, 0, 0), (1, 1, 1, 1, 1)]


@pytest.mark.parametrize("state", [-1, 2, True])
def test_invalid_plant_state_is_refused(state: int) -> None:
    with _open_feedback() as realtime, pytest.raises(ValueError, match="plant state"):
        feedback_step(realtime, state, request_id=0)


@pytest.mark.parametrize("errors", [(0,), (0, 0, 0, 0, 2), (0, 0, 0, 0, True)])
def test_invalid_observations_are_refused(errors: tuple[int, ...]) -> None:
    with (
        _open_feedback() as realtime,
        pytest.raises(ValueError, match="five binary integers"),
    ):
        feedback_step(realtime, 0, request_id=0, readout_errors=errors)


@pytest.mark.parametrize(
    ("state", "first_request_id"),
    [(2, 0), (True, 0), (0, -1), (0, True), (0, 2**32 - 2)],
)
def test_invalid_shot_initialization_is_refused(
    state: int, first_request_id: int
) -> None:
    with _open_feedback() as realtime, pytest.raises(ValueError):
        run_shot(realtime, initial_state=state, first_request_id=first_request_id)
