"""Close a classical feedback loop through the offline realtime channel.

The plant is one binary state. Five noisy observations of that same state go to
the existing majority-vote handler; the returned bit commands an XOR action that
tries to restore zero. At most two observation errors per round are correctable.
This is a classical control model, not a quantum circuit or a QEC decoder.

Each shot initializes its own plant state. The handler is stateless: opening a
channel does not reset a decoder, and a request id does not identify a shot or
provide deduplication. No clock, deadline, transport I/O or hardware is involved.
"""

from __future__ import annotations

import flagquantum.realtime as rt
from examples.realtime.device_call import majority_decode

SCHEMA = rt.HandlerSchema(
    args=(rt.TypeDescriptor(rt.PayloadType.BIT_PACKED, 1, 5),),
    results=(rt.TypeDescriptor(rt.PayloadType.UINT8, 1),),
)
HANDLER_ID = rt.function_id("majority_decode")


def feedback_step(
    realtime: rt.RealtimeSession,
    state: int,
    *,
    request_id: int,
    readout_errors: tuple[int, ...] = (0, 0, 0, 0, 0),
) -> int:
    """Observe the plant and apply one validated feedback response.

    Returns the resulting plant state. A failed call, mismatched response or
    non-binary command raises before the caller can commit an updated state.
    """
    if type(state) is not int or state not in (0, 1):
        raise ValueError("the plant state must be a binary integer")
    if len(readout_errors) != 5 or any(
        type(bit) is not int or bit not in (0, 1) for bit in readout_errors
    ):
        raise ValueError("readout errors must contain five binary integers")
    samples = tuple(state ^ error for error in readout_errors)
    outcome = realtime.call(
        0, HANDLER_ID, rt.encode_arguments(SCHEMA, (samples,)), request_id=request_id
    )
    if not outcome.succeeded:
        raise rt.RealtimeCallError(
            f"feedback request {request_id} failed with status {outcome.response.status}"
        )
    (command,) = rt.decode_results(SCHEMA, outcome.result)
    if type(command) is not int or command not in (0, 1):
        raise rt.RealtimeCallError("the feedback command must be a binary integer")
    return state ^ command


def run_shot(
    realtime: rt.RealtimeSession,
    *,
    initial_state: int,
    first_request_id: int,
) -> tuple[int, ...]:
    """Run three feedback rounds with explicit shot-local plant state."""
    if type(initial_state) is not int or initial_state not in (0, 1):
        raise ValueError("the initial plant state must be a binary integer")
    if type(first_request_id) is not int or not 0 <= first_request_id <= 2**32 - 3:
        raise ValueError("the first request id must leave room for three rounds")
    state = initial_state
    history = []
    for offset, disturbance in enumerate((1, 0, 1)):
        state ^= disturbance
        state = feedback_step(
            realtime,
            state,
            request_id=first_request_id + offset,
            readout_errors=(1, 0, 0, 1, 0),
        )
        history.append(state)
    return tuple(history)


def main() -> None:
    service = rt.FunctionTableService(
        [rt.FunctionEntry("majority_decode", SCHEMA, majority_decode)]
    )
    handler_session = service.open_session("host_call")
    assert handler_session is not None
    driver = rt.DeviceCallChannelDriver(
        {"feedback": rt.loopback_channel("feedback", handler_session)}
    )
    try:
        with rt.RealtimeSession(driver) as realtime:
            channel = realtime.open_channel(0, "feedback")
            for shot, initial_state in enumerate((0, 1)):
                states = run_shot(
                    realtime,
                    initial_state=initial_state,
                    first_request_id=shot * 3,
                )
                assert states == (0, 0, 0), states
                print(
                    f"shot {shot}: initial={initial_state}, corrected states={states}"
                )
        assert not channel.is_open
    finally:
        handler_session.stop()
    print("FlagQuantum offline feedback loop check passed")


if __name__ == "__main__":
    main()
