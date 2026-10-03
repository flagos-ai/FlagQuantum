"""Carry a realtime device call from a running program to a classical handler.

This example is the ten-minute path for the realtime messaging protocol: declare
what a handler accepts and returns, register a channel, open a session for a
device id, and send one call. It then shows the two ways the exchange refuses
rather than guesses -- a payload that does not match the declared schema, and a
device id nobody opened.

It performs no I/O, needs no hardware, and contacts nothing, so it is safe to
run and to read. The transport here is an in-process stand-in; a deployment
replaces it with a channel that talks to an FPGA or a NIC, and that half is
deliberately not part of this repository.
"""

from __future__ import annotations

import flagquantum.remote.realtime as rt


def majority_decode(arguments: tuple[object, ...]) -> tuple[int]:
    """Decode five detection bits into one correction bit by majority vote."""
    events = arguments[0]
    assert isinstance(events, tuple)
    return (int(sum(bool(bit) for bit in events) >= 3),)


def main() -> None:
    # One decoding round: five detection events in, one correction out. The
    # events are bit-packed because a round is a handful of bits, and the
    # schema says so rather than leaving the reader to guess.
    schema = rt.HandlerSchema(
        args=(rt.TypeDescriptor(rt.PayloadType.BIT_PACKED, 1, 5),),
        results=(rt.TypeDescriptor(rt.PayloadType.UINT8, 1),),
    )
    service = rt.FunctionTableService(
        [rt.FunctionEntry("majority_decode", schema, majority_decode)]
    )

    session = service.open_session("host_call")
    assert session is not None
    # A mode this service does not serve returns nothing. It does not degrade
    # to a mode it does serve.
    assert service.open_session("graph_launch") is None

    driver = rt.DeviceCallChannelDriver(
        {"host-dispatch": rt.loopback_channel("host-dispatch", session)}
    )
    print("registered channels:", driver.names)

    # The compiled half, in CUDA-Q, is a device id and a hashed handler name.
    # Here the hash is computed from the name the table registered, so a caller
    # that mistypes the name sends a request no handler owns.
    handler_id = rt.function_id("majority_decode")
    print(f"handler id: 0x{handler_id:08x}")

    with rt.RealtimeSession(driver) as realtime:
        channel = realtime.open_channel(0, "host-dispatch")
        events = (1, 1, 0, 1, 0)
        arguments = rt.encode_arguments(schema, (events,))
        outcome = realtime.call(
            0, handler_id, arguments, request_id=17, ptp_timestamp=1234567890
        )
        assert outcome.succeeded, outcome.response
        print("round 1 correction:", rt.decode_results(schema, outcome.result)[0])
        # The response echoes what the caller chose, so a caller watching a
        # stream of shots can match an answer to the shot that asked.
        assert outcome.response.request_id == 17
        assert outcome.response.ptp_timestamp == 1234567890
        print("slot geometry:", channel.slot_size, "bytes, header", rt.HEADER_SIZE)
        # The layout is printed over a 32-byte slot rather than the channel's own
        # 384-byte one, because the bytes after the payload are padding and the
        # point here is the layout. The geometry above is the real one.
        request = rt.frame_request(
            rt.RPCHeader(handler_id, 1, 17), arguments, slot_size=32
        )
        print("request bytes over a 32-byte slot:", request.hex())

        # A payload that does not match the declared schema is refused. The
        # schema declares one bit-packed byte for five events, so two bytes is a
        # caller defect: the service reports a protocol-level status and echoes
        # the request id rather than reading the extra byte as more events.
        header, payload = rt.read_request(
            rt.frame_request(rt.RPCHeader(handler_id, 2, 18), b"\x01\x02", slot_size=32)
        )
        response, results = session.dispatch(header, payload)
        assert response.status == rt.STATUS_INVALID_ARGUMENTS, response.status
        assert response.request_id == 18
        assert results == b""
        print("mismatched schema status:", response.status)

        # A device id with no channel is named as such, and the message lists
        # the ids that do have one.
        try:
            realtime.call(1, handler_id, arguments)
        except rt.RealtimeCallError as error:
            print("unopened device:", error)

    print("FlagQuantum realtime device call check passed")


if __name__ == "__main__":
    main()
