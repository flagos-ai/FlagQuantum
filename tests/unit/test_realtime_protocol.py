"""The realtime messaging protocol and the device-call channel abstraction.

Every test here is offline: the only transport exercised is the in-process
stand-in, no clock is read, and no test can produce a latency number. What the
tests prove is the software half of the surface -- the slot framing, the payload
type system, the status convention, the function-id hash, and the channel and
session lifecycle -- and, just as importantly, the refusals that keep a caller
from mistaking a malformed call for a served one.

The contract this file witnesses is ``contracts/realtime-messaging-v1-candidate.json``;
``tools/check_realtime_messaging_contract.py`` reconciles that contract with the
implementation and requires the eight witness tests named below to exist, so
renaming one of them is a contract change rather than a private detail.
"""

from __future__ import annotations

import struct

import pytest

from flagquantum.remote import realtime as rt
from flagquantum.remote.realtime import channels, payload, protocol, service, session

pytestmark = pytest.mark.unit

_DECODE = "majority_decode"


def _schema(
    *,
    args: tuple[rt.TypeDescriptor, ...] | None = None,
    results: tuple[rt.TypeDescriptor, ...] | None = None,
) -> rt.HandlerSchema:
    return rt.HandlerSchema(
        args=args or (rt.TypeDescriptor(rt.PayloadType.UINT8, 1),),
        results=results or (rt.TypeDescriptor(rt.PayloadType.UINT8, 1),),
    )


def _echo(arguments: tuple[object, ...]) -> tuple[int]:
    return (int(arguments[0]),)


def _service(
    invoke: object = _echo, *, modes: tuple[rt.DispatchMode, ...] = ("host_call",)
) -> rt.FunctionTableService:
    return rt.FunctionTableService(
        [rt.FunctionEntry(_DECODE, _schema(), invoke)],  # type: ignore[arg-type]
        modes=modes,
    )


def _session(
    invoke: object = _echo, *, modes: tuple[rt.DispatchMode, ...] = ("host_call",)
) -> rt.DeviceCallSession:
    opened = _service(invoke, modes=modes).open_session(modes[0])
    assert opened is not None
    return opened


def _driver(session: rt.DeviceCallSession | None = None) -> rt.DeviceCallChannelDriver:
    return rt.DeviceCallChannelDriver(
        {"host-dispatch": rt.loopback_channel("host-dispatch", session or _session())}
    )


# --- the wire format ---


def test_request_slot_matches_the_declared_bytes() -> None:
    handler_id = rt.function_id(_DECODE)
    header = rt.RPCHeader(handler_id, 1, 17)
    slot = rt.frame_request(header, b"\x0b", slot_size=32)
    assert len(slot) == 32
    # magic, function_id, arg_len, request_id, ptp_timestamp, payload, padding.
    assert slot.hex() == (
        "52515543" "d10627f6" "01000000" "11000000" "0000000000000000" "0b" + "00" * 7
    )
    decoded, body = rt.read_request(slot)
    assert decoded == header
    assert body == b"\x0b"


def test_response_slot_matches_the_declared_bytes() -> None:
    response = rt.RPCResponse(0, 1, 17)
    slot = rt.frame_response(response, b"\x01", slot_size=32)
    assert len(slot) == 32
    # The published protocol's own worked example starts `53 51 55 43`.
    assert slot[:4] == bytes.fromhex("53515543")
    assert slot.hex() == (
        "53515543" "00000000" "01000000" "11000000" "0000000000000000" "01" + "00" * 7
    )
    decoded, body = rt.read_response(slot)
    assert decoded == response
    assert body == b"\x01"


def test_the_magic_words_are_the_documented_ones() -> None:
    assert rt.MAGIC_REQUEST == 0x43555152
    assert rt.MAGIC_RESPONSE == 0x43555153
    assert rt.MAGIC_REQUEST.to_bytes(4, "little") == b"RQUC"
    assert rt.MAGIC_RESPONSE.to_bytes(4, "little") == b"SQUC"


def test_the_header_is_packed_with_no_padding() -> None:
    # A naturally aligned struct would be 24 bytes here too, so the interesting
    # fact is that no field carries padding on any platform.
    assert protocol._REQUEST.size == protocol._RESPONSE.size == rt.HEADER_SIZE == 24
    assert protocol._REQUEST.format == "<IIIIQ"
    assert protocol._RESPONSE.format == "<IiIIQ"


def test_a_foreign_magic_is_refused_rather_than_decoded() -> None:
    good = rt.frame_request(rt.RPCHeader(1, 0, 0), b"", slot_size=32)
    foreign = b"\x00\x00\x00\x00" + good[4:]
    with pytest.raises(
        rt.RealtimeProtocolError, match="does not carry a realtime request"
    ):
        rt.read_request(foreign)
    with pytest.raises(
        rt.RealtimeProtocolError, match="does not carry a realtime response"
    ):
        rt.read_response(foreign)


def test_a_short_slot_is_refused() -> None:
    with pytest.raises(
        rt.RealtimeProtocolError, match="too few for a 24-byte request header"
    ):
        rt.read_request(b"\x52\x51\x55\x43")
    header, payload = rt.read_request(
        rt.RPCHeader(1, 4, 0).encode() + b"\x01\x02\x03\x04"
    )
    assert header.arg_len == 4 and payload == b"\x01\x02\x03\x04"
    with pytest.raises(
        rt.RealtimeProtocolError, match="too few for a 28-byte request payload"
    ):
        rt.read_request(rt.RPCHeader(1, 4, 0).encode())
    with pytest.raises(
        rt.RealtimeProtocolError, match="too few for a 28-byte result payload"
    ):
        rt.read_response(rt.RPCResponse(0, 4, 0).encode())


def test_a_payload_that_overflows_its_slot_is_refused() -> None:
    with pytest.raises(rt.RealtimeProtocolError, match="does not fit the 8 bytes"):
        rt.frame_request(rt.RPCHeader(1, 9, 0), bytes(9), slot_size=32)
    with pytest.raises(rt.RealtimeProtocolError, match="at least 24 bytes"):
        rt.frame_request(rt.RPCHeader(1, 0, 0), b"", slot_size=23)
    with pytest.raises(rt.RealtimeProtocolError, match="but 2 were supplied"):
        rt.frame_request(rt.RPCHeader(1, 3, 0), b"\x01\x02", slot_size=32)
    # A response is framed by the same rule as a request: a result that does not
    # fit is refused rather than written past the end of the slot.
    with pytest.raises(rt.RealtimeProtocolError, match="does not fit the 8 bytes"):
        rt.frame_response(rt.RPCResponse(0, 9, 0), bytes(9), slot_size=32)
    with pytest.raises(rt.RealtimeProtocolError, match="but 2 were supplied"):
        rt.frame_response(rt.RPCResponse(0, 3, 0), b"\x01\x02", slot_size=32)


def test_a_header_field_outside_the_wire_range_is_refused() -> None:
    with pytest.raises(rt.RealtimeProtocolError, match="function_id must be an"):
        rt.RPCHeader(2**32, 0, 0)
    with pytest.raises(rt.RealtimeProtocolError, match="request_id must be an"):
        rt.RPCHeader(0, 0, -1)
    with pytest.raises(rt.RealtimeProtocolError, match="ptp_timestamp must be an"):
        rt.RPCHeader(0, 0, 0, -1)
    with pytest.raises(rt.RealtimeProtocolError, match="arg_len must be a"):
        rt.RPCHeader(0, -1, 0)
    with pytest.raises(rt.RealtimeProtocolError, match="status must be a signed"):
        rt.RPCResponse(2**31, 0, 0)
    with pytest.raises(rt.RealtimeProtocolError, match="result_len must be a"):
        rt.RPCResponse(0, -1, 0)
    with pytest.raises(rt.RealtimeProtocolError, match="request_id must be an"):
        rt.RPCResponse(0, 0, -1)
    with pytest.raises(rt.RealtimeProtocolError, match="ptp_timestamp must be an"):
        rt.RPCResponse(0, 0, 0, -1)


def test_the_default_slot_size_is_this_package_s_own_choice() -> None:
    # CUDA-Q requires a slot size explicitly and documents no default, so the
    # constant below must not be described as inherited from anything.
    assert channels.DEFAULT_SLOT_SIZE == 384
    assert rt.HEADER_SIZE < channels.DEFAULT_SLOT_SIZE


# --- the payload type system ---


def test_bit_packed_payload_is_lsb_first() -> None:
    schema = rt.HandlerSchema(
        args=(rt.TypeDescriptor(rt.PayloadType.BIT_PACKED, 1, 5),)
    )
    assert rt.encode_arguments(schema, ((1, 1, 0, 1, 0),)) == b"\x0b"
    assert rt.decode_arguments(schema, b"\x0b") == ((True, True, False, True, False),)
    eight = rt.HandlerSchema(args=(rt.TypeDescriptor(rt.PayloadType.BIT_PACKED, 1, 8),))
    assert rt.encode_arguments(eight, ((True,) + (False,) * 7,)) == b"\x01"
    assert rt.encode_arguments(eight, ((False,) * 7 + (True,),)) == b"\x80"
    eleven = rt.HandlerSchema(
        args=(rt.TypeDescriptor(rt.PayloadType.BIT_PACKED, 2, 11),)
    )
    packed = rt.encode_arguments(eleven, ((True,) * 11,))
    assert packed == b"\xff\x07"
    assert rt.decode_arguments(eleven, packed) == ((True,) * 11,)


def test_a_bit_packed_payload_refuses_a_non_bit() -> None:
    schema = rt.HandlerSchema(
        args=(rt.TypeDescriptor(rt.PayloadType.BIT_PACKED, 1, 5),)
    )
    with pytest.raises(rt.RealtimeProtocolError, match="bit 2 is neither 0 nor 1"):
        rt.encode_arguments(schema, ((1, 1, 2, 1, 0),))
    with pytest.raises(
        rt.RealtimeProtocolError, match="for 5 bits but 3 were supplied"
    ):
        rt.encode_arguments(schema, ((1, 1, 0),))


def test_schema_image_is_the_declared_size() -> None:
    assert rt.TYPE_DESCRIPTOR_SIZE == 12
    assert rt.SCHEMA_SIZE == 4 + 12 * (rt.MAX_ARGUMENTS + rt.MAX_RESULTS) == 148
    schema = _schema()
    image = schema.encode()
    assert len(image) == rt.SCHEMA_SIZE
    assert rt.HandlerSchema.decode(image) == schema
    assert rt.HandlerSchema().encode() == bytes(rt.SCHEMA_SIZE)
    # The header carries the argument and result counts; the rest is zero.
    assert image[:4] == bytes((1, 1, 0, 0))


def test_a_schema_image_round_trips_both_counts() -> None:
    # The argument and result counts are separate fields, and the result region
    # does not start where the arguments happen to end: an image whose two
    # counts differ is the only one that can tell the two apart.
    schema = rt.HandlerSchema(
        args=(
            rt.TypeDescriptor(rt.PayloadType.UINT8, 1),
            rt.TypeDescriptor(rt.PayloadType.FLOAT64, 8),
        ),
        results=(rt.TypeDescriptor(rt.PayloadType.ARRAY_INT32, 8, 2),),
    )
    assert (schema.num_args, schema.num_results) == (2, 1)
    assert (schema.argument_bytes, schema.result_bytes) == (9, 8)
    assert struct.unpack("<BB", schema.encode()[:2]) == (2, 1)
    assert rt.HandlerSchema.decode(schema.encode()) == schema
    encoded = rt.encode_results(schema, ((1, 2),))
    assert len(encoded) == schema.result_bytes
    assert rt.decode_results(schema, encoded) == ((1, 2),)


def test_a_schema_image_beyond_the_declared_limits_is_refused() -> None:
    # The image is the only place the counts can lie, because the wire carries
    # no schema at all.
    image = payload._SCHEMA_HEADER.pack(rt.MAX_ARGUMENTS + 1, 0, 0) + bytes(
        rt.SCHEMA_SIZE - payload._SCHEMA_HEADER_SIZE
    )
    with pytest.raises(rt.RealtimeProtocolError, match="beyond the 8 and 4"):
        rt.HandlerSchema.decode(image)
    image = payload._SCHEMA_HEADER.pack(0, rt.MAX_RESULTS + 1, 0) + bytes(
        rt.SCHEMA_SIZE - payload._SCHEMA_HEADER_SIZE
    )
    with pytest.raises(rt.RealtimeProtocolError, match="beyond the 8 and 4"):
        rt.HandlerSchema.decode(image)


def test_the_type_ids_are_the_documented_ones() -> None:
    assert {member.name: int(member) for member in rt.PayloadType} == {
        "UINT8": 0x10,
        "INT32": 0x11,
        "INT64": 0x12,
        "FLOAT32": 0x13,
        "FLOAT64": 0x14,
        "ARRAY_UINT8": 0x20,
        "ARRAY_INT32": 0x21,
        "ARRAY_FLOAT32": 0x22,
        "ARRAY_FLOAT64": 0x23,
        "BIT_PACKED": 0x30,
    }


def test_a_descriptor_must_describe_a_possible_buffer() -> None:
    assert rt.TypeDescriptor(rt.PayloadType.UINT8, 1).num_elements == 1
    assert rt.TypeDescriptor(rt.PayloadType.ARRAY_FLOAT64, 24, 3).size_bytes == 24
    assert rt.TypeDescriptor(rt.PayloadType.BIT_PACKED, 2, 11).is_bit_packed
    with pytest.raises(rt.RealtimeProtocolError, match="holds one element, not 2"):
        rt.TypeDescriptor(rt.PayloadType.UINT8, 1, 2)
    with pytest.raises(rt.RealtimeProtocolError, match="occupies 12 bytes, not 8"):
        rt.TypeDescriptor(rt.PayloadType.ARRAY_INT32, 8, 3)
    with pytest.raises(rt.RealtimeProtocolError, match="occupies 2 bytes, not 1"):
        rt.TypeDescriptor(rt.PayloadType.BIT_PACKED, 1, 11)
    with pytest.raises(rt.RealtimeProtocolError, match="holds no bytes"):
        rt.TypeDescriptor(rt.PayloadType.ARRAY_UINT8, 1, 0)
    with pytest.raises(rt.RealtimeProtocolError, match="is not a PayloadType"):
        rt.TypeDescriptor(0x10, 1)  # type: ignore[arg-type]
    # A boolean is an `int` to the interpreter but not a count: `True` would
    # otherwise pass every size check below, since `True == 1`.
    with pytest.raises(rt.RealtimeProtocolError, match="size_bytes must be a"):
        rt.TypeDescriptor(rt.PayloadType.UINT8, True)  # type: ignore[arg-type]
    with pytest.raises(rt.RealtimeProtocolError, match="num_elements must be a"):
        rt.TypeDescriptor(rt.PayloadType.ARRAY_FLOAT64, 8, True)  # type: ignore[arg-type]


def test_a_descriptor_with_an_undefined_type_is_refused() -> None:
    with pytest.raises(
        rt.RealtimeProtocolError, match="0x7f is not a realtime payload"
    ):
        rt.TypeDescriptor.decode(bytes((0x7F, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)))
    with pytest.raises(rt.RealtimeProtocolError, match="descriptor is 12 bytes, not 4"):
        rt.TypeDescriptor.decode(b"\x10\x00\x00\x00")


def test_a_schema_beyond_the_declared_limits_is_refused() -> None:
    with pytest.raises(rt.RealtimeProtocolError, match="at most 8 argument"):
        rt.HandlerSchema(args=(rt.TypeDescriptor(rt.PayloadType.UINT8, 1),) * 9)
    with pytest.raises(rt.RealtimeProtocolError, match="at most 4 result"):
        rt.HandlerSchema(results=(rt.TypeDescriptor(rt.PayloadType.UINT8, 1),) * 5)
    with pytest.raises(rt.RealtimeProtocolError, match="argument 1 is not a"):
        rt.HandlerSchema(args=(rt.TypeDescriptor(rt.PayloadType.UINT8, 1), "nope"))  # type: ignore[arg-type]
    with pytest.raises(
        rt.RealtimeProtocolError, match="a handler schema image is 148 bytes, not 4"
    ):
        rt.HandlerSchema.decode(b"\x01\x01\x00\x00")


def test_arguments_are_concatenated_with_no_delimiters() -> None:
    schema = rt.HandlerSchema(
        args=(
            rt.TypeDescriptor(rt.PayloadType.UINT8, 1),
            rt.TypeDescriptor(rt.PayloadType.INT32, 4),
            rt.TypeDescriptor(rt.PayloadType.ARRAY_FLOAT32, 8, 2),
        )
    )
    encoded = rt.encode_arguments(schema, (7, 8, (1.5, -2.5)))
    assert encoded == b"\x07\x08\x00\x00\x00" + struct.pack("<2f", 1.5, -2.5)
    assert schema.argument_bytes == len(encoded) == 13
    assert rt.decode_arguments(schema, encoded) == (7, 8, (1.5, -2.5))


def test_a_payload_that_disagrees_with_the_schema_is_refused() -> None:
    schema = _schema()
    with pytest.raises(rt.RealtimeProtocolError, match="declares 1 argument bytes"):
        rt.decode_arguments(schema, b"\x01\x02")
    # A list of the right length would slice and unpack like bytes, so the
    # payload type is checked before the length.
    with pytest.raises(rt.RealtimeProtocolError, match="must be bytes"):
        rt.decode_arguments(schema, [1])
    with pytest.raises(rt.RealtimeProtocolError, match="must be a sequence"):
        rt.encode_arguments(schema, b"\x01")
    with pytest.raises(rt.RealtimeProtocolError, match="declares 1 arguments but 2"):
        rt.encode_arguments(schema, (1, 2))
    with pytest.raises(rt.RealtimeProtocolError, match="must be an integer"):
        rt.encode_arguments(schema, (1.5,))
    with pytest.raises(rt.RealtimeProtocolError, match="must be a number"):
        rt.encode_arguments(
            rt.HandlerSchema(args=(rt.TypeDescriptor(rt.PayloadType.FLOAT64, 8),)),
            ("wide",),
        )
    with pytest.raises(rt.RealtimeProtocolError, match="does not fit a UINT8"):
        rt.encode_arguments(schema, (256,))
    with pytest.raises(rt.RealtimeProtocolError, match="must be a sequence"):
        rt.encode_arguments(
            rt.HandlerSchema(
                args=(rt.TypeDescriptor(rt.PayloadType.ARRAY_UINT8, 2, 2),)
            ),
            (b"\x01\x02",),
        )
    with pytest.raises(
        rt.RealtimeProtocolError, match="of 2 elements but 1 were supplied"
    ):
        rt.encode_arguments(
            rt.HandlerSchema(
                args=(rt.TypeDescriptor(rt.PayloadType.ARRAY_UINT8, 2, 2),)
            ),
            ((1,),),
        )


def test_a_scalar_round_trips_every_declared_width() -> None:
    for payload_type, value, width in (
        (rt.PayloadType.UINT8, 255, 1),
        (rt.PayloadType.INT32, -(2**31), 4),
        (rt.PayloadType.INT64, 2**62, 8),
        (rt.PayloadType.FLOAT32, 0.5, 4),
        (rt.PayloadType.FLOAT64, -0.5, 8),
    ):
        descriptor = rt.TypeDescriptor(payload_type, width)
        schema = rt.HandlerSchema(args=(descriptor,), results=(descriptor,))
        encoded = rt.encode_arguments(schema, (value,))
        assert len(encoded) == width
        assert rt.decode_arguments(schema, encoded) == (value,)


def test_payload_widths_are_little_endian() -> None:
    schema = rt.HandlerSchema(args=(rt.TypeDescriptor(rt.PayloadType.INT32, 4),))
    assert rt.encode_arguments(schema, (1,)) == b"\x01\x00\x00\x00"


# --- the function id ---


def test_function_id_matches_the_published_vectors() -> None:
    # The FNV specification's own 32-bit vectors, which this repository did not
    # choose and therefore can be checked against.
    assert rt.function_id("a") == 0xE40C292C
    assert rt.function_id("foobar") == 0xBF9CF968
    assert rt.function_id(_DECODE) == 0xF62706D1
    assert rt.function_id("qec_decode") == 0xC4B0E4BF
    assert protocol._FNV_OFFSET_BASIS == 2166136261
    assert protocol._FNV_PRIME == 16777619
    assert protocol._FNV_MASK == 0xFFFFFFFF


def test_a_blank_handler_name_has_no_function_id() -> None:
    for name in ("", "   ", "\t"):
        with pytest.raises(rt.RealtimeProtocolError, match="non-empty string"):
            rt.function_id(name)
    with pytest.raises(rt.RealtimeProtocolError, match="non-empty string"):
        rt.function_id(None)  # type: ignore[arg-type]


# --- the status convention ---


def test_a_served_call_reports_success_and_echoes_the_request() -> None:
    response, result = _session().dispatch(
        rt.RPCHeader(rt.function_id(_DECODE), 1, 4242, 99), b"\x2a"
    )
    assert response.status == rt.STATUS_SUCCESS == 0
    assert response.succeeded
    assert (response.request_id, response.ptp_timestamp) == (4242, 99)
    assert rt.decode_results(_schema(), result) == (42,)


def test_a_failed_call_echoes_the_request_id() -> None:
    def explode(_: tuple[object, ...]) -> tuple[int]:
        raise RuntimeError("the handler itself is broken")

    def refuse(_: tuple[object, ...]) -> tuple[int]:
        raise rt.RealtimeHandlerError(7, "the handler refused this round")

    for invoke, expected in ((explode, rt.STATUS_HANDLER_RAISED), (refuse, 7)):
        response, result = _session(invoke).dispatch(
            rt.RPCHeader(rt.function_id(_DECODE), 1, 4242, 99), b"\x00"
        )
        assert response.status == expected
        assert not response.succeeded
        assert (response.request_id, response.ptp_timestamp) == (4242, 99)
        assert result == b""
    assert rt.STATUS_UNKNOWN_FUNCTION < 0
    assert rt.STATUS_INVALID_ARGUMENTS < 0
    assert rt.STATUS_HANDLER_RAISED < 0


def test_an_unknown_function_id_gets_no_result() -> None:
    response, result = _session().dispatch(rt.RPCHeader(12345, 0, 5, 11), b"")
    assert response.status == rt.STATUS_UNKNOWN_FUNCTION
    assert (response.request_id, response.ptp_timestamp, result) == (5, 11, b"")


def test_arguments_that_do_not_match_the_schema_are_a_protocol_status() -> None:
    response, result = _session().dispatch(
        rt.RPCHeader(rt.function_id(_DECODE), 2, 5), b"\x01\x02"
    )
    assert response.status == rt.STATUS_INVALID_ARGUMENTS
    assert result == b""


def test_a_result_that_does_not_match_the_schema_is_not_reported_as_success() -> None:
    def wide(_: tuple[object, ...]) -> tuple[int, int]:
        return (1, 2)

    response, result = _session(wide).dispatch(
        rt.RPCHeader(rt.function_id(_DECODE), 1, 5), b"\x01"
    )
    assert response.status == rt.STATUS_HANDLER_RAISED
    assert result == b""


def test_a_handler_status_must_be_a_handler_failure() -> None:
    for status in (0, -1):
        with pytest.raises(rt.RealtimeCallError, match="positive signed 32-bit"):
            rt.RealtimeHandlerError(status, "not a handler failure")
    assert rt.RealtimeHandlerError(3, "refused").status == 3


def test_a_single_result_need_not_be_wrapped_in_a_sequence() -> None:
    def scalar(_: tuple[object, ...]) -> int:
        return 3

    response, result = _session(scalar).dispatch(
        rt.RPCHeader(rt.function_id(_DECODE), 1, 5), b"\x01"
    )
    assert response.status == rt.STATUS_SUCCESS
    assert rt.decode_results(_schema(), result) == (3,)


# --- the function table ---


def test_a_function_table_hashes_every_handler_name() -> None:
    entry = rt.FunctionEntry(_DECODE, _schema(), _echo)
    assert entry.function_id == rt.function_id(_DECODE)
    assert list(_session().function_table()) == [rt.function_id(_DECODE)]
    assert _service().entries == (entry,)


def test_a_function_table_refuses_a_collision() -> None:
    entry = rt.FunctionEntry(_DECODE, _schema(), _echo)
    with pytest.raises(rt.RealtimeCallError, match="collide on function id"):
        rt.FunctionTableService([entry, entry])


def test_a_function_table_refuses_an_empty_or_malformed_entry() -> None:
    with pytest.raises(rt.RealtimeCallError, match="at least one handler"):
        rt.FunctionTableService([])
    with pytest.raises(rt.RealtimeCallError, match="non-empty handler name"):
        rt.FunctionEntry("  ", _schema(), _echo)
    with pytest.raises(rt.RealtimeCallError, match="needs a HandlerSchema"):
        rt.FunctionEntry(_DECODE, object(), _echo)  # type: ignore[arg-type]
    with pytest.raises(rt.RealtimeCallError, match="needs a callable handler"):
        rt.FunctionEntry(_DECODE, _schema(), object())  # type: ignore[arg-type]


def test_a_function_table_stays_valid_until_the_session_stops() -> None:
    session = _session()
    table = session.function_table()
    assert session.mode == "host_call"
    assert list(table) == [rt.function_id(_DECODE)]
    session.stop()
    with pytest.raises(rt.RealtimeCallError, match="no longer owns a function table"):
        session.function_table()
    with pytest.raises(rt.RealtimeCallError, match="session is stopped"):
        session.dispatch(rt.RPCHeader(1, 0, 0), b"")
    session.stop()


# --- the channel abstraction ---


def test_a_channel_is_selected_at_run_time_by_name() -> None:
    # The channel name is not compiled in: any name reaches the same table, and
    # no channel name appears in that table.
    driver = rt.DeviceCallChannelDriver(
        {"shared-memory": rt.loopback_channel("shared-memory", _session())}
    )
    assert driver.names == ("shared-memory",)
    with rt.RealtimeSession(driver) as realtime:
        channel = realtime.open_channel(0, "shared-memory")
        assert channel.name == "shared-memory"
        assert channel.is_open
        assert channel.slot_size == rt.DEFAULT_SLOT_SIZE
        outcome = realtime.call(
            0, rt.function_id(_DECODE), rt.encode_arguments(_schema(), (9,))
        )
    assert outcome.succeeded
    assert rt.decode_results(_schema(), outcome.result) == (9,)
    assert all(
        entry.name != "shared-memory" for entry in _session().function_table().values()
    )


def test_a_channel_nobody_registered_is_refused_by_name() -> None:
    driver = rt.DeviceCallChannelDriver(
        {"host-dispatch": rt.loopback_channel("host-dispatch", _session())}
    )
    with pytest.raises(
        rt.RealtimeCallError, match="registered channels are host-dispatch"
    ):
        driver.open("gpu-dispatch")
    with pytest.raises(rt.RealtimeCallError, match="registered channels are none"):
        rt.DeviceCallChannelDriver().open("host-dispatch")


def test_a_channel_is_registered_once() -> None:
    driver = rt.DeviceCallChannelDriver()
    factory = rt.loopback_channel("host-dispatch", _session())
    driver.register("host-dispatch", factory)
    with pytest.raises(rt.RealtimeCallError, match="already registered"):
        driver.register("host-dispatch", factory)
    with pytest.raises(rt.RealtimeCallError, match="non-empty string"):
        driver.register("", factory)
    with pytest.raises(rt.RealtimeCallError, match="needs a callable factory"):
        driver.register("other", object())  # type: ignore[arg-type]


def test_a_channel_refuses_a_call_before_it_is_open() -> None:
    channel = rt.DeviceCallChannel(
        "host-dispatch",
        channels.LoopbackTransport(_session(), slot_size=64),
        slot_size=64,
    )
    with pytest.raises(rt.RealtimeCallError, match="is not open"):
        channel.call(1)
    channel.open()
    with pytest.raises(rt.RealtimeCallError, match="is already open"):
        channel.open()
    channel.close()
    channel.close()
    with pytest.raises(rt.RealtimeCallError, match="is not open"):
        channel.call(1)


def test_a_channel_refuses_a_nameless_channel() -> None:
    with pytest.raises(rt.RealtimeCallError, match="non-empty string"):
        rt.DeviceCallChannel(
            "", channels.LoopbackTransport(_session(), slot_size=64), slot_size=64
        )


def test_a_response_for_another_request_is_refused() -> None:
    # A ring that answers something other than what was asked must not be read
    # as an answer to this call.
    class _WrongId:
        def __init__(self, session: rt.DeviceCallSession, *, slot_size: int) -> None:
            self._inner = channels.LoopbackTransport(session, slot_size=slot_size)
            self._slot_size = slot_size

        def publish_request(self, slot: bytes) -> None:
            header, body = rt.read_request(slot)
            self._inner.publish_request(
                rt.frame_request(
                    rt.RPCHeader(
                        header.function_id, header.arg_len, header.request_id + 1
                    ),
                    body,
                    slot_size=self._slot_size,
                )
            )

        def await_response(self) -> bytes:
            return self._inner.await_response()

        def close(self) -> None:
            self._inner.close()

    channel = rt.DeviceCallChannel(
        "host-dispatch", _WrongId(_session(), slot_size=64), slot_size=64
    )
    channel.open()
    with pytest.raises(
        rt.RealtimeProtocolError, match="sent request id 3 but the response echoes 4"
    ):
        channel.call(rt.function_id(_DECODE), b"\x01", request_id=3)
    channel.close()


def test_a_response_for_another_timestamp_is_refused() -> None:
    class _WrongTimestamp(channels.LoopbackTransport):
        def await_response(self) -> bytes:
            response, body = rt.read_response(super().await_response())
            return rt.frame_response(
                rt.RPCResponse(
                    response.status,
                    response.result_len,
                    response.request_id,
                    response.ptp_timestamp + 1,
                ),
                body,
                slot_size=self._slot_size,
            )

    channel = rt.DeviceCallChannel(
        "host-dispatch",
        _WrongTimestamp(_session(), slot_size=64),
        slot_size=64,
    )
    channel.open()
    with pytest.raises(
        rt.RealtimeProtocolError, match="sent timestamp 5 but the response echoes 6"
    ):
        channel.call(rt.function_id(_DECODE), b"\x01", ptp_timestamp=5)
    channel.close()


def test_the_loopback_ring_only_answers_a_request_it_dispatched() -> None:
    transport = channels.LoopbackTransport(_session(), slot_size=64)
    assert transport.processed == 0
    with pytest.raises(rt.RealtimeCallError, match="holds no response"):
        transport.await_response()
    transport.publish_request(
        rt.frame_request(
            rt.RPCHeader(rt.function_id(_DECODE), 1, 0), b"\x01", slot_size=64
        )
    )
    assert transport.processed == 1
    assert rt.read_response(transport.await_response())[0].succeeded
    transport.close()
    with pytest.raises(rt.RealtimeCallError, match="ring is closed"):
        transport.publish_request(b"")
    assert transport.processed == 1


def test_a_loopback_channel_takes_no_in_flight_count() -> None:
    # A slot count bounds how many requests a ring can hold at once, and an
    # in-process ring dispatches each request as it arrives.
    import inspect as _inspect

    parameters = _inspect.signature(rt.loopback_channel).parameters
    assert "slot_count" not in parameters
    assert parameters["slot_size"].default == rt.DEFAULT_SLOT_SIZE


# --- the device-call session ---


def test_a_session_refuses_a_mode_it_does_not_serve() -> None:
    service_ = _service(modes=("host_call",))
    assert service_.modes == ("host_call",)
    assert service_.open_session("graph_launch") is None
    assert service_.open_session("host_call").mode == "host_call"  # type: ignore[union-attr]
    with pytest.raises(
        rt.RealtimeCallError, match="not one of host_call, graph_launch"
    ):
        service_.open_session("unified")  # type: ignore[arg-type]
    with pytest.raises(
        rt.RealtimeCallError, match="not one of host_call, graph_launch"
    ):
        rt.FunctionTableService(
            [rt.FunctionEntry(_DECODE, _schema(), _echo)],
            modes=("unified",),  # type: ignore[arg-type]
        )
    assert rt.DISPATCH_MODES == ("host_call", "graph_launch")


def test_a_session_is_keyed_by_device_id() -> None:
    driver = _driver()
    with rt.RealtimeSession(driver) as realtime:
        assert realtime.device_ids == ()
        realtime.open_channel(0, "host-dispatch")
        assert realtime.device_ids == (0,)
        realtime.open_channel(1, "host-dispatch")
        assert realtime.device_ids == (0, 1)
        assert realtime.channel(1) is not realtime.channel(0)
        with pytest.raises(rt.RealtimeCallError, match="already has an open channel"):
            realtime.open_channel(0, "host-dispatch")
        with pytest.raises(rt.RealtimeCallError, match="non-negative integer"):
            realtime.open_channel(-1, "host-dispatch")
        with pytest.raises(rt.RealtimeCallError, match="open device ids are 0, 1"):
            realtime.call(2, 1)
        outcome = realtime.call(
            1, rt.function_id(_DECODE), rt.encode_arguments(_schema(), (7,))
        )
        assert rt.decode_results(_schema(), outcome.result) == (7,)


def test_a_closed_session_refuses_further_use() -> None:
    realtime = rt.RealtimeSession(_driver())
    # Before it is closed a session still answers, by name, what it cannot find.
    with pytest.raises(rt.RealtimeCallError, match="open device ids are none"):
        realtime.channel(0)
    realtime.close()
    for attempt in (
        lambda: realtime.channel(0),
        lambda: realtime.open_channel(0, "host-dispatch"),
        lambda: realtime.call(0, 1),
        realtime.__enter__,
    ):
        with pytest.raises(rt.RealtimeCallError, match="session is closed"):
            attempt()
    realtime.close()


def test_closing_a_session_closes_the_channels_it_opened() -> None:
    driver = _driver()
    realtime = rt.RealtimeSession(driver)
    channel = realtime.open_channel(0, "host-dispatch")
    realtime.close()
    assert not channel.is_open
    assert realtime.device_ids == ()


# --- the honesty boundary ---


def test_no_module_in_the_package_reads_a_clock_or_opens_a_socket() -> None:
    """The package carries timestamps and never measures them."""
    import pathlib

    text = "".join(
        pathlib.Path(module.__file__ or "").read_text(encoding="utf-8")
        for module in (protocol, payload, channels, service, session)
    )
    for forbidden in (
        "import socket",
        "import asyncio",
        "import time",
        "time.time",
        "time.monotonic",
        "perf_counter",
        "import threading",
        "import multiprocessing",
        "import ctypes",
    ):
        assert forbidden not in text, forbidden


def test_the_package_makes_no_hardware_or_timing_claim() -> None:
    import flagquantum.remote.realtime as package

    docstring = package.__doc__ or ""
    for term in ("FPGA", "RDMA", "GPUDirect", "NIC"):
        assert term in docstring, term
    lowered = docstring.lower()
    for forbidden in ("microsecond latency", "we achieve", "guarantees"):
        assert forbidden not in lowered, forbidden


def test_the_protocol_vocabulary_is_this_package_s_own() -> None:
    # The layout, ids, and limits come from CUDA-Q's published protocol; the
    # names do not. A reader must be able to tell which is which.
    assert protocol.__doc__ is not None
    assert "not here" in payload.__doc__.lower()
    assert "not here" in channels.__doc__.lower()
    assert "not here" in service.__doc__.lower()
    assert rt.PROTOCOL_SCHEMA == "flagquantum.realtime_messaging.v1"
