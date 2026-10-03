"""The realtime RPC wire format: a fixed header, an opaque payload, one slot.

A realtime device call reaches a classical service while a program is still
running. The service sits behind a ring buffer, so the unit of interchange is
one fixed-size *slot* holding an :class:`RPCHeader`, the payload, and padding.
This module owns that framing and nothing else: the payload's meaning belongs
to the handler schema in :mod:`.payload`, and the trip out and back belongs to
the channels in :mod:`.channels`.

Every field is defined here rather than inferred from a vendor header, so an
interoperating implementation -- including one on an FPGA -- has exactly one
place to read the layout from. `contracts/realtime-messaging-v1-candidate.json`
restates that layout for tooling, and `tools/check_realtime_messaging_contract.py`
fails if the two disagree.

**What is not here.** No dispatcher, no ring buffer, no network transport, no
clock. `ptp_timestamp` is carried and echoed verbatim; this module never reads a
clock and makes no latency claim.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

__all__ = [
    "HEADER_SIZE",
    "MAGIC_REQUEST",
    "MAGIC_RESPONSE",
    "PROTOCOL_SCHEMA",
    "STATUS_SUCCESS",
    "RPCHeader",
    "RPCResponse",
    "RealtimeCallError",
    "RealtimeProtocolError",
    "frame_request",
    "frame_response",
    "function_id",
    "read_request",
    "read_response",
]

PROTOCOL_SCHEMA = "flagquantum.realtime_messaging.v1"

#: Magic words are little-endian 32-bit and spell a four-character tag so a
#: misconfigured ring is caught on the first slot rather than decoded as data.
MAGIC_REQUEST = 0x43555152  # "CUQR"
MAGIC_RESPONSE = 0x43555153  # "CUQS"

#: A request is magic, function_id, arg_len, request_id, ptp_timestamp. A
#: response replaces function_id and arg_len with status and result_len. Both
#: are packed with no padding, so both are exactly this many bytes.
HEADER_SIZE = 24

#: Status 0 is success, positive is a handler-specific error, negative is a
#: protocol-level error. The dispatcher never invents a status outside that
#: convention; see :mod:`.channels`.
STATUS_SUCCESS = 0

_REQUEST = struct.Struct("<IIIIQ")
_RESPONSE = struct.Struct("<IiIIQ")

#: FNV-1a parameters, fixed by the contract so a second implementation can
#: reproduce a function id without reading this file.
_FNV_OFFSET_BASIS = 2166136261
_FNV_PRIME = 16777619
_FNV_MASK = 0xFFFFFFFF


class RealtimeProtocolError(ValueError):
    """A slot, header, or payload that this protocol refuses to read."""


class RealtimeCallError(RuntimeError):
    """A device call refused before or around the exchange itself.

    Wire and payload malformation raises :class:`RealtimeProtocolError`; a
    channel that is not open, a channel name nobody registered, or an
    unregistered device id is a lifecycle mistake instead, and separating the
    two lets a caller retry one and not the other.
    """


@dataclass(frozen=True, slots=True)
class RPCHeader:
    """The fixed 24-byte prefix of a request slot.

    `ptp_timestamp` is opaque: by convention it carries a Precision Time
    Protocol time-of-day in nanoseconds, and the dispatcher echoes it without
    interpreting it, so measuring latency stays the caller's concern.
    """

    function_id: int
    arg_len: int
    request_id: int
    ptp_timestamp: int = 0

    def __post_init__(self) -> None:
        if type(self.function_id) is not int or not 0 <= self.function_id <= _FNV_MASK:
            raise RealtimeProtocolError(
                "function_id must be an unsigned 32-bit integer"
            )
        if type(self.arg_len) is not int or self.arg_len < 0:
            raise RealtimeProtocolError("arg_len must be a non-negative integer")
        if type(self.request_id) is not int or not 0 <= self.request_id <= _FNV_MASK:
            raise RealtimeProtocolError("request_id must be an unsigned 32-bit integer")
        if type(self.ptp_timestamp) is not int or not 0 <= self.ptp_timestamp < 2**64:
            raise RealtimeProtocolError(
                "ptp_timestamp must be an unsigned 64-bit integer"
            )

    def encode(self) -> bytes:
        """Return the tag, the function id, the payload length, and the ids."""
        return _REQUEST.pack(
            MAGIC_REQUEST,
            self.function_id,
            self.arg_len,
            self.request_id,
            self.ptp_timestamp,
        )

    @classmethod
    def decode(cls, slot: bytes) -> RPCHeader:
        """Read a header from the front of `slot`, refusing a foreign tag."""
        _require_room(slot, HEADER_SIZE, owner="request header")
        magic, function_id, arg_len, request_id, ptp_timestamp = _REQUEST.unpack_from(
            slot
        )
        if magic != MAGIC_REQUEST:
            raise RealtimeProtocolError(
                f"ring slot does not carry a realtime request: magic 0x{magic:08x} "
                f"is not 0x{MAGIC_REQUEST:08x}"
            )
        return cls(function_id, arg_len, request_id, ptp_timestamp)


@dataclass(frozen=True, slots=True)
class RPCResponse:
    """The fixed 24-byte prefix of a response slot.

    `request_id` and `ptp_timestamp` are echoed verbatim from the request, so a
    caller can match a response to a shot index or a sequence number it chose.
    """

    status: int
    result_len: int
    request_id: int
    ptp_timestamp: int = 0

    def __post_init__(self) -> None:
        if type(self.status) is not int or not -(2**31) <= self.status < 2**31:
            raise RealtimeProtocolError("status must be a signed 32-bit integer")
        if type(self.result_len) is not int or self.result_len < 0:
            raise RealtimeProtocolError("result_len must be a non-negative integer")
        if type(self.request_id) is not int or not 0 <= self.request_id <= _FNV_MASK:
            raise RealtimeProtocolError("request_id must be an unsigned 32-bit integer")
        if type(self.ptp_timestamp) is not int or not 0 <= self.ptp_timestamp < 2**64:
            raise RealtimeProtocolError(
                "ptp_timestamp must be an unsigned 64-bit integer"
            )

    @property
    def succeeded(self) -> bool:
        """Whether the dispatcher reported success rather than a failure."""
        return self.status == STATUS_SUCCESS

    def encode(self) -> bytes:
        """Return the tag, the status, the result length, and the echoed ids."""
        return _RESPONSE.pack(
            MAGIC_RESPONSE,
            self.status,
            self.result_len,
            self.request_id,
            self.ptp_timestamp,
        )

    @classmethod
    def decode(cls, slot: bytes) -> RPCResponse:
        """Read a response header from the front of `slot`, refusing a foreign tag."""
        _require_room(slot, HEADER_SIZE, owner="response header")
        magic, status, result_len, request_id, ptp_timestamp = _RESPONSE.unpack_from(
            slot
        )
        if magic != MAGIC_RESPONSE:
            raise RealtimeProtocolError(
                f"ring slot does not carry a realtime response: magic 0x{magic:08x} "
                f"is not 0x{MAGIC_RESPONSE:08x}"
            )
        return cls(status, result_len, request_id, ptp_timestamp)


def function_id(name: str) -> int:
    """Return the FNV-1a 32-bit id that selects the handler called `name`.

    The id is baked into a compiled call site while the handler name stays a
    string in the service's table, so the two ends agree on a number and the
    number is pinned by this function rather than by a header a caller cannot
    read.
    """
    if not isinstance(name, str) or not name.strip():
        raise RealtimeProtocolError("a handler name must be a non-empty string")
    digest = _FNV_OFFSET_BASIS
    for byte in name.encode("utf-8"):
        digest = ((digest ^ byte) * _FNV_PRIME) & _FNV_MASK
    return digest


def frame_request(header: RPCHeader, payload: bytes, *, slot_size: int) -> bytes:
    """Lay a header and its payload into one `slot_size` byte request slot.

    The trailing bytes are zero padding: a slot is fixed size because a ring
    buffer is addressed by index, not by length. Refuses a payload that does not
    fit rather than truncating it, because a truncated request would be decodable
    as a different request.
    """
    _require_slot_size(slot_size)
    if len(payload) != header.arg_len:
        raise RealtimeProtocolError(
            f"header declares {header.arg_len} payload bytes but {len(payload)} were supplied"
        )
    room = slot_size - HEADER_SIZE
    if header.arg_len > room:
        raise RealtimeProtocolError(
            f"a {header.arg_len}-byte payload does not fit the {room} bytes a "
            f"{slot_size}-byte ring slot leaves after its header"
        )
    return header.encode() + payload + bytes(room - header.arg_len)


def read_request(slot: bytes) -> tuple[RPCHeader, bytes]:
    """Return the header and exactly the declared payload of a request slot."""
    header = RPCHeader.decode(slot)
    _require_room(slot, HEADER_SIZE + header.arg_len, owner="request payload")
    return header, bytes(slot[HEADER_SIZE : HEADER_SIZE + header.arg_len])


def frame_response(response: RPCResponse, payload: bytes, *, slot_size: int) -> bytes:
    """Lay a response header and its result payload into one response slot."""
    _require_slot_size(slot_size)
    if len(payload) != response.result_len:
        raise RealtimeProtocolError(
            f"response declares {response.result_len} result bytes but "
            f"{len(payload)} were supplied"
        )
    room = slot_size - HEADER_SIZE
    if response.result_len > room:
        raise RealtimeProtocolError(
            f"a {response.result_len}-byte result does not fit the {room} bytes a "
            f"{slot_size}-byte ring slot leaves after its header"
        )
    return response.encode() + payload + bytes(room - response.result_len)


def read_response(slot: bytes) -> tuple[RPCResponse, bytes]:
    """Return the header and exactly the declared result of a response slot."""
    response = RPCResponse.decode(slot)
    _require_room(slot, HEADER_SIZE + response.result_len, owner="result payload")
    return response, bytes(slot[HEADER_SIZE : HEADER_SIZE + response.result_len])


def _require_slot_size(slot_size: int) -> None:
    if type(slot_size) is not int or slot_size < HEADER_SIZE:
        raise RealtimeProtocolError(
            f"slot_size must be an integer of at least {HEADER_SIZE} bytes"
        )


def _require_room(slot: bytes, needed: int, *, owner: str) -> None:
    if len(slot) < needed:
        raise RealtimeProtocolError(
            f"ring slot holds {len(slot)} bytes, too few for a {needed}-byte {owner}"
        )
