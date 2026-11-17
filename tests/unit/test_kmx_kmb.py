"""Byte parity with the locked KMB v2 writer and fail-closed packet reads."""

import json
from base64 import b64decode
from pathlib import Path
from struct import pack_into, unpack_from

import pytest

from flagquantum.compiler.formats.kmb import decode_kmb, encode_kmb
from flagquantum.compiler.formats.kmb_operations import OPERATION_CODES

_FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "kickmix_kmb.json").read_text()
)


@pytest.mark.parametrize("name", tuple(_FIXTURES))
def test_source_kmb_bytes_roundtrip(name: str) -> None:
    fixture = _FIXTURES[name]
    raw = b64decode(fixture["kmb"])
    assert decode_kmb(raw) == fixture["kmx"]
    assert encode_kmb(fixture["kmx"]) == raw


def test_source_opcode_map_covers_every_operation_byte() -> None:
    raw = b64decode(_FIXTURES["all_operations"]["kmb"])
    packet, length = unpack_from("<QQ", raw, 104)
    assert packet == 1
    assert set(raw[120 : 120 + length]) == set(OPERATION_CODES.values())
    assert len(OPERATION_CODES) == 39


@pytest.mark.parametrize(
    "offset,fmt,value,diagnostic",
    [
        (0, "<B", 0, "magic"),
        (16, "<I", 1, "version"),
        (20, "<I", 7, "width"),
        (28, "<I", 1_000_001, "limits"),
        (40, "<Q", 3, "column count"),
        (104, "<Q", 91, "packet order"),
        (112, "<Q", 999, "packet order"),
        (120, "<B", 255, "Unknown KMB operation"),
        (138, "<I", 1, "repeats a quantum wire"),
    ],
)
def test_corrupt_kmb_header_or_payload_is_refused(
    offset: int, fmt: str, value: int, diagnostic: str
) -> None:
    raw = bytearray(b64decode(_FIXTURES["guarded_cswap"]["kmb"]))
    pack_into(fmt, raw, offset, value)
    with pytest.raises(ValueError, match=diagnostic):
        decode_kmb(bytes(raw))


def test_kmb_rejects_truncation_trailing_bytes_and_bad_tag_offsets() -> None:
    raw = b64decode(_FIXTURES["metadata"]["kmb"])
    for end in (0, 16, 104, len(raw) - 1):
        with pytest.raises(ValueError, match="Truncated"):
            decode_kmb(raw[:end])
    with pytest.raises(ValueError, match="trailing"):
        decode_kmb(raw + b"\0")
    altered = bytearray(raw)
    packets = 104
    while packets < len(raw):
        kind, size = unpack_from("<QQ", raw, packets)
        if kind == 11:
            pack_into("<Q", altered, packets + 24, 65536)
            break
        packets += 16 + size
    with pytest.raises(ValueError, match="tag offset"):
        decode_kmb(bytes(altered))


def test_binary_ingress_accepts_comments_and_rejects_invalid_operations() -> None:
    assert decode_kmb(encode_kmb("  X q0 # comment\n")) == "X q0"
    with pytest.raises(ValueError, match="repeats"):
        encode_kmb("CX q0 q0")
    with pytest.raises(ValueError, match="Invalid KMX operand"):
        encode_kmb("X b0")
    with pytest.raises(ValueError, match="Invalid KMX operation"):
        encode_kmb("POP_CONDITION if b0")
