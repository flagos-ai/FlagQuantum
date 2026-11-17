"""Byte parity with the locked KMB v2 writer and fail-closed packet reads."""

import json
from base64 import b64decode
from hashlib import sha256
from pathlib import Path
from struct import pack_into, unpack_from

import pytest

from flagquantum.compiler.formats.kmb import decode_kmb, encode_kmb
from flagquantum.compiler.formats.kmb_operations import OPERATION_CODES

_FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "kickmix_kmb.json").read_text()
)


@pytest.mark.parametrize(
    "count,expected",
    [
        (0, "4a5f45be02c47817938a91c5c9b8fc7c0653183e204968e9f06fa846069b014a"),
        (1, "03a0a3e142fb78a78e42bf3221422fa000b0e9b25b9e58a3cab1d3d93edd7781"),
        (127, "aa16903fcc523269b7b6ac93d2823f2d9234b104dfd35e88ea63759169dfa2e6"),
        (128, "a63b37e1b185f9ae87362f74ccac9e4c32070c865be7fd60f5eecf8117dcac8f"),
        (129, "62e06d568a9eeedf8764133725a8d8c95f51c6f0b88fbba7235de0c34e1f0e61"),
        (257, "9f7b99ca196096d1b87b4ea8516d7a0d25962ba06826327e2efc9f050204c71d"),
    ],
)
def test_word_block_matches_locked_kmb_writer(count: int, expected: str) -> None:
    text = "\n".join(f"X q{0x01234500 + index}" for index in range(count))
    raw = encode_kmb(text)
    assert sha256(raw).hexdigest() == expected
    assert decode_kmb(raw) == text
    if count:
        with pytest.raises(ValueError, match="Truncated"):
            decode_kmb(raw[: 216 + 5 * count - 1])


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
