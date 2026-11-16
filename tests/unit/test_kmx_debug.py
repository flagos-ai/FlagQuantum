"""Physical DEBUG_PRINT records stay ordered beside native KMX operations."""

from pathlib import Path

import pytest

from flagquantum.compiler._component_trace.trace import ComponentCall
from flagquantum.compiler.formats.kmx import decode_kmx
from flagquantum.compiler.formats.kmx_debug import (
    KmxDebugRecord,
    decode_kmx_debug,
    encode_kmx_debug,
)
from flagquantum.compiler.formats.kmx_registers import KmxRegister

pytestmark = pytest.mark.unit
_HEADER = 'INFO_TAG "@flagquantum:reversible-phase-v1 wires=2 bits=4 inputs=0,1,2"\n'
_DECLARATIONS = (
    "APPEND_TO_REGISTER q0 r0\nAPPEND_TO_REGISTER q1 r0\nAPPEND_TO_REGISTER b2 r0\n"
)
_SOURCE = (
    "DEBUG_PRINT q1\n"
    "DEBUG_PRINT q0\n"
    "DEBUG_PRINT\n"
    "DEBUG_PRINT q0 if b0\n"
    "DEBUG_PRINT q1 if b0\n"
    "DEBUG_PRINT b2 if b0\n"
    "DEBUG_PRINT if b0\n"
    "BIT_STORE1 b3\n"
    "DEBUG_PRINT b3\n"
    "BIT_STORE0 b3\n"
    "DEBUG_PRINT b3\n"
    "DEBUG_PRINT b2\n"
    "BIT_STORE0 b2\n"
    "PUSH_CONDITION if b0\n"
    "BIT_STORE1 b2 if b1\n"
    "POP_CONDITION\n"
)


@pytest.mark.ftqc_reference
def test_locked_debug_record_stream_round_trips() -> None:
    km = pytest.importorskip("kickmix")
    assert km.__version__ == "0.1" and hasattr(km, "Simulator")
    assert list(Path(km.__file__).parent.glob("*.so"))
    text = _DECLARATIONS + _HEADER + _SOURCE
    program, registers, calls, records = decode_kmx_debug(text)
    assert registers == (KmxRegister(members=(("q", 0), ("q", 1), ("b", 2))),)
    assert not calls
    assert records[:3] == (
        KmxDebugRecord(0, ("q", 1)),
        KmxDebugRecord(0, ("q", 0)),
        KmxDebugRecord(0),
    )
    assert records[6] == KmxDebugRecord(0, condition=0)
    assert records[7:] == (
        KmxDebugRecord(1, ("b", 3)),
        KmxDebugRecord(2, ("b", 3)),
        KmxDebugRecord(2, ("b", 2)),
    )
    assert [operation.opcode for operation in program.operations] == [
        "bit_store1",
        "bit_store0",
        "bit_store0",
        "push_condition",
        "bit_store1",
        "pop_condition",
    ]
    assert encode_kmx_debug(program, registers, records) == text
    assert str(km.Circuit(text)).strip() == text.strip()


def test_debug_positions_are_distinct_across_empty_marked_block() -> None:
    content = (
        "DEBUG_PRINT q1\n"
        'INFO_TAG "@enter:empty"\n'
        "DEBUG_PRINT if b0\n"
        'INFO_TAG "@exit:empty"\n'
        "DEBUG_PRINT b1\n"
        "X q0\n"
        "DEBUG_PRINT\n"
    )
    text = _HEADER + content
    program, registers, calls, records = decode_kmx_debug(text)
    assert calls == (ComponentCall("empty", 0, 0),)
    assert records == (
        KmxDebugRecord(0, ("q", 1)),
        KmxDebugRecord(1, condition=0),
        KmxDebugRecord(2, ("b", 1)),
        KmxDebugRecord(3),
    )
    assert encode_kmx_debug(program, registers, records, calls=calls) == text


@pytest.mark.parametrize(
    "record,message",
    [
        ("DEBUG_PRINT q2", "operand is out of range"),
        ("DEBUG_PRINT b4", "operand is out of range"),
        ("DEBUG_PRINT if b4", "condition is out of range"),
        ("DEBUG_PRINT r0", "Invalid physical KMX debug record"),
    ],
)
def test_invalid_debug_records_are_refused(record: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        decode_kmx_debug(_HEADER + record)


def test_debug_record_bounds_and_header_order() -> None:
    with pytest.raises(ValueError, match="precedes the version header"):
        decode_kmx_debug("DEBUG_PRINT\n" + _HEADER)
    program = decode_kmx(_HEADER + "X q0\n")
    for records in ((KmxDebugRecord(2),), (KmxDebugRecord(1), KmxDebugRecord(0))):
        with pytest.raises(ValueError, match="positions must be ordered"):
            encode_kmx_debug(program, (), records)
