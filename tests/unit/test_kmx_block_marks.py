"""The locked builder's deferred marks in the native KMX format layer."""

from pathlib import Path

import pytest

from flagquantum.compiler._component_trace.trace import ComponentCall
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.formats.kmx_block_marks import (
    decode_kmx_block_marks,
    encode_kmx_block_marks,
)

pytestmark = pytest.mark.unit
_HEADER = 'INFO_TAG "@flagquantum:reversible-phase-v1 wires=2 bits=0"\n'
_GATES = "X q0\nCX q0 q1\n"
_SOURCE = (
    'INFO_TAG "@enter:outer"\n'
    "X q0\n"
    'INFO_TAG "@enter:inner_empty"\n'
    'INFO_TAG "@exit:inner_empty"\n'
    'INFO_TAG "@enter:inner"\n'
    "CX q0 q1\n"
    'INFO_TAG "@exit:inner"\n'
    'INFO_TAG "@exit:outer"\n'
)


@pytest.mark.ftqc_reference
def test_original_nested_mark_fixture_round_trips() -> None:
    km = pytest.importorskip("kickmix")
    assert km.__version__ == "0.1" and hasattr(km, "Simulator")
    assert any(path.suffix == ".so" for path in Path(km.__file__).parent.glob("*.so"))
    assert str(km.Circuit(_SOURCE)).strip() == _SOURCE.strip()
    program, calls = decode_kmx_block_marks(_HEADER + _SOURCE)
    assert calls == (
        ComponentCall(
            "outer",
            0,
            2,
            children=(ComponentCall("inner_empty", 1, 1), ComponentCall("inner", 1, 2)),
        ),
    )
    assert encode_kmx(program) == _HEADER + _GATES
    assert encode_kmx_block_marks(program, calls) == _HEADER + _SOURCE
    for first in (0, 1):
        for second in (0, 1):
            sim = km.Simulator(batch_size=1, seed=1)
            sim.write_across_shots(km.q(0), first)
            sim.write_across_shots(km.q(1), second)
            sim.do(km.Circuit(_SOURCE))
            assert sim.read_across_shots(km.q(0), out=int) == first ^ 1
            assert sim.read_across_shots(km.q(1), out=int) == second ^ first ^ 1
            assert sim.read_shot_phase(0) == 0


@pytest.mark.ftqc_reference
def test_block_mark_label_uses_source_escapes() -> None:
    km = pytest.importorskip("kickmix")
    name = 'qubit#"\\\r\n'
    program = decode_kmx(_HEADER + _GATES)
    calls = (ComponentCall(name, 0, 0), ComponentCall(name, 2, 2))
    encoded = encode_kmx_block_marks(program, calls)
    assert r'INFO_TAG "@enter:qubit\P\Q\B\r\n"' in encoded
    assert str(km.Circuit(encoded)).strip() == encoded.strip()
    assert decode_kmx_block_marks(encoded) == (program, calls)


@pytest.mark.parametrize(
    "marks,message",
    [
        ('INFO_TAG "@exit:outer"', "Unmatched"),
        ('INFO_TAG "@enter:outer"', "Unclosed"),
        ('INFO_TAG "@enter:a"\nINFO_TAG "@exit:b"', "Unmatched"),
        (r'INFO_TAG "@enter:bad\t"', "Invalid"),
        ('INFO_TAG "@enter:"', "Invalid"),
    ],
)
def test_invalid_block_mark_stream_is_refused(marks: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        decode_kmx_block_marks(_HEADER + marks)


@pytest.mark.parametrize(
    "calls,message",
    [
        ((ComponentCall("", 0, 0),), "name"),
        ((ComponentCall("parameters", 0, 0, (1,)),), "parameters"),
        ((ComponentCall("past end", 1, 3),), "range"),
        ((ComponentCall("first", 0, 2), ComponentCall("second", 1, 2)), "range"),
    ],
)
def test_unrepresentable_component_trace_is_refused(
    calls: tuple[ComponentCall, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        encode_kmx_block_marks(decode_kmx(_HEADER + _GATES), calls)
