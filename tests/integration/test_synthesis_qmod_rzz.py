"""Check the pinned Qmod RZZ original and relative ZZ phases."""

import cmath
import hashlib
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from tests.synthesis_workflow_helpers import check_state

pytestmark = pytest.mark.integration
SOURCE = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/RZZ.qmod"


def test_pinned_rzz_original() -> None:
    raw = SOURCE.read_bytes()
    blob = f"blob {len(raw)}\0".encode() + raw
    assert hashlib.sha1(blob).hexdigest() == "3d19d2b567f23800acee8db04dc348d79865f577"
    program = synth.synthesize(SOURCE)
    check_state(program, fq.run(program), {(0,): cmath.exp(-0.5j)})


def test_rzz_preserves_relative_phase() -> None:
    source = SOURCE.read_text().replace(
        "RZZ(1, q);", "hadamard_transform(q); RZZ(1, q);"
    )
    program = synth.synthesize(source, source_format="qmod")
    even, odd = (cmath.exp(sign * 0.5j) / 2 for sign in (-1, 1))
    check_state(
        program, fq.run(program), {(0,): even, (1,): odd, (2,): odd, (3,): even}
    )


def test_rzz_refuses_wrong_array_width() -> None:
    source = SOURCE.read_text().replace("allocate(2,", "allocate(3,")
    with pytest.raises(InteropConversionError, match="two-qubit array"):
        synth.synthesize(source, source_format="qmod")
