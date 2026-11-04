"""Check the pinned Bell preparation example and its coherent sign."""

import hashlib
import math
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/prepare_bell_state.qmod"
)


def test_pinned_bell_original() -> None:
    raw = SOURCE.read_bytes()
    blob = f"blob {len(raw)}\0".encode() + raw
    assert hashlib.sha1(blob).hexdigest() == "a848085bc035d413768249b115b2707152833a0d"
    program = synth.synthesize(SOURCE)
    amplitude = 1 / math.sqrt(2)
    check_state(program, fq.run(program), {(1,): amplitude, (2,): amplitude})
    assert program.report["workspace_qubits"] == 0


def test_unverified_bell_variant_is_refused() -> None:
    source = SOURCE.read_text().replace("state(2,", "state(1,")
    with pytest.raises(InteropConversionError, match="documented Bell state 2"):
        synth.synthesize(source, source_format="qmod")
