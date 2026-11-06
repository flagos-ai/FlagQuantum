"""Check the pinned Qmod GHZ original against its analytic amplitudes."""

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
    Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/prepare_ghz_state.qmod"
)


def test_pinned_ghz_original() -> None:
    raw = SOURCE.read_bytes()
    blob = f"blob {len(raw)}\0".encode() + raw
    assert hashlib.sha1(blob).hexdigest() == "e8666b8b49d1de36de614a8f2897afb869cbdbe7"
    program = synth.synthesize(SOURCE)
    check_state(
        program, fq.run(program), {(0,): 1 / math.sqrt(2), (31,): 1 / math.sqrt(2)}
    )
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    ("declaration", "statements"),
    [
        ("qbit[]", "prepare_ghz_state(0,q);"),
        ("qnum", "prepare_ghz_state(2,q);"),
        ("qbit[]", "allocate(2,q); prepare_ghz_state(2,q);"),
    ],
)
def test_ghz_requires_fresh_array_and_valid_size(declaration, statements) -> None:
    with pytest.raises(InteropConversionError):
        synth.synthesize(
            f"qfunc main(output q:{declaration}) {{ {statements} }}",
            source_format="qmod",
        )
