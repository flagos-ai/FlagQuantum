"""Verify indexed array gates and the pinned entanglement original."""

import hashlib
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
SOURCE = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/entanglement.qmod"


def test_original_entanglement_matches_independent_circuit() -> None:
    raw = SOURCE.read_bytes()
    blob = f"blob {len(raw)}\0".encode() + raw
    assert hashlib.sha1(blob).hexdigest() == "05d9776aacf23ebac13fea6b805240618cd00146"
    program = synth.synthesize(SOURCE)
    reference = fq.Circuit(6, dtype=torch.complex128).h(5)
    for control in range(5, 0, -1):
        reference.cx(control, control - 1)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


def test_index_selects_the_low_bit_and_accepts_distinct_same_array_operands():
    source = """qfunc main(output q:qbit[]) {
        allocate(3,q); X(q[0]); CX(q[0],q[2]);
    }"""
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(3, dtype=torch.complex128).x(2).cx(2, 0)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("body", "diagnostic"),
    [
        ("X(q[3]);", "out of bounds"),
        ("X(q[0-1]);", "out of bounds"),
        ("CX(q[1],q[1]);", "distinct"),
        ("repeat(i:257) { X(q[0]); }", "repeat count"),
        ("repeat(i:q) { X(q[0]); }", "static integer"),
        ("repeat(i:1) { X(q[i]); } X(q[i]);", "static integer"),
        ("repeat(i:0) { allocate(3,q); }", "only gates"),
    ],
)
def test_static_repeat_and_index_reject_invalid_source(body, diagnostic):
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(
            f"qfunc main(output q:qbit[]) {{ allocate(3,q); {body} }}",
            source_format="qmod",
        )


def test_repeat_respects_total_effect_budget():
    gates = "X(q[0]);" * 17
    source = (
        f"qfunc main(output q:qbit[]) {{ allocate(1,q); repeat(i:256) {{ {gates} }} }}"
    )
    with pytest.raises(InteropConversionError, match="4096 effect operations"):
        synth.synthesize(source, source_format="qmod")
