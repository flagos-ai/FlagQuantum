"""Verify numeric equality across signed and unsigned encodings."""

import hashlib
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/comparator_2vars_example.qmod"
)


def test_original_signed_comparator_full_state() -> None:
    raw = SOURCE.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "861f5f78c8747b00813b83c33d1ae5782286e39e"
    )
    program = synth.synthesize(SOURCE)
    reference = fq.Circuit(program.circuit.n_wires, dtype=torch.complex128)
    reference.x(8)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("relation", ["==", "!="])
@pytest.mark.parametrize(("signed_width", "unsigned_width"), [(5, 3), (3, 5)])
def test_mixed_sign_all_inputs_preserve_phase(relation, signed_width, unsigned_width):
    source = f"""qfunc main(output a:qnum, output b:qnum, output res:qnum) {{
        allocate({signed_width}, SIGNED, 0, a);
        allocate({unsigned_width}, UNSIGNED, 0, b);
        hadamard_transform(a); hadamard_transform(b);
        phase(b == 7, pi);
        res = a {relation} b;
    }}"""
    program = synth.synthesize(source, source_format="qmod")
    sign = 1 << (signed_width - 1)
    expected = {}
    for raw in range(1 << signed_width):
        signed = raw - (2 * sign) if raw & sign else raw
        for b in range(1 << unsigned_width):
            bit = int((signed == b) == (relation == "=="))
            expected[raw, b, bit] = (-1 if b == 7 else 1) / 16
    check_state(program, fq.run(program), expected)


@pytest.mark.parametrize(("expression", "matching"), [("a == -1", 3), ("a == 3", None)])
def test_signed_word_and_constant(expression, matching):
    program = synth.synthesize(
        f"""qfunc main(output a:qnum, output res:qnum) {{
            allocate(2, SIGNED, 0, a); hadamard_transform(a); res = {expression};
        }}""",
        source_format="qmod",
    )
    check_state(
        program, fq.run(program), {(a, int(a == matching)): 0.5 for a in range(4)}
    )


def test_signed_order_comparison_remains_outside_profile():
    with pytest.raises(InteropConversionError, match="signed operands"):
        synth.synthesize(
            "qfunc main(output a:qnum, output res:qnum) {allocate(2,SIGNED,0,a); res=a<0;}",
            source_format="qmod",
        )
