"""Verify exact fixed-point multiplication in the original Qmod example."""

import hashlib
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from tests.synthesis_workflow_helpers import check_state

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/multiplication_float_example.qmod"
)


def test_original_fixed_multiply_full_state_and_named_readout() -> None:
    raw = SOURCE.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "6020faf8a6c7601558b5f1bf42900ff5bc0f677b"
    )
    program = synth.synthesize(SOURCE)
    assert tuple(map(len, program.registers.values())) == (2, 5)
    matrix = torch.zeros((128, 128), dtype=torch.complex128)
    for a in range(4):
        for b in range(32):
            matrix[32 * a + (b ^ (7 * a)), 32 * a + b] = 1
    reference = fq.Circuit(program.circuit.n_wires, dtype=torch.complex128)
    reference.h(0).h(1)
    reference.any(*range(7), matrix=matrix)
    result = fq.run(program)
    torch.testing.assert_close(
        result.statevector(), fq.run(reference).statevector(), atol=1e-12, rtol=0
    )
    assert program.report["qmod_fractional_bits"] == {"res": 1}
    scale = 1 << program.report["qmod_fractional_bits"]["res"]
    decoded = {
        (a, raw / scale): probability
        for (a, raw), probability in program.probabilities(
            result, registers=("a", "res")
        ).items()
    }
    assert decoded == pytest.approx({(a, 3.5 * a): 0.25 for a in range(4)})


@pytest.mark.parametrize(
    ("expression", "fractional_bits", "numerator"),
    [("a * 3.5", 1, 7), ("(a * 0.5) * 1.25", 3, 5), ("a * 2.0", 0, 2)],
)
def test_fixed_multiply_preserves_relative_phase(
    expression, fractional_bits, numerator
):
    program = synth.synthesize(
        f"""qfunc main(output a:qnum, output res:qnum) {{
            allocate(2,a); hadamard_transform(a); phase(a==2,pi);
            res={expression};
        }}""",
        source_format="qmod",
    )
    assert (
        program.report.get("qmod_fractional_bits", {}).get("res", 0) == fractional_bits
    )
    check_state(
        program,
        fq.run(program),
        {(a, numerator * a): -0.5 if a == 2 else 0.5 for a in range(4)},
    )


@pytest.mark.parametrize(
    ("setup", "expression", "diagnostic"),
    [
        ("allocate(2,a);", "0.1 * a", "exactly binary representable"),
        ("allocate(2,SIGNED,0,a);", "3.5 * a", "signed operands"),
    ],
)
def test_fixed_multiply_refusals(setup, expression, diagnostic):
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(
            f"qfunc main(output a:qnum, output res:qnum) {{{setup} res={expression};}}",
            source_format="qmod",
        )
