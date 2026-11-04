"""Validate original unsigned bitwise programs and coherent register inputs."""

import hashlib
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


@pytest.mark.parametrize(
    ("name", "blob_sha", "widths", "values"),
    [
        (
            "bitwise_and_2vars_example",
            "c76a6ecb65c3850ec4efb57a077f2829c24a2c5d",
            (3, 3, 3),
            (4, 5, 4),
        ),
        (
            "bitwise_and_integer_example",
            "3211930ea1f583bea5615d690a1bb12d16564653",
            (3, 2),
            (5, 1),
        ),
        (
            "bitwise_invert_example",
            "f3b4dbdc508f27ff9e59fe37af67b769d8c3c959",
            (3, 3),
            (6, 1),
        ),
        (
            "bitwise_or_integer_example",
            "4f8cd81400abe1033395ccd76e02d52787d397bf",
            (3, 3),
            (4, 7),
        ),
        (
            "bitwise_xor_integer_example",
            "6893bb965f797d60bf4992810702b2d3d343085f",
            (3, 3),
            (4, 7),
        ),
    ],
)
def test_pinned_bitwise_original_matches_basis_preparation_circuit(
    name, blob_sha, widths, values
):
    source = FIXTURES / f"{name}.qmod"
    raw = source.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob_sha

    program = synth.synthesize(source)
    assert tuple(map(len, program.registers.values())) == widths
    reference = fq.Circuit(
        sum(widths) + program.report["workspace_qubits"], dtype=torch.complex128
    )
    offset = 0
    for width, value in zip(widths, values, strict=True):
        for index in range(width):
            if value & (1 << (width - 1 - index)):
                reference.x(offset + index)
        offset += width
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("expression", "oracle"),
    [
        ("x & y", lambda x, y: x & y),
        ("x | y", lambda x, y: x | y),
        ("x ^ y", lambda x, y: x ^ y),
        ("~x", lambda x, y: x ^ 3),
        ("x | x", lambda x, y: x),
        ("x & x", lambda x, y: x),
        ("x ^ x", lambda x, y: 0),
        ("~(x ^ y)", lambda x, y: (x ^ y) ^ 7),
        ("(x & 1) + (y ^ 2)", lambda x, y: (x & 1) + (y ^ 2)),
    ],
)
def test_unsigned_bitwise_preserves_coherence_and_cleans_workspace(expression, oracle):
    source = f"""qfunc main(output x:qnum<2>, output y:qnum<3>, output z:qnum) {{
        allocate(x);
        allocate(y);
        hadamard_transform(x);
        hadamard_transform(y);
        z = {expression};
    }}"""
    program = synth.synthesize(source, source_format="qmod")
    check_state(
        program,
        fq.run(program),
        {(x, y, oracle(x, y)): 1 / (32**0.5) for x in range(4) for y in range(8)},
    )
