"""Validate original bitwise programs and coherent register inputs."""

import hashlib
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from tests.synthesis_workflow_helpers import check_state

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
        (
            "bitwise_or_2vars_example",
            "620705df415ca6e09fe5ec3b92df176b3f54ac69",
            (5, 3, 5),
            (4, 5, 5),
        ),
        (
            "bitwise_xor_2vars_example",
            "e9247706ed18cce9a048f2b046b8752c924643e1",
            (5, 3, 5),
            (4, 5, 1),
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


def test_signed_bitwise_extends_negative_superposition_against_circuit():
    source = """qfunc main(output a:qnum, output b:qnum, output res:qnum) {
        allocate(2, SIGNED, 0, a);
        allocate(3, UNSIGNED, 0, b);
        hadamard_transform(a);
        b ^= 4;
        res = a ^ b;
    }"""
    program = synth.synthesize(source, source_format="qmod")
    assert tuple(map(len, program.registers.values())) == (2, 3, 4)
    reference = fq.Circuit(
        9 + program.report["workspace_qubits"], dtype=torch.complex128
    )
    # a's sign controls three output bits. b's high bit flips result bit 2.
    reference.h(0).h(1).x(2).cx(0, 5).cx(0, 6).cx(0, 7).cx(1, 8).cx(2, 6)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


def test_xor_update_sign_extends_distinct_source_and_is_reversible():
    source = """qfunc main(output a:qnum, output b:qnum, output c:qnum) {
        allocate(2, True, 0, a);
        allocate(4, False, 0, b);
        allocate(4, UNSIGNED, 0, c);
        a ^= -1;
        hadamard_transform(a);
        b ^= a;
        c ^= a;
        c ^= a;
    }"""
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(10, dtype=torch.complex128)
    reference.x(0).x(1).h(0).h(1).cx(0, 2).cx(0, 3).cx(0, 4).cx(1, 5)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("body", "diagnostic"),
    [
        ("allocate(2,SIGNED,1,a); allocate(2,b);", "zero fractional bits"),
        ("allocate(2,MAYBE,0,a); allocate(2,b);", "literal signedness"),
        ("allocate(2,a); allocate(2,b); a ^= 4;", "fit the target width"),
        ("allocate(2,a); allocate(2,b); a ^= -1;", "fit the signed target"),
        ("allocate(2,a); allocate(2,b); a ^= a;", "distinct integer"),
        ("allocate(2,a); b ^= 0;", "uninitialized"),
        (
            "allocate(2,SIGNED,0,a); b = a + 1;",
            "require bitwise operations, subtraction or negation",
        ),
    ],
)
def test_typed_allocate_and_xor_refuse_unsupported_semantics(body, diagnostic):
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(
            f"qfunc main(output a:qnum<2>, output b:qnum<2>) {{ {body} }}",
            source_format="qmod",
        )
