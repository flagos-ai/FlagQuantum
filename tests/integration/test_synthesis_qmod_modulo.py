"""Check the original power-of-two modulo programs and coherent arithmetic."""

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
MEMORY_LIMIT = 512 * 1024**2


@pytest.mark.parametrize(
    ("name", "blob_sha", "widths", "values"),
    [
        (
            "modulo_example",
            "5b805a72df4332bb190bedca1d5f05a031bcda13",
            (5, 5, 2),
            (4, 7, 3),
        ),
        (
            "arithmetic_expression",
            "18fe46e9d47a8a825d163bf30188528a6a071441",
            (2, 1, 3, 1),
            (2, 1, 5, 0),
        ),
        (
            "arithmetic_expression_example",
            "18fe46e9d47a8a825d163bf30188528a6a071441",
            (2, 1, 3, 1),
            (2, 1, 5, 0),
        ),
    ],
)
def test_original_modulo_full_state(name, blob_sha, widths, values):
    source = FIXTURES / f"{name}.qmod"
    raw = source.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob_sha
    program = synth.synthesize(source, memory_limit_bytes=MEMORY_LIMIT)
    assert tuple(map(len, program.registers.values())) == widths
    reference = fq.Circuit(
        sum(widths) + program.report["workspace_qubits"], dtype=torch.complex128
    )
    expected = "".join(
        f"{value:0{width}b}" for width, value in zip(widths, values, strict=True)
    )
    for wire, bit in enumerate(expected):
        if bit == "1":
            reference.x(wire)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("expression", "oracle"),
    [
        ("(a+b)%4", lambda a, b: (a + b) % 4),
        ("a + b * a % 4", lambda a, b: a + (b * a) % 4),
        ("(a%8)%2", lambda a, b: a % 2),
        ("a%1", lambda a, b: 0),
        ("((a + b) % 4) == 2", lambda a, b: int((a + b) % 4 == 2)),
    ],
)
def test_modulo_on_superposition(expression, oracle):
    source = f"""qfunc main(output a:qnum<2>, output b:qnum<2>, output res:qnum) {{
        allocate(a);
        allocate(b);
        hadamard_transform(a);
        hadamard_transform(b);
        res = {expression};
    }}"""
    program = synth.synthesize(source, source_format="qmod")
    check_state(
        program,
        fq.run(program),
        {(a, b, oracle(a, b)): 0.25 for a in range(4) for b in range(4)},
    )


@pytest.mark.parametrize("divisor", ["0", "3", "a", "(2+2)"])
def test_modulo_rejects_non_power_of_two_literals(divisor):
    with pytest.raises(InteropConversionError, match="positive power-of-two literal"):
        synth.synthesize(
            f"qfunc main(output a:qnum<2>, output res:qnum) {{allocate(a); res = a % {divisor};}}",
            source_format="qmod",
        )


def test_modulo_one_still_requires_initialized_input():
    with pytest.raises(InteropConversionError, match="uninitialized"):
        synth.synthesize(
            "qfunc main(output a:qnum<2>, output res:qnum) {res = a % 1;}",
            source_format="qmod",
        )
