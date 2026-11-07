"""Check original Qmod differences against independent Circuit oracles."""

import hashlib
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


@pytest.mark.parametrize(
    ("name", "blob", "widths", "signed", "fractional"),
    [
        (
            "negation_example",
            "e6b8030e522ec664c34108fe8544829eef05ec89",
            (3, 4),
            ("a", "b"),
            {},
        ),
        (
            "subtraction_2vars_example",
            "5ba63d99f072d7af128c0fe89744f38c256b4ea5",
            (3, 3, 4),
            ("res",),
            {},
        ),
        (
            "subtraction_float_example",
            "ba263fb49122a235a1c0d9f992feb384be828b39",
            (3, 4),
            ("res",),
            {"res": 1},
        ),
    ],
)
def test_original_subtraction_full_state(name, blob, widths, signed, fractional):
    path = FIXTURES / f"{name}.qmod"
    raw = path.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
    program = synth.synthesize(path, memory_limit_bytes=512 * 1024**2)
    assert tuple(map(len, program.registers.values())) == widths
    assert program.report.get("qmod_signed_registers", ()) == signed
    assert program.report.get("qmod_fractional_bits", {}) == fractional

    reference = fq.Circuit(program.circuit.n_wires, dtype=torch.complex128)
    if name == "negation_example":
        matrix = torch.zeros((128, 128), dtype=torch.complex128)
        for a in range(8):
            reference_value = a if a < 4 else a - 8
            for b in range(16):
                matrix[16 * a + (b ^ (-reference_value & 15)), 16 * a + b] = 1
        for wire in range(3):
            reference.h(wire)
        reference.any(*range(7), matrix=matrix)
    else:
        expected = (4, 5, 15) if name == "subtraction_2vars_example" else (4, 1)
        bits = "".join(
            f"{value:0{width}b}" for value, width in zip(expected, widths, strict=True)
        )
        for wire, bit in enumerate(bits):
            if bit == "1":
                reference.x(wire)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("expression", "width", "oracle"),
    [
        ("a-b", 3, lambda a, b: a - b),
        ("a-a", 3, lambda a, b: 0),
        ("(-1)-a", 3, lambda a, b: -1 - a),
        ("a - b - 0.5", 4, lambda a, b: 2 * (a - b) - 1),
    ],
)
def test_coherent_subtraction(expression, width, oracle):
    source = f"""qfunc main(output a:qnum<2>, output b:qnum<2>, output res:qnum) {{
        allocate(a); hadamard_transform(a); phase(a>1,pi);
        allocate(b); hadamard_transform(b); phase(b>1,pi);
        res = {expression};
    }}"""
    program = synth.synthesize(source, source_format="qmod")
    assert len(program.registers["res"]) == width
    check_state(
        program,
        fq.run(program),
        {
            (a, b, oracle(a, b) & ((1 << width) - 1)): (-1) ** ((a > 1) + (b > 1)) / 4
            for a in range(4)
            for b in range(4)
        },
    )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("allocate(a); res=a-0.1;", "exactly binary representable"),
        ("res=a-a;", "uninitialized"),
        ("allocate(a); res=a-0.5; H(res);", "terminal output"),
    ],
)
def test_invalid_subtraction(body, message):
    source = f"qfunc main(output a:qnum<2>, output res:qnum) {{{body}}}"
    with pytest.raises(InteropConversionError, match=message):
        synth.synthesize(source, source_format="qmod")
