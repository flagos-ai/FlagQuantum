"""Compare original extrema programs with independently derived circuits."""

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
    ("name", "blob", "widths", "fractional_bits"),
    [
        (
            "minimum_2vars_example",
            "d75c7c814f0036f8d2f4eb3a79ae85bd153d73cb",
            (3, 3, 3),
            {},
        ),
        (
            "maximum_float_example",
            "3fb6caa2b94624cac6022ca0fc9e6566f8144298",
            (3, 4),
            {"res": 1},
        ),
        (
            "maximum_integer_example",
            "3fb6caa2b94624cac6022ca0fc9e6566f8144298",
            (3, 4),
            {"res": 1},
        ),
        (
            "arithmetic_expressions",
            "77672bda8b2754f9a8102d988b6a5ca2a8cec420",
            (2, 1, 1),
            {},
        ),
    ],
)
def test_original_extrema_full_state(name, blob, widths, fractional_bits):
    path = FIXTURES / f"{name}.qmod"
    raw = path.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
    program = synth.synthesize(path, memory_limit_bytes=512 * 1024**2)
    assert tuple(map(len, program.registers.values())) == widths
    assert program.report.get("qmod_fractional_bits", {}) == fractional_bits
    reference = fq.Circuit(program.circuit.n_wires, dtype=torch.complex128)
    if name.startswith("minimum"):
        reference.x(0)
        for wire in (3, 4, 5):
            reference.h(wire)
        # min(4,b): keep the high bit, copy lower bits only if it is zero.
        reference.cx(3, 6)
        reference.x(3)
        reference.ccx(3, 4, 7)
        reference.ccx(3, 5, 8)
        reference.x(3)
    elif name.startswith("maximum"):
        for wire in (0, 1, 2):
            reference.h(wire)
        # max(3.5,a) in half units: select 0111 or a shifted left one bit.
        for wire in (4, 5, 6):
            reference.x(wire)
            reference.cx(0, wire)
        reference.cx(0, 3)
        reference.ccx(0, 1, 4)
        reference.ccx(0, 2, 5)
    else:
        for wire in (0, 1, 2):
            reference.h(wire)
        # If y=0, z is x>=2. If y=1, z is x>=1.
        reference.cx(0, 3)
        reference.x(0)
        reference.ccx(0, 1, 4)
        reference.ccx(4, 2, 3)
        reference.ccx(0, 1, 4)
        reference.x(0)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("expression", "oracle"),
    [
        ("min(a,b)", min),
        ("max(a,b)", max),
        ("min(a,1)", lambda a, b: min(a, 1)),
        ("max(a,a)", lambda a, b: a),
        ("max(min(a,1.25),0.5)", lambda a, b: max(min(4 * a, 5), 2)),
    ],
)
def test_extrema_with_relative_phase_and_ties(expression, oracle):
    source = f"""qfunc main(output a:qnum<2>, output b:qbit, output res:qnum) {{
        allocate(a); hadamard_transform(a); phase(a>1,pi);
        allocate(b); H(b); S(b);
        res = {expression};
    }}"""
    program = synth.synthesize(source, source_format="qmod")
    check_state(
        program,
        fq.run(program),
        {
            (a, b, oracle(a, b)): (-1) ** (a > 1) * 1j**b / 8**0.5
            for a in range(4)
            for b in range(2)
        },
    )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("allocate(a); res=max(a,0.1);", "exactly binary representable"),
        ("allocate(a); res=max(a,0.5)+1;", "only in min/max"),
        ("allocate(a); res=max(a,0.5); H(res);", "terminal output"),
        ("res=min(a,0);", "uninitialized"),
        ("allocate(2,True,0,a); res=max(a,0);", "signed operands"),
    ],
)
def test_extrema_reject_unsupported_domains(body, message):
    with pytest.raises(InteropConversionError, match=message):
        synth.synthesize(
            f"qfunc main(output a:qnum<2>, output res:qnum) {{{body}}}",
            source_format="qmod",
        )
