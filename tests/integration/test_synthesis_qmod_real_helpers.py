"""Compare bound Qmod helpers with independently wired Circuits."""

import hashlib
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.ecosystem import InteropConversionError
from flagquantum.services.synthesis import _synthesize

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
DTYPE = torch.complex128


def _posture(values):
    reference = fq.Circuit(2, dtype=DTYPE)
    for wire, offset in ((1, 0), (0, 3), (0, 0)):
        reference.rz(wire, theta=values[offset + 2])
        reference.ry(wire, theta=values[offset + 1])
        reference.rx(wire, theta=values[offset])
    return reference


def _drude(values):
    reference = fq.Circuit(6, dtype=DTYPE)
    theta = 1.570796326794897
    pairs = ((5, 4), (3, 2), (1, 0), (4, 3), (2, 1)) * 2
    for offset, (a, b) in enumerate(pairs):
        p, q, r, s = values[4 * offset : 4 * offset + 4]
        reference.rz(a, theta=theta).rz(b, theta=theta).ry(b, theta=theta)
        reference.cx(b, a).ry(a, theta=p).rz(a, theta=q)
        reference.ry(b, theta=r).rz(b, theta=s).cx(b, a)
        reference.ry(b, theta=-theta).rz(a, theta=-theta).rz(b, theta=-theta)
    return reference


def _vqls(values):
    reference = fq.Circuit(3, dtype=DTYPE)
    links = (((2, 1), (2, 0), (1, 0)), ((0, 2), (0, 1), (1, 2)), ())
    for layer, pairs in enumerate(links):
        for index, wire in enumerate((2, 1, 0)):
            reference.ry(wire, theta=values[layer * 3 + index])
        for a, b in pairs:
            reference.cz(a, b)
    return reference


@pytest.mark.parametrize("scale", [0.02, -0.031])
@pytest.mark.parametrize(
    ("name", "blob", "binding", "size", "oracle"),
    [
        (
            "robust_posture_optimization",
            "60519b0b320701cc75d90fb75cd529d9f47645c1",
            "theta",
            6,
            _posture,
        ),
        (
            "quantum_drude_oscillator",
            "3a0aa8228232a15814868cbf0595f2f45de0cac3",
            "params",
            40,
            _drude,
        ),
        (
            "vqls_with_lcu",
            "5cf9bac60fb7ac51c02147e1bbe79b8699c643ac",
            "params",
            9,
            _vqls,
        ),
    ],
)
def test_real_helper_original(name, blob, binding, size, oracle, scale):
    source = FIXTURES / f"{name}.qmod"
    raw = source.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
    values = [scale * (index + 1) for index in range(size)]
    program = _synthesize(source, classical_inputs={binding: values})
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(oracle(values)).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    ("actuals", "diagnostic"),
    [
        ("data[0:3], {q[0],q[1]}, q[2]", "slice is out of bounds"),
        ("data[0:1], {q[0],q[1]}, q[2]", "real array length"),
        ("data, {q[0]}, q[2]", "quantum array length"),
        ("data, {q[0],q[0]}, q[2]", "must be disjoint"),
        ("data, {q[0],q[1]}, q[1]", "must be disjoint"),
        ("data, {q[0],q[3]}, q[2]", "array index is out of bounds"),
        ("[data[0],data[2]], {q[0],q[1]}, q[2]", "real array index is out of bounds"),
    ],
)
def test_invalid_helper_views(actuals, diagnostic):
    with pytest.raises(InteropConversionError, match=diagnostic):
        _synthesize(
            "qfunc layer(p:real[2], a:qbit[2], b:qbit) { RY(p[0],a[0]); CX(a[1],b); }"
            f"qfunc main(data:real[2], output q:qbit[3]) {{allocate(q); layer({actuals});}}",
            source_format="qmod",
            classical_inputs={"data": [0.2, -0.3]},
        )


def test_view_order_survives_nested_helper_and_pair_gate():
    source = """
        qfunc kick(p:real[], a:qbit, b:qbit) { RY(p[0],a); CX(a,b); }
        qfunc layer(p:real[], a:qbit[]) { kick(p,a[0],a[1]); }
        qfunc main(p:real[1],output q:qbit[3]) {
          allocate(q); layer(p,{q[2],q[0]});
        }
    """
    program = _synthesize(source, source_format="qmod", classical_inputs={"p": [0.31]})
    reference = fq.Circuit(3, dtype=DTYPE).ry(0, theta=0.31).cx(0, 2)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
