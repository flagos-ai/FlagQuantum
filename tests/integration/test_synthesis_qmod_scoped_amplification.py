"""Test the borrowed exact-amplification proof and clean workspace."""

import math

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
_DTYPE = torch.complex128
_SOURCE = """
qfunc entangle(p:qbit[2]) { H(p[0]); S(p[0]); CX(p[0], p[1]); }
qfunc improve(q:qbit[2]) {
  exact_amplitude_amplification(AMPLITUDE, lambda(o) {Z(o[1]);},
    lambda(p) {entangle(p);}, q);
}
qfunc main(output q:qbit[2]) {allocate(q); improve(q);}
"""


def _source() -> str:
    return _SOURCE.replace("AMPLITUDE", repr(math.sqrt(0.5)))


def test_scoped_exact_amplification_preserves_the_marked_phase() -> None:
    program = synth.synthesize(_source(), source_format="qmod", max_workspace=1)
    assert program.report["workspace_qubits"] == 1
    reference = fq.Circuit(3, dtype=_DTYPE).x(0).x(1).s(1)
    actual = fq.run(program).statevector()
    torch.testing.assert_close(
        actual, fq.run(reference).statevector(), atol=1e-12, rtol=0
    )
    assert actual.reshape(4, 2)[:, 1].abs().max() < 1e-12
    with pytest.raises(ValueError, match="workspace|budget"):
        synth.synthesize(_source(), source_format="qmod", max_workspace=0)


def test_scoped_exact_amplification_remaps_each_borrowed_workspace() -> None:
    source = _source().replace(
        "qfunc main(output q:qbit[2]) {allocate(q); improve(q);}",
        """qfunc delegated(q:qbit[2]) {improve(q);}
        qfunc main(output a:qbit[2], output b:qbit[2]) {
            allocate(a); allocate(b); delegated(a); delegated(b);
        }""",
    )
    program = synth.synthesize(source, source_format="qmod", max_workspace=2)
    assert program.report["workspace_qubits"] == 2
    reference = fq.Circuit(6, dtype=_DTYPE).x(0).x(1).x(2).x(3).s(1).s(3)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "before,after,diagnostic",
    [
        ("AMPLITUDE", "0.6", "disagrees with proven amplitude"),
        (
            "allocate(q); improve(q);",
            "allocate(q); X(q[0]); improve(q);",
            "proven zero entry",
        ),
        ("Z(o[1]);", "X(o[1]);", "oracle must implement indexed basis signs"),
        ("Z(o[1]);", "phase(0.3);", "oracle must apply signs"),
        ("}, q);", "}, {q[0], q[0]});", "distinct bits"),
        ("entangle(p);", "allocate(p);", "bound allocation requires a local qbit"),
    ],
)
def test_scoped_exact_amplification_refuses_invalid_programs(
    before: str, after: str, diagnostic: str
) -> None:
    source = _SOURCE.replace(before, after).replace("AMPLITUDE", repr(math.sqrt(0.5)))
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")


def test_scoped_exact_amplification_rejects_external_quantum_capture() -> None:
    source = _source().replace("improve(q:qbit[2])", "improve(q:qbit[2], other:qbit)")
    source = source.replace("entangle(p);", "entangle(p); X(other);")
    source = source.replace(
        "qfunc main(output q:qbit[2]) {allocate(q); improve(q);}",
        "qfunc main(output q:qbit[2], output flag:qbit) {allocate(q); allocate(flag); improve(q, flag);}",
    )
    with pytest.raises(InteropConversionError, match="outside the packed target"):
        synth.synthesize(source, source_format="qmod")
