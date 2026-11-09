"""Verify expanded QSVT, projected rotations, and quantum Const borrowing."""

import cmath
import hashlib
import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
DTYPE = torch.complex128


def original() -> tuple[Path, dict[str, str]]:
    manifest = json.loads((FIXTURES / "qsvt_expanded_sources.json").read_text())
    [row] = manifest["fixtures"]
    path = FIXTURES / row["file"]
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == row["git_blob_sha1"]
    )
    assert (
        hashlib.sha256((FIXTURES / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )
    return path, row


def reference() -> fq.Circuit:
    circuit = fq.Circuit(10, dtype=DTYPE)
    state, ref, indicator, aux = tuple(range(4)), tuple(range(4, 8)), 8, 9
    comparator = torch.zeros((512, 512), dtype=DTYPE)
    for s in range(16):
        for r in range(16):
            for bit in range(2):
                column = 32 * s + 2 * r + bit
                comparator[column ^ int(s <= r), column] = 1

    def block(inverse: bool = False) -> None:
        if inverse:
            circuit.any(*state, *ref, indicator, matrix=comparator)
        for wire in ref:
            circuit.h(wire)
        if not inverse:
            circuit.any(*state, *ref, indicator, matrix=comparator)

    def project(angle: float, bits: tuple[int, ...]) -> None:
        phases = []
        for value in range(1 << len(bits)):
            for marker in range(2):
                toggled = marker ^ int(value == 0)
                phases.append(cmath.exp(1j * angle * (toggled - 0.5)))
        circuit.any(*bits, aux, matrix=torch.diag(torch.tensor(phases, dtype=DTYPE)))

    for wire in state:
        circuit.h(wire)
    circuit.h(aux)
    project(-2.401688306254694, ref)
    for odd, even in (
        (2.669374314067509, 3.817567704355659),
        (3.817567704355659, 2.669374314067509),
    ):
        block()
        project(odd, (indicator,))
        block(inverse=True)
        project(even, ref)
    block()
    project(-14.968058920613865, (indicator,))
    circuit.h(aux)
    return circuit


def test_original_expanded_qsvt_full_state_and_clean_workspace() -> None:
    path, _ = original()
    program = synth.synthesize(path)
    actual = fq.run(program).statevector().reshape(1024, -1)
    expected = fq.run(reference()).statevector().flatten()
    assert actual.shape[1] == 8
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-11, rtol=0)
    torch.testing.assert_close(
        actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-12, rtol=0
    )


PROJECTOR = """
qperm pred(m:qbit, const q:qbit) {m ^= not q;}
qperm rotate(angle:real,const m:qbit,const q:qbit) {
    within {pred(m,q);} apply {RZ(angle,m);}
}
"""


@pytest.mark.parametrize("marker", [0, 1])
def test_projector_keeps_absolute_phase_for_either_marker(marker: int) -> None:
    source = f"""
        qfunc main(output q:qbit,output m:qbit) {{
            allocate(q); allocate(m); H(q);
            if (False) {{X(q);}} else {{ {"X(m);" if marker else ""} }}
            rotate(0.81,m,q);
        }}
    """
    source = PROJECTOR + source
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(2, dtype=DTYPE).h(0)
    if marker:
        reference.x(1)
    reference.x(0).cnot(0, 1).rz(1, theta=0.81).cnot(0, 1).x(0)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("definition", "call", "message"),
    [
        (PROJECTOR, "rotate(0.81,q,q);", "must be disjoint"),
        (
            "qfunc flip(q:qbit) {X(q);} qfunc read(const q:qbit,m:qbit) {flip(q);}",
            "read(q,m);",
            "cannot write a Const quantum port",
        ),
        (
            PROJECTOR.replace("qperm rotate", "qfunc rotate").replace(
                "RZ(angle,m)", "RX(angle,m)"
            ),
            "rotate(0.81,m,q);",
            "writable initialized",
        ),
    ],
)
def test_projector_and_const_borrow_reject_alias_and_writes(
    definition: str, call: str, message: str
) -> None:
    source = (
        definition
        + f"qfunc main(output q:qbit,output m:qbit) {{allocate(q);allocate(m);{call}}}"
    )
    with pytest.raises(InteropConversionError, match=message):
        synth.synthesize(source, source_format="qmod")


def test_expanded_qsvt_refuses_insufficient_workspace_budget() -> None:
    path, _ = original()
    with pytest.raises(synth.SynthesisError, match="workspace"):
        synth.synthesize(path, max_workspace=2)
