"""Compare private feedback with the exact dynamic measurement channel."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
_DTYPE = torch.complex128
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


def _original() -> Path:
    manifest = json.loads((_FIXTURES / "teleportation_sources.json").read_text())
    [row] = manifest["fixtures"]
    path = _FIXTURES / row["file"]
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == row["git_blob_sha1"]
    )
    license_bytes = (_FIXTURES / manifest["license_file"]).read_bytes()
    assert hashlib.sha256(license_bytes).hexdigest() == manifest["license_sha256"]
    return path


@pytest.mark.parametrize("angles", [None, (0.43, -0.71), (-1.1, 0.28)])
def test_original_teleportation_output_density(
    angles: tuple[float, float] | None,
) -> None:
    path = _original()
    # Output wire 0 is Bob. Alice's input and Bell partner are wires 1 and 2.
    dynamic = DynamicCircuit(3, dtype=_DTYPE)
    if angles is None:
        program = synth.synthesize(path)
        dynamic.h(1)
    else:
        theta, phi = angles
        source = path.read_text().replace("H(q);", f"RY({theta},q); RZ({phi},q);", 1)
        program = synth.synthesize(source, source_format="qmod")
        dynamic.ry(1, theta=theta).rz(1, theta=phi)
    dynamic.h(2).cx(2, 0).cx(1, 2).h(1)
    dynamic.measure(1, classical_bit=0).measure(2, classical_bit=1)
    dynamic.conditional("x", 0, classical_bit=1).conditional("z", 0, classical_bit=0)
    expected = torch.einsum("aibi->ab", dynamic.density_matrix()[0].reshape(2, 4, 2, 4))

    assert program.registers == {"bob_qubit": (0,)}
    assert program.report["qmod_measurement_profile"] == "deferred_private_measurements"
    result = fq.run(program)
    amplitudes = result.statevector().reshape(2, 4)
    actual = amplitudes @ amplitudes.conj().T
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    probabilities = program.probabilities(result, registers=("bob_qubit",))
    for word, probability in probabilities.items():
        assert probability == pytest.approx(actual[word[0], word[0]].real.item())


def test_private_measurement_can_leave_a_mixed_output() -> None:
    source = """
        qfunc feedback(a:qbit,b:qbit) {
            flag:bool; flag=measure(a); if(flag) { X(b); }
        }
        qfunc main(output b:qbit) {
            a:qbit; allocate(a); allocate(b); H(a); feedback(a,b); drop(a);
        }
    """
    program = synth.synthesize(source, source_format="qmod")
    amplitudes = fq.run(program).statevector().reshape(2, 2)
    torch.testing.assert_close(
        amplitudes @ amplitudes.conj().T,
        torch.eye(2, dtype=_DTYPE) / 2,
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("before", "after", "diagnostic"),
    [
        ("drop(alice_qubit);", "", "must be immediately dropped"),
        ("X(bob_qubit);", "X(alice_qubit);", "disjoint unmeasured port"),
        ("if (m2)", "if (unused)", "initialized measurement flag"),
        (
            "alice_qubit, alice_bell, bob_qubit);",
            "alice_qubit, alice_qubit, bob_qubit);",
            "distinct scalar actuals",
        ),
        (
            "bob_qubit: qbit) {\n  alice_qubit:",
            "bob_qubit: qnum<1>) {\n  alice_qubit:",
            "distinct scalar actuals",
        ),
    ],
)
def test_private_feedback_rejects_observable_or_invalid_measurement(
    before: str, after: str, diagnostic: str
) -> None:
    source = _original().read_text()
    assert before in source
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source.replace(before, after, 1), source_format="qmod")
