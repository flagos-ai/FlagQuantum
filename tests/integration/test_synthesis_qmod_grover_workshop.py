"""Check original multiword Grover phase kickback and clean scratch."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
DTYPE = torch.complex128


def original() -> Path:
    manifest = json.loads((FIXTURES / "grover_workshop_sources.json").read_text())
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
    return path


def reference() -> fq.Circuit:
    circuit = fq.Circuit(8, dtype=DTYPE)
    scratch = 7

    def kickback(marked: Callable[[int], bool]) -> None:
        # Use a reversible truth-table permutation and a real scratch wire.
        matrix = torch.zeros((256, 256), dtype=DTYPE)
        for basis in range(128):
            for bit in range(2):
                matrix[2 * basis + (bit ^ int(marked(basis))), 2 * basis + bit] = 1
        circuit.x(scratch).h(scratch)
        circuit.any(*range(8), matrix=matrix)
        circuit.h(scratch).x(scratch)

    for wire in range(7):
        circuit.h(wire)
    kickback(lambda basis: (basis & 7) == 2 * (basis >> 5) + ((basis >> 3) & 3))
    for wire in range(7):
        circuit.h(wire)
    kickback(lambda basis: basis != 0)
    for wire in range(7):
        circuit.h(wire)
    return circuit


def test_original_grover_workshop_full_amplitudes_and_no_scratch_escape():
    program = synth.synthesize(original())
    expected = fq.run(reference()).statevector().reshape(128, 2)
    actual = fq.run(program).statevector().flatten()
    assert dict(program.registers) == {"a": (0, 1), "b": (2, 3), "c": (4, 5, 6)}
    assert program.report["workspace_qubits"] == 0
    torch.testing.assert_close(actual, expected[:, 0], atol=1e-11, rtol=0)
    torch.testing.assert_close(
        expected[:, 1], torch.zeros_like(expected[:, 1]), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    ("old", "new", "diagnostic"),
    [
        ("H(x);", "Z(x);", "cannot prove annotated phase-kickback"),
        ("res ^= c ==", "res ^= res ==", "cannot prove annotated phase-kickback"),
        ("oracle_function(a, b, c);", "oracle_function(a, a, c);", "disjoint"),
        (
            "prepare_minus_state(aux);\n  }",
            "X(aux);\n  }",
            "cannot prove annotated phase-kickback",
        ),
        ("res ^= c ==", "X(a); res ^= c ==", "cannot prove annotated phase-kickback"),
    ],
)
def test_unproved_kickback_or_alias_is_refused(old, new, diagnostic):
    source = original().read_text()
    assert old in source
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source.replace(old, new, 1), source_format="qmod")
