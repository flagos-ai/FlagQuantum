"""Absolute-phase checks for the pinned equatorial rotation original."""

import cmath
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from flagquantum.synth import Output, QBit, R, allocate, qfunc
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration
CORPUS = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


@qfunc
def native_rotation(q: Output[QBit]) -> None:
    allocate(q)
    R(1, 2, q)


def test_pinned_original_matches_native_and_circuit() -> None:
    manifest = json.loads((CORPUS / "r_sources.json").read_text())
    item = manifest["fixtures"][0]
    path = CORPUS / item["file"]
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == item["sha256"]
    assert (
        hashlib.sha1(f"blob {len(source)}\0".encode() + source).hexdigest()
        == item["git_blob_sha1"]
    )
    assert (
        hashlib.sha256((CORPUS / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )
    expected = {(0,): math.cos(0.5), (1,): -1j * cmath.exp(2j) * math.sin(0.5)}
    for program in (
        synth.synthesize(path, source_format="qmod"),
        synth.synthesize(native_rotation),
    ):
        check_state(program, fq.run(program), expected)
    circuit = fq.Circuit(1, dtype=torch.complex128).rz(0, -2.0).rx(0, 1.0).rz(0, 2.0)
    torch.testing.assert_close(
        fq.run(circuit).statevector().flatten(),
        torch.tensor(list(expected.values()), dtype=torch.complex128),
        atol=1e-12,
        rtol=0,
    )


def test_r_coherent_input_with_decimal_angles() -> None:
    theta, phi = 1.25, 0.75
    program = synth.synthesize(
        "qfunc main(output q:qbit) {allocate(q);H(q);R(1.25,0.75,q);}",
        source_format="qmod",
    )
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    expected = {
        (0,): (c - 1j * cmath.exp(-1j * phi) * s) / math.sqrt(2),
        (1,): (c - 1j * cmath.exp(1j * phi) * s) / math.sqrt(2),
    }
    check_state(program, fq.run(program), expected)


@pytest.mark.parametrize("args", ["1, q", "1, pi, q", "1, 9007199254740992.1, q"])
def test_r_rejects_invalid_signature_or_second_angle(args: str) -> None:
    with pytest.raises(InteropConversionError, match=r"qmod\.(signature|angle)"):
        synth.synthesize(
            f"qfunc main(output q:qbit) {{allocate(q);R({args});}}",
            source_format="qmod",
        )
