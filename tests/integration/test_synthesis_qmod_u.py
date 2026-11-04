"""Exercise the pinned Qmod U original and its absolute phase."""

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
from flagquantum.synth import Output, QBit, R, allocate, phase, qfunc
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration
SOURCE = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/U.qmod"
HALF_PI = math.pi / 2


@qfunc
def native(q: Output[QBit]) -> None:
    allocate(q)
    phase(q, 1.5)
    R(1, HALF_PI, q)
    phase(q, 2)
    phase(1, 1.1)


@pytest.mark.parametrize("coherent", [False, True])
def test_u_agrees_with_circuit_including_global_phase(coherent):
    raw = SOURCE.read_bytes()
    manifest = json.loads(SOURCE.with_name("u_sources.json").read_text())
    item = manifest["fixtures"][0]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == item["git_blob_sha1"]
    )
    assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    license_bytes = SOURCE.with_name(manifest["license_file"]).read_bytes()
    assert hashlib.sha256(license_bytes).hexdigest() == manifest["license_sha256"]
    circuit = fq.Circuit(1, dtype=torch.complex128)
    if coherent:
        circuit.h(0)
    circuit.u3(0, theta=1, phi=2, lbd=1.5)
    expected = cmath.exp(1.1j) * fq.run(circuit).statevector()
    qmod = raw.decode().replace("allocate(q);", "allocate(q); H(q);")
    program = (
        synth.synthesize(qmod, source_format="qmod")
        if coherent
        else synth.synthesize(SOURCE)
    )
    torch.testing.assert_close(
        fq.run(program).statevector(), expected, atol=1e-12, rtol=0
    )
    if not coherent:
        native_program = synth.synthesize(native)
        check_state(
            native_program,
            fq.run(native_program),
            {(i,): value for i, value in enumerate(expected.flatten().tolist())},
        )


@pytest.mark.parametrize("args", ["1, 2, 1.5, q", "1, 2, 1.5, q, q"])
def test_u_rejects_missing_or_dynamic_angle(args):
    with pytest.raises(InteropConversionError, match=r"qmod\.(signature|angle)"):
        synth.synthesize(
            f"qfunc main(output q:qbit) {{allocate(q); U({args});}}",
            source_format="qmod",
        )
