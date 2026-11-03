"""Pinned Classiq originals and coherent checks for the admitted gate subset."""

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis_qmod_pair_gates import bell, zero_pair
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.ecosystem import InteropConversionError
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration
CORPUS = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
MANIFEST = json.loads((CORPUS / "pair_gate_sources.json").read_text())


@pytest.mark.parametrize("item", MANIFEST["fixtures"], ids=lambda item: item["file"])
def test_original_pair_gate(item):
    path = CORPUS / item["file"]
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == item["sha256"]
    assert (
        hashlib.sha1(f"blob {len(source)}\0".encode() + source).hexdigest()
        == item["git_blob_sha1"]
    )
    license_bytes = (CORPUS / MANIFEST["license_file"]).read_bytes()
    assert hashlib.sha256(license_bytes).hexdigest() == MANIFEST["license_sha256"]
    program = synth.synthesize(path, source_format="qmod")
    check_state(program, fq.run(program), {(0, 0): 1})
    native = synth.synthesize(zero_pair)
    check_state(native, fq.run(native), {(0, 0): 1})
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("H(b); CX(b,a);", {(0, 0): 2**-0.5, (1, 1): 2**-0.5}),
        ("H(b); SWAP(a,b);", {(0, 0): 2**-0.5, (1, 0): 2**-0.5}),
        (
            "H(b); CRX(1.25,b,a);",
            {
                (0, 0): 2**-0.5,
                (0, 1): math.cos(0.625) * 2**-0.5,
                (1, 1): -1j * math.sin(0.625) * 2**-0.5,
            },
        ),
    ],
)
def test_coherent_pair_gates(action, expected):
    source = (
        f"qfunc main(output a:qbit,output b:qbit) {{allocate(a);allocate(b);{action}}}"
    )
    program = synth.synthesize(source, source_format="qmod")
    check_state(program, fq.run(program), expected)


def test_native_fresh_assignment_matches_bell_circuit():
    program = synth.synthesize(bell)
    reference = fq.Circuit(2, dtype=torch.complex128).h(1).cx(1, 0)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("body", "diagnostic"),
    [
        ("allocate(a); allocate(b); CX(a,a);", "register.gate"),
        ("allocate(a); CRX(1,a,b);", "register.uninitialized"),
        ("allocate(a); allocate(b); CRX(-1,a,b);", "qmod.expression"),
        ("allocate(a); allocate(b); SWAP(a);", "qmod.signature"),
    ],
)
def test_pair_gate_rejections(body, diagnostic):
    source = f"qfunc main(output a:qbit,output b:qbit) {{{body}}}"
    with pytest.raises(
        (HybridVerificationError, InteropConversionError), match=diagnostic
    ):
        synth.synthesize(source, source_format="qmod")
