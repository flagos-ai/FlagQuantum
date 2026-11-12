"""Pinned Classiq originals and coherent checks for the admitted gate subset."""

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.ecosystem import InteropConversionError
from flagquantum.synth import CRX, CX, SWAP, Output, QBit, allocate, qfunc
from tests.synthesis_workflow_helpers import check_state
from tools.synthesis_import_examples.qmod_pair_gates.main import (
    QMOD_PAIR,
    native_pair,
    pair_circuit,
)

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


def test_native_direct_calls_match_qmod_circuit_and_oracle():
    native = synth.synthesize(native_pair)
    imported = synth.synthesize(QMOD_PAIR, source_format="qmod")
    expected = {
        (0, 0): 2**-0.5,
        (1, 1): math.cos(0.625) * 2**-0.5,
        (1, 0): -1j * math.sin(0.625) * 2**-0.5,
    }
    for program in (native, imported):
        check_state(program, fq.run(program), expected)
    assert native.circuit.instructions == imported.circuit.instructions
    torch.testing.assert_close(
        fq.run(native).statevector(),
        fq.run(pair_circuit()).statevector(),
        atol=1e-12,
        rtol=0,
    )


def test_native_uses_shared_alias_and_initialization_checks():
    @qfunc
    def aliased(a: Output[QBit]) -> None:
        allocate(a)
        SWAP(a, a)

    @qfunc
    def uninitialized(a: Output[QBit], b: Output[QBit]) -> None:
        allocate(a)
        CX(a, b)

    for source, diagnostic in (
        (aliased, "register.gate"),
        (uninitialized, "register.uninitialized"),
    ):
        with pytest.raises(HybridVerificationError, match=diagnostic):
            synth.synthesize(source)


def test_native_crx_requires_finite_static_angle():
    angle = math.inf

    @qfunc
    def invalid(a: Output[QBit], b: Output[QBit]) -> None:
        allocate(a)
        allocate(b)
        CRX(angle, a, b)

    with pytest.raises(HybridVerificationError, match="program.angle"):
        synth.synthesize(invalid)


@pytest.mark.parametrize(
    ("body", "diagnostic"),
    [
        ("allocate(a); allocate(b); CX(a,a);", "register.gate"),
        ("allocate(a); CRX(1,a,b);", "register.uninitialized"),
        ("allocate(a); allocate(b); SWAP(a);", "qmod.signature"),
    ],
)
def test_pair_gate_rejections(body, diagnostic):
    source = f"qfunc main(output a:qbit,output b:qbit) {{{body}}}"
    with pytest.raises(
        (HybridVerificationError, InteropConversionError), match=diagnostic
    ):
        synth.synthesize(source, source_format="qmod")


def test_negative_crx_angle_preserves_full_coherent_state() -> None:
    p = synth.synthesize(
        "qfunc main(output a:qbit,output b:qbit) {allocate(a);allocate(b);H(a);CRX(-1,a,b);}",
        source_format="qmod",
    )
    c = fq.Circuit(2, dtype=torch.complex128).h(0).crx(0, 1, theta=-1)
    torch.testing.assert_close(
        fq.run(p).statevector(), fq.run(c).statevector(), atol=1e-12, rtol=0
    )
