"""Run the pinned whole-array Hadamard example across three entry points."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, allocate, hadamard_transform, qfunc
from tests.synthesis_workflow_helpers import check_state

pytestmark = pytest.mark.integration
CORPUS = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


@qfunc
def native(x: Output[QNum]) -> None:
    allocate(3, x)
    hadamard_transform(x)


def test_original_array_hadamard_matches_native_and_circuit() -> None:
    manifest = json.loads((CORPUS / "hadamard_transform_sources.json").read_text())
    item = manifest["fixtures"][0]
    source = CORPUS / item["file"]
    raw = source.read_bytes()
    blob = f"blob {len(raw)}\0".encode() + raw
    assert hashlib.sha1(blob).hexdigest() == item["git_blob_sha1"]
    assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    license_bytes = (CORPUS / manifest["license_file"]).read_bytes()
    assert hashlib.sha256(license_bytes).hexdigest() == manifest["license_sha256"]

    circuit = fq.Circuit(3, dtype=torch.complex128)
    for qubit in range(3):
        circuit.h(qubit)
    expected = {
        (i,): amplitude
        for i, amplitude in enumerate(fq.run(circuit).statevector().flatten().tolist())
    }
    for program in (synth.synthesize(source), synth.synthesize(native)):
        check_state(program, fq.run(program), expected)
        resources = program.report["resources"]
        assert (resources.n_qubits, resources.n_operations, resources.depth) == (
            3,
            3,
            1,
        )
        assert program.report["workspace_qubits"] == 0
