"""Verify the original DQI preparation and coherent GF(2) decoding."""

import ast
import hashlib
import json
import math
import re
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
_DTYPE = torch.complex128
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


def _original() -> Path:
    manifest = json.loads((_FIXTURES / "dqi_consuming_sources.json").read_text())
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


def _weight_amplitudes(source: str) -> list[float]:
    [literal] = re.findall(
        r"prepare_amplitudes\((\[[^\]]+\]), 0, k_num_errors\);", source
    )
    return ast.literal_eval(literal)


def reference(source: str) -> fq.Circuit:
    # A Dicke state assigns equal real amplitude to all words of its weight.
    weights = _weight_amplitudes(source)
    amplitudes = torch.tensor(
        [
            weights[k] / math.sqrt(math.comb(6, k)) if k < len(weights) else 0.0
            for word in range(64)
            for k in (word.bit_count(),)
        ],
        dtype=_DTYPE,
    )
    identity = torch.eye(64, dtype=_DTYPE)
    axis = identity[0] - amplitudes
    preparation = identity - 2 * torch.outer(axis, axis.conj()) / torch.vdot(axis, axis)
    y, solution = tuple(range(6)), tuple(range(6, 12))
    circuit = fq.Circuit(12, dtype=_DTYPE).any(*y, matrix=preparation)
    [signs] = re.findall(
        r"vector_product_phase_expanded___0\((\[[0-9, ]+\]), y\);", source
    )
    for bit, value in enumerate(ast.literal_eval(signs)):
        if value % 2:
            circuit.z(y[-1 - bit])

    for bit_text, parity in re.findall(r"out\[(\d+)\] \^= (.*?);", source):
        for control in re.findall(r"y\[(\d+)\]", parity):
            circuit.cx(y[-1 - int(control)], solution[-1 - int(bit_text)])
    terms = re.findall(r"control \(syndrome == (\d+)\) \{\s*error \^= (\d+);", source)
    assert len(terms) == len({int(word) for word, _ in terms})
    for word_text, mask_text in terms:
        word, mask = int(word_text), int(mask_text)
        basis = torch.eye(128, dtype=_DTYPE)
        row = word << 1
        basis[[row, row + 1]] = basis[[row + 1, row]]
        for bit, wire in enumerate(reversed(y)):
            if mask >> bit & 1:
                circuit.any(*solution, wire, matrix=basis)
    for wire in solution:
        circuit.h(wire)
    return circuit


def compare_original(changed_weights: bool) -> tuple[float, float, object]:
    path = _original()
    source = path.read_text()
    limits = {"max_qubits": 14, "max_workspace": 2, "max_operations": 5000}
    if changed_weights:
        start = source.index("[", source.index("  prepare_amplitudes("))
        end = source.index("]", start) + 1
        source = source[:start] + "[0.3, -0.4, 0.8660254037844386, 0.0]" + source[end:]
        program = synth.synthesize(source, source_format="qmod", **limits)
    else:
        program = synth.synthesize(path, **limits)
    actual = fq.run(program).statevector().reshape(4096, -1)
    expected = fq.run(reference(source)).statevector().flatten()
    error = (actual[:, 0] - expected).abs().max().item()
    workspace = torch.linalg.vector_norm(actual[:, 1:]).item()
    return error, workspace, program.report["resources"]


@pytest.mark.parametrize("changed_weights", [False, True])
def test_dqi_original_and_changed_weights(changed_weights: bool) -> None:
    error, workspace, _ = compare_original(changed_weights)
    assert error < 1e-10
    assert workspace < 1e-10


def test_dqi_rejects_nonzero_free() -> None:
    source = _original().read_text().replace("  X(qvar[0]);", "  X(qvar[1]);", 1)
    with pytest.raises(InteropConversionError, match="free requires zero"):
        synth.synthesize(source, source_format="qmod")
