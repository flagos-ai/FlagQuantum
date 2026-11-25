"""Check the original unary-selected qubitization walk with dense matrix oracles."""

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
_PAULIS = {
    axis: torch.tensor(entries, dtype=_DTYPE)
    for axis, entries in {
        "I": [[1, 0], [0, 1]],
        "X": [[0, 1], [1, 0]],
        "Y": [[0, -1j], [1j, 0]],
        "Z": [[1, 0], [0, -1]],
    }.items()
}


def _original() -> Path:
    manifest = json.loads((_FIXTURES / "qubitization_qpe_sources.json").read_text())
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


def _table(source: str, call: str) -> list[float]:
    start = source.index("[", source.index(call))
    end = source.index("]", start)
    return ast.literal_eval(source[start : end + 1])


def _preparation(probabilities: list[float]) -> torch.Tensor:
    size = len(probabilities)
    matrix = torch.eye(size, dtype=_DTYPE)
    for width in range(size.bit_length() - 1):
        length = size >> width
        half = length // 2
        for start in range(0, size, length):
            left = sum(probabilities[start : start + half])
            right = sum(probabilities[start + half : start + length])
            if left + right == 0:
                continue
            c = math.sqrt(left / (left + right))
            s = math.sqrt(right / (left + right))
            a = matrix[start : start + half].clone()
            b = matrix[start + half : start + length].clone()
            matrix[start : start + half] = c * a - s * b
            matrix[start + half : start + length] = s * a + c * b
    return matrix


def _walk(source: str) -> torch.Tensor:
    table = {}
    for label, body in re.findall(
        r"^qfunc unitaries_(\d+)_lambda_[^\n]*\{\n(.*?)^\}",
        source,
        re.MULTILINE | re.DOTALL,
    ):
        word = torch.eye(16, dtype=_DTYPE)
        for axis, index_text in re.findall(
            r"pauli=Pauli::([IXYZ]),\s*index=(\d+)", body
        ):
            index = int(index_text)
            gate = torch.ones((1, 1), dtype=_DTYPE)
            for bit in reversed(range(4)):
                gate = torch.kron(gate, _PAULIS[axis if bit == index else "I"])
            word = gate @ word
        table[int(label)] = word
    assert set(table) == set(range(15))
    # The final unary branch has no right leaf. It dispatches the left leaf.
    table[15] = table[14]
    phases = _table(source, "assign_phase_table(")
    selection = torch.block_diag(
        *(
            complex(math.cos(angle), math.sin(angle)) * table[i]
            for i, angle in enumerate(phases)
        )
    )
    preparation = torch.kron(
        _preparation(_table(source, "inplace_prepare_state(")),
        torch.eye(16, dtype=_DTYPE),
    )
    reflection = torch.diag(torch.tensor([1] * 16 + [-1] * 240, dtype=_DTYPE))
    return reflection @ preparation.mH @ selection @ preparation


def reference(source: str) -> fq.Circuit:
    block, phase, data = tuple(range(4)), tuple(range(4, 9)), tuple(range(9, 13))
    circuit = fq.Circuit(13, dtype=_DTYPE).x(data[3]).x(data[1])
    for wire in phase:
        circuit.h(wire)
    walk = _walk(source)
    circuit.any(
        phase[-1],
        *block,
        *data,
        matrix=torch.block_diag(torch.eye(256, dtype=_DTYPE), walk),
    )
    signs = torch.ones(32, dtype=_DTYPE)
    signs[0] = -1
    reflection = torch.diag(signs)
    for index in range(4):
        phase_wire = phase[-2 - index]
        circuit.any(phase_wire, *block, matrix=reflection)
        circuit.any(*block, *data, matrix=torch.linalg.matrix_power(walk, 1 << index))
        circuit.any(phase_wire, *block, matrix=reflection)
    words = torch.arange(32, dtype=torch.float64)
    inverse_qft = torch.exp(-2j * math.pi / 32 * torch.outer(words, words)) / math.sqrt(
        32
    )
    return circuit.any(*phase, matrix=inverse_qft)


def compare_original() -> tuple[float, float, object]:
    path = _original()
    program = synth.synthesize(
        path, max_qubits=15, max_workspace=2, max_operations=60_000
    )
    assert program.registers == {"block": (0, 1, 2, 3), "phase_var": (4, 5, 6, 7, 8)}
    state = fq.run(program).statevector().reshape(8192, -1)
    expected = fq.run(reference(path.read_text())).statevector().reshape(-1)
    error = torch.max(torch.abs(state[:, 0] - expected)).item()
    workspace = torch.linalg.vector_norm(state[:, 1:]).item()
    return error, workspace, program.report["resources"]


def test_original_qubitization_all_amplitudes() -> None:
    error, workspace, _ = compare_original()
    assert error < 1e-10
    assert workspace < 1e-10


@pytest.mark.parametrize(
    ("before", "after", "diagnostic"),
    [
        (
            "CX(parent, aux);\n      X(sel_bit);",
            "CX(parent, aux);",
            "did not return to zero|changed a borrowed bit",
        ),
        (
            "_unary_root_expanded___0(sel_bits[3]",
            "_unary_root_expanded___0(sel_bits[2]",
            "arguments must be disjoint",
        ),
    ],
)
def test_unproved_selector_is_refused(before: str, after: str, diagnostic: str) -> None:
    source = _original().read_text()
    assert before in source
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source.replace(before, after, 1), source_format="qmod")
