"""Verify shared controlled unitary powers with the unchanged Lanchester source."""

import ast
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth

pytestmark = pytest.mark.integration
_DTYPE = torch.complex128
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


def _original() -> Path:
    manifest = json.loads((_FIXTURES / "lanchester_sources.json").read_text())
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


def _bracketed_after(source: str, call: str) -> object:
    start = source.index("[", source.index(call))
    depth = 0
    for end in range(start, len(source)):
        depth += int(source[end] == "[") - int(source[end] == "]")
        if depth == 0:
            return ast.literal_eval(source[start : end + 1])
    raise AssertionError(call)


def reference(source: str) -> fq.Circuit:
    matrix = torch.tensor(_bracketed_after(source, "unitary("), dtype=_DTYPE)
    amplitudes = torch.tensor(
        _bracketed_after(source, "prepare_amplitudes("), dtype=_DTYPE
    )
    angles = _bracketed_after(source, "select_rotation(Pauli::Y,")
    assert matrix.shape == (32, 32)
    assert len(angles) == 64

    indicator, data, phase = 0, tuple(range(1, 6)), tuple(range(6, 12))
    identity = torch.eye(32, dtype=_DTYPE)
    axis = identity[0] - amplitudes
    preparation = identity - 2 * torch.outer(axis, axis.conj()) / torch.vdot(axis, axis)

    qpe = fq.Circuit(12, dtype=_DTYPE)
    for control in reversed(phase):
        qpe.h(control)
    powers = tuple(
        torch.block_diag(identity, torch.linalg.matrix_power(matrix, 1 << bit))
        for bit in range(6)
    )
    for control, block in zip(reversed(phase), powers, strict=True):
        qpe.any(control, *data, matrix=block)
    grid = torch.arange(64, dtype=torch.float64)
    fourier = torch.exp(-2j * math.pi / 64 * torch.outer(grid, grid)) / 8
    qpe.any(*phase, matrix=fourier)

    inverse_qpe = fq.Circuit(12, dtype=_DTYPE).any(*phase, matrix=fourier.mH)
    for control, block in zip(phase, reversed(powers), strict=True):
        inverse_qpe.any(control, *data, matrix=block.mH)
    for control in phase:
        inverse_qpe.h(control)

    rotation_blocks = []
    for angle in angles:
        cosine, sine = math.cos(angle / 2), math.sin(angle / 2)
        rotation_blocks.append(
            torch.tensor([[cosine, -sine], [sine, cosine]], dtype=_DTYPE)
        )
    circuit = fq.Circuit(12, dtype=_DTYPE)
    return (
        circuit.any(*data, matrix=preparation)
        .compose(qpe)
        .any(*phase, indicator, matrix=torch.block_diag(*rotation_blocks))
        .compose(inverse_qpe)
    )


def compare_original() -> tuple[float, float, object]:
    path = _original()
    program = synth.synthesize(
        path, max_qubits=16, max_workspace=4, max_operations=25_000
    )
    assert program.registers == {
        "indicator": (0,),
        "time_index_and_group.rabbits": (5,),
        "time_index_and_group.time_index": (1, 2, 3, 4),
        "rescaled_eig": (6, 7, 8, 9, 10, 11),
    }
    actual = fq.run(program).statevector().reshape(4096, -1)
    expected = fq.run(reference(path.read_text())).statevector().reshape(-1)
    error = (actual[:, 0] - expected).abs().max().item()
    work_norm = torch.linalg.vector_norm(actual[:, 1:]).item()
    assert error < 1e-10
    assert work_norm < 1e-10
    phase_tail = torch.linalg.vector_norm(actual[:, 0].reshape(64, 64)[:, 1:])
    reference_tail = torch.linalg.vector_norm(expected.reshape(64, 64)[:, 1:])
    assert reference_tail.item() > 0.05
    torch.testing.assert_close(phase_tail, reference_tail, atol=1e-10, rtol=0)
    return error, work_norm, program.report["resources"]


def test_lanchester_full_state() -> None:
    compare_original()


def test_flat_record_can_infer_one_prepared_field_and_refresh_nested_views() -> None:
    source = """
        qstruct Inner { bit:qbit; value:qnum; }
        qstruct Outer { tag:qbit; inner:Inner; }
        qfunc main(output sample:Inner,output wrapper:Outer) {
            prepare_amplitudes([0,0,0,1],0,sample);
            allocate(wrapper);
            X(wrapper.inner.bit);
        }
    """
    program = synth.synthesize(source, source_format="qmod")
    result = fq.run(program)
    assert dict(
        program.probabilities(result, registers=("sample.value", "wrapper.inner.bit"))
    ) == {(1, 1): 1.0}


def test_conflicting_whole_record_preparations_are_refused() -> None:
    source = """
        qstruct Data { marker:qbit; index:qnum; }
        qfunc main(output first:Data, output second:Data) {
            prepare_amplitudes([0,0,0,1],0,first);
            prepare_amplitudes([0,0,0,0,0,0,0,1],0,second);
        }
    """
    with pytest.raises(ValueError, match="unambiguous whole-record preparation"):
        synth.synthesize(source, source_format="qmod")
