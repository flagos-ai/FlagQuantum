"""Analytic full-state Circuit for the option-pricing originals."""

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
from flagquantum.services.synthesis import _synthesize


def option_reference(source: str, repetitions: int) -> fq.Circuit:
    probability_text = re.search(r"inplace_prepare_state\((\[.*?\])", source, re.S)
    angle_text = re.search(r"select_rotation\(Pauli::Y,\s*(\[.*?\])", source, re.S)
    threshold_text = re.search(r"control \(asset >= (\d+)\)", source)
    assert probability_text and angle_text and threshold_text
    probabilities = ast.literal_eval(probability_text[1])
    angles = ast.literal_eval(angle_text[1])
    threshold = int(threshold_text[1])
    assert len(probabilities) == len(angles) == 32

    dtype = torch.complex128
    prepared = torch.zeros(64, dtype=dtype)
    for price, (probability, angle) in enumerate(
        zip(probabilities, angles, strict=True)
    ):
        amplitude = math.sqrt(probability / math.fsum(probabilities))
        angle = angle if price >= threshold else 0.0
        prepared[price] = amplitude * math.cos(angle / 2)
        prepared[32 + price] = amplitude * math.sin(angle / 2)

    zero = torch.eye(64, dtype=dtype)[:, 0]
    difference = zero - prepared
    identity = torch.eye(64, dtype=dtype)
    # The Householder extension prepares the same state from zero. The
    # diffuser depends only on that state, regardless of the extension.
    prepare = identity - 2 * torch.outer(
        difference, difference.conj()
    ) / difference.vdot(difference)
    diffuse = 2 * torch.outer(prepared, prepared.conj()) - identity
    wires = tuple(range(6))
    circuit = fq.Circuit(6, dtype=dtype).any(*wires, matrix=prepare)
    for _ in range(repetitions):
        circuit.z(0).any(*wires, matrix=diffuse)
    return circuit


pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


def originals() -> tuple[Path, ...]:
    manifest = json.loads((FIXTURES / "option_pricing_sources.json").read_text())
    assert (
        hashlib.sha256((FIXTURES / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )
    paths = []
    for row in manifest["fixtures"]:
        path = FIXTURES / row["file"]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            == row["git_blob_sha1"]
        )
        paths.append(path)
    return tuple(paths)


@pytest.mark.parametrize("k", [0, 1, 2])
@pytest.mark.parametrize("source_index", [0, 1])
def test_original_option_pricing_full_state(k, source_index):
    source = originals()[source_index]
    program = _synthesize(source, classical_inputs={"k": k}, max_workspace=0)
    expected = fq.run(option_reference(source.read_text(), k)).statevector()
    actual = fq.run(program).statevector()
    assert dict(program.registers) == {"indicator": (0,)}
    assert program.report["workspace_qubits"] == 0
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)


@pytest.mark.parametrize(
    ("before", "after", "diagnostic"),
    [
        (
            "qvar_num: qnum<6, False, 0>",
            "qvar_num: qnum<5, False, 0>",
            "cannot prove annotated borrowed phase view",
        ),
        (
            "disable_const_checks(qvar)",
            "disable_const_checks(qvar_num)",
            "cannot prove annotated borrowed phase view",
        ),
        (
            "phase (pi);\n    }",
            "X(qvar_num);\n    }",
            "cannot prove annotated borrowed phase view",
        ),
    ],
)
def test_borrowed_phase_view_refuses_escape_or_mismatched_annotations(
    before, after, diagnostic
):
    raw = originals()[0].read_text()
    assert before in raw
    with pytest.raises(InteropConversionError, match=diagnostic):
        _synthesize(
            raw.replace(before, after, 1),
            source_format="qmod",
            classical_inputs={"k": 1},
        )


def test_guard_must_use_the_actual_rotation_selector():
    source = """
    qfunc select(const index: qnum<1>, target: qbit) {
      select_rotation(Pauli::Y, [0.2, 0.9], index, target);
    }
    qfunc main(output guard: qnum<1>, output index: qnum<1>, output target: qbit) {
      allocate(guard); allocate(index); allocate(target);
      control (guard >= 1) { select(index, target); }
    }
    """
    with pytest.raises(InteropConversionError, match="exactly the rotation selector"):
        synth.synthesize(source, source_format="qmod")
