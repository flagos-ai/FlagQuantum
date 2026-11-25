"""Verify controlled preparation, coherent QPE, and the original QMC program."""

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
DTYPE = torch.complex128


def original() -> tuple[Path, dict[str, str]]:
    manifest = json.loads((FIXTURES / "user_defined_qmc_sources.json").read_text())
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
    return path, row


def reference() -> tuple[fq.Circuit, torch.Tensor]:
    probabilities = torch.tensor(
        [
            0.0,
            0.0357142857,
            0.0714285714,
            0.1071428571,
            0.1428571429,
            0.1785714286,
            0.2142857143,
            0.25,
        ],
        dtype=torch.float64,
    )
    preparation = torch.eye(8, dtype=DTYPE)
    for stride in (4, 2, 1):
        layer = torch.zeros_like(preparation)
        for group in range(0, 8, 2 * stride):
            left = math.sqrt(float(probabilities[group : group + stride].sum()))
            right = math.sqrt(
                float(probabilities[group + stride : group + 2 * stride].sum())
            )
            angle = math.atan2(right, left)
            for i in range(stride):
                a, b = group + i, group + i + stride
                layer[a, a] = layer[b, b] = math.cos(angle)
                layer[b, a] = math.sin(angle)
                layer[a, b] = -math.sin(angle)
        preparation = layer @ preparation

    rotations = []
    for x in range(8):
        angle = 0.25 * x + 0.2
        co, si = math.cos(angle), math.sin(angle)
        rotations.append(torch.tensor([[co, -si], [si, co]], dtype=DTYPE))
    identity = torch.eye(16, dtype=DTYPE)
    loader = torch.block_diag(*rotations) @ torch.kron(
        preparation, torch.eye(2, dtype=DTYPE)
    )
    reflection = identity.clone()
    reflection[0, 0] = -1
    oracle = torch.diag(torch.tensor([1, -1] * 8, dtype=DTYPE))
    grover = -loader @ reflection @ loader.mH @ oracle

    wires = (3, 4, 5, 6)
    circuit = fq.Circuit(7, dtype=DTYPE).any(*wires, matrix=loader)
    for bit in range(3):
        power = torch.linalg.matrix_power(grover, 1 << bit)
        circuit.h(2 - bit).any(
            2 - bit, *wires, matrix=torch.block_diag(identity, power)
        )
    inverse = torch.fft.fft(torch.eye(8, dtype=DTYPE), dim=0, norm="ortho")
    circuit.any(0, 1, 2, matrix=inverse)

    state = torch.stack(
        [torch.linalg.matrix_power(grover, index) @ loader[:, 0] for index in range(8)]
    ) / math.sqrt(8)
    analytic = torch.fft.fft(state, dim=0, norm="ortho")
    return circuit, analytic


def test_original_user_defined_qmc_full_state_and_signed_phase():
    path, _ = original()
    program = synth.synthesize(path)
    result = fq.run(program)
    actual = result.statevector().reshape(8, 16)
    circuit, analytic = reference()
    torch.testing.assert_close(
        actual, fq.run(circuit).statevector().reshape(8, 16), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(actual, analytic, atol=1e-12, rtol=0)
    assert program.report["qmod_fractional_bits"] == {"phase": 3}
    assert program.report["qmod_signed_registers"] == ("phase",)
    assert program.report["workspace_qubits"] == 0
    marginal = analytic.abs().square().sum(dim=1)
    distribution = program.probabilities(result, registers=("phase",))
    assert [distribution.get((i,), 0.0) for i in range(8)] == pytest.approx(
        marginal.tolist(), abs=1e-12
    )


def test_linear_rotation_rejects_overlapping_physical_views():
    source = """qfunc main(output x:qbit[3]) {
      allocate(x);
      linear_pauli_rotations([Pauli.Y],[0.5],[0.4],x[0:2],x[1]);
    }"""
    with pytest.raises(InteropConversionError, match="disjoint qubit arrays"):
        synth.synthesize(source, source_format="qmod")
