"""Verify original width-dependent Fourier phases, including signed words."""

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from flagquantum.services.synthesis import _synthesize

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
DTYPE = torch.complex128


def original(name: str) -> tuple[Path, dict[str, str]]:
    manifest = json.loads((FIXTURES / "fourier_phase_sources.json").read_text())
    row = next(row for row in manifest["fixtures"] if row["file"] == name)
    path = FIXTURES / name
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


def fourier(size: int) -> torch.Tensor:
    index = torch.arange(size, dtype=torch.float64)
    return torch.exp(2j * math.pi / size * torch.outer(index, index)) / math.sqrt(size)


def tutorial_reference() -> tuple[fq.Circuit, torch.Tensor]:
    circuit = fq.Circuit(7, dtype=DTYPE).h(0).h(1).h(2)
    wires = tuple(range(3, 7))
    transform = fourier(16)
    x = torch.arange(8, dtype=torch.float64)
    y = torch.arange(16, dtype=torch.float64)
    diagonal = torch.exp((2j * math.pi / 16) * torch.outer(x.square(), y))
    circuit.any(*wires, matrix=transform)
    circuit.any(*range(7), matrix=torch.diag(diagonal.flatten()))
    circuit.any(*wires, matrix=transform.mH.contiguous())
    expected = torch.zeros((8, 16), dtype=DTYPE)
    expected[x.long(), x.long().square() % 16] = 1 / math.sqrt(8)
    return circuit, expected.flatten()


def sawtooth_reference(kicks: int) -> fq.Circuit:
    circuit = fq.Circuit(3, dtype=DTYPE).x(0).x(1)
    transform = fourier(8)
    signed = torch.tensor([0, 1, 2, 3, -4, -3, -2, -1], dtype=torch.float64)
    # The pinned main calls single_step with 0.1 independently of its input port.
    position = torch.diag(torch.exp(0.1j * math.pi / 8 * signed.square()))
    momentum = torch.diag(torch.exp(-1j * math.pi / 8 * signed.square()))
    for _ in range(kicks):
        circuit.any(0, 1, 2, matrix=transform)
        circuit.any(0, 1, 2, matrix=position)
        circuit.any(0, 1, 2, matrix=transform.mH.contiguous())
        circuit.any(0, 1, 2, matrix=momentum)
    return circuit


def test_original_fourier_arithmetic_matches_basis_mapping() -> None:
    path, _ = original("algo_design_QCE_tutorial_part_I.qmod")
    program = synth.synthesize(path)
    actual = fq.run(program).statevector().flatten()
    circuit, analytic = tutorial_reference()
    torch.testing.assert_close(
        actual, fq.run(circuit).statevector().flatten(), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(actual, analytic, atol=1e-12, rtol=0)
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(("kicks", "input_kick"), [(0, 0.31), (1, 0.17), (2, 0.43)])
def test_original_sawtooth_preserves_twos_complement_phases(
    kicks: int, input_kick: float
) -> None:
    path, _ = original("quantum_sawtooth_map.qmod")
    program = _synthesize(
        path, classical_inputs={"num_kicks": kicks, "kick_mag": input_kick}
    )
    actual = fq.run(program).statevector()
    expected = fq.run(sawtooth_reference(kicks)).statevector()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert program.report["workspace_qubits"] == 0


def test_nonliteral_quantum_power_requires_explicit_support() -> None:
    source = "qfunc main(output p:qnum<3>) {allocate(p); phase(p ** p.size);}"
    with pytest.raises(InteropConversionError, match="integer literal exponent"):
        synth.synthesize(source, source_format="qmod")
