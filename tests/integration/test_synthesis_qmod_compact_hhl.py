"""Independent dense Circuit reference for the original compact HHL."""

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

DTYPE = torch.complex128


def compact_hhl_reference(source: str) -> fq.Circuit:
    i = torch.eye(2, dtype=DTYPE)
    x = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE)
    z = torch.diag(torch.tensor([1, -1], dtype=DTYPE))
    source_terms = re.findall(r"paulis=\[(.*?)\],\s*coefficient=([\d.]+)", source, re.S)
    assert len(source_terms) == 4
    hamiltonian = torch.zeros((4, 4), dtype=DTYPE)
    for indices, coefficient in source_terms:
        axes = [i, i]
        for pauli, wire in re.findall(r"pauli=(\d), index=(\d)", indices):
            axes[1 - int(wire)] = (i, x, y, z)[int(pauli)]
        hamiltonian += float(coefficient) * torch.kron(*axes)
    denominators = re.findall(r"sqrt\(1/(\d+)\)", source)
    prepared = torch.tensor([math.sqrt(1 / int(d)) for d in denominators], dtype=DTYPE)
    assert prepared.shape == (4,)
    identity = torch.eye(4, dtype=DTYPE)
    direction = identity[:, 0] - prepared
    prepare = identity - 2 * torch.outer(direction, direction.conj()) / direction.vdot(
        direction
    )
    phase = (2, 3, 4, 5)
    circuit = fq.Circuit(7, dtype=DTYPE).any(0, 1, matrix=prepare)
    qpe = fq.Circuit(7, dtype=DTYPE)
    for wire in phase:
        qpe.h(wire)
    for bit, wire in enumerate(reversed(phase)):
        evolution = torch.linalg.matrix_exp(2j * math.pi * (1 << bit) * hamiltonian)
        qpe.any(wire, 0, 1, matrix=torch.block_diag(identity, evolution))
    axes = torch.arange(16, dtype=torch.float64)
    fourier = torch.exp(-2j * math.pi / 16 * torch.outer(axes, axes)) / 4
    qpe.any(*phase, matrix=fourier)
    circuit.compose(qpe)
    inversion = source.split("qfunc assign_inversion_size4")[1].split("qfunc main")[0]
    operations = re.findall(
        r"RY\(([-.\d]+), indicator\)|CX\(value\[(\d)\], indicator\)", inversion
    )
    assert len(operations) == 32
    for angle, bit in operations:
        if angle:
            circuit.ry(6, theta=float(angle))
        else:
            circuit.cx(phase[-1 - int(bit)], 6)
    circuit.compose(qpe.adjoint())
    return circuit


pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


def original() -> Path:
    manifest = json.loads((FIXTURES / "compact_hhl_sources.json").read_text())
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
    return path


def test_compact_hhl_signed_phase_and_full_state():
    path = original()
    program = synth.synthesize(path, max_workspace=0)
    actual = fq.run(program).statevector()
    expected = fq.run(compact_hhl_reference(path.read_text())).statevector()
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
    assert dict(program.registers) == {
        "solution": (0, 1),
        "phase_var": (2, 3, 4, 5),
        "indicator": (6,),
    }
    assert program.report["workspace_qubits"] == 0
    assert program.report["qmod_fractional_bits"] == {"phase_var": 4}
    assert program.report["qmod_signed_registers"] == ("phase_var",)
    residual = actual.reshape(4, 16, 2)[:, 1:, :].abs().square().sum().item()
    assert residual == pytest.approx(0.13679306724196644, abs=1e-12)
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(path, max_operations=10)


@pytest.mark.parametrize(
    ("before", "after", "diagnostic"),
    [
        ("sqrt(1/6)", "sqrt(1/0)", "denominator"),
        ("sqrt(1/6)", "sqrt(-1)", "finite nonnegative literal"),
        ("sqrt(1/6)", "sqrt(sqrt(1/6))", "literal"),
        ("pauli=1", "pauli=4", "Pauli"),
        ("-2*pi* k", "-2*pi*k*k", "linear power"),
        ("-2*pi* k", "-2*pi", "linear power"),
        (
            "powered_suzuki_trotter_commuting_terms(k, hamiltonian, solution);",
            "powered_suzuki_trotter_commuting_terms(1, hamiltonian, solution);",
            "forward its lexical power",
        ),
        (
            "skip_control { CX(value[0], indicator); }",
            "CX(value[0], indicator);",
            "cyclic CX network",
        ),
    ],
)
def test_compact_hhl_invalid_literal_or_specialization(before, after, diagnostic):
    source = original().read_text()
    assert before in source
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source.replace(before, after, 1), source_format="qmod")


def test_cyclic_skip_control_matches_independent_coherent_control():
    source = """
    qfunc rotate(const bits:qbit[1], target:qbit) {
      RY(0.4, target);
      skip_control { CX(bits[0], target); }
      RY(0.6, target);
      skip_control { CX(bits[0], target); }
    }
    qfunc main(output coin:qbit, output bits:qbit[1], output target:qbit) {
      allocate(coin); allocate(bits); allocate(target);
      H(coin); S(coin); H(bits[0]); H(target); S(target);
      control(coin) {rotate(bits, target);}
    }
    """
    identity = torch.eye(2, dtype=DTYPE)

    def ry(angle: float) -> torch.Tensor:
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        return torch.tensor([[c, -s], [s, c]], dtype=DTYPE)

    expected = fq.Circuit(3, dtype=DTYPE).h(0).s(0).h(1).h(2).s(2)
    expected.any(0, 1, 2, matrix=torch.block_diag(identity, identity, ry(1), ry(-0.2)))
    program = synth.synthesize(source, source_format="qmod", max_workspace=0)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(expected).statevector(),
        atol=1e-12,
        rtol=0,
    )
