"""Verify the unchanged Poisson program with independent Fourier and Pauli matrices."""

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
    manifest = json.loads((_FIXTURES / "discrete_poisson_sources.json").read_text())
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


def _literal_table(source: str, operation: str) -> list[float]:
    start = source.index("[", source.index(operation))
    end = source.index("]", start) + 1
    return ast.literal_eval(source[start:end])


def _discrete_type2(*, sine: bool) -> torch.Tensor:
    size = 8
    index = torch.arange(size, dtype=torch.float64)
    row = index + int(sine)
    normalization = torch.where(
        index == (size - 1 if sine else 0),
        torch.ones_like(index),
        torch.full_like(index, math.sqrt(2.0)),
    )
    angles = math.pi / size * torch.outer(row, index + 0.5)
    return (
        (-angles.sin() if sine else angles.cos())
        * normalization[:, None]
        / math.sqrt(size)
    ).to(_DTYPE)


def _qpe_blocks(source: str) -> tuple[torch.Tensor, ...]:
    evo = source[
        source.index("qfunc powered_hamiltonian_evolution_") : source.index(
            "qfunc unitary_with_power_"
        )
    ]
    [base] = re.findall(r"p \* \(\((-?[0-9.]+)\) \* scaling\)", evo)
    [scale] = re.findall(r"\], ([0-9.]+), p, target", evo)
    terms = re.findall(
        r"PauliTerm\s*\{\s*pauli=\[(.*?)\],\s*coefficient=([-+0-9.eE]+)\s*\}",
        evo,
        re.DOTALL,
    )
    assert len(terms) == 16
    identity = torch.eye(64, dtype=_DTYPE)
    gates = {
        "I": torch.eye(2, dtype=_DTYPE),
        "X": torch.tensor([[0, 1], [1, 0]], dtype=_DTYPE),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=_DTYPE),
        "Z": torch.diag(torch.tensor([1, -1], dtype=_DTYPE)),
    }
    operators = []
    for axes, coefficient in terms:
        axes_in_order = re.findall(r"Pauli::([IXYZ])", axes)
        assert len(axes_in_order) == 6
        pauli = torch.ones((1, 1), dtype=_DTYPE)
        for axis in reversed(axes_in_order):
            pauli = torch.kron(pauli, gates[axis])
        operators.append((pauli, float(coefficient)))
    result = []
    for power in range(6):
        evolution = identity
        for pauli, coefficient in operators:
            angle = (1 << power) * float(base) * float(scale) * coefficient
            evolution = (
                math.cos(angle) * identity - 1j * math.sin(angle) * pauli
            ) @ evolution
        result.append(torch.block_diag(identity, evolution))
    return tuple(result)


def reference(source: str) -> fq.Circuit:
    data, phase, indicator = tuple(range(6)), tuple(range(6, 12)), 12
    preparation = torch.tensor(
        _literal_table(source, "prepare_amplitudes("), dtype=_DTYPE
    )
    axis = torch.eye(64, dtype=_DTYPE)[0] - preparation
    reflector = torch.eye(64, dtype=_DTYPE) - 2 * torch.outer(
        axis, axis.conj()
    ) / torch.vdot(axis, axis)
    cosine, negative_sine = _discrete_type2(sine=False), _discrete_type2(sine=True)
    transform = torch.kron(cosine, negative_sine)

    qpe = fq.Circuit(13, dtype=_DTYPE)
    for wire in phase:
        qpe.h(wire)
    matrices = _qpe_blocks(source)
    for control, block in zip(reversed(phase), matrices, strict=True):
        qpe.any(control, *data, matrix=block)
    axis = torch.arange(64, dtype=torch.float64)
    inverse_qft = torch.exp(-2j * math.pi / 64 * torch.outer(axis, axis)) / 8
    qpe.any(*phase, matrix=inverse_qft)

    reverse = fq.Circuit(13, dtype=_DTYPE).any(*phase, matrix=inverse_qft.mH)
    for control, block in zip(phase, reversed(matrices), strict=True):
        reverse.any(control, *data, matrix=block.mH)
    for wire in phase:
        reverse.h(wire)
    angles = _literal_table(source, "select_rotation(Pauli::Y,")
    assert len(angles) == 64
    blocks = [
        torch.tensor(
            [[math.cos(a / 2), -math.sin(a / 2)], [math.sin(a / 2), math.cos(a / 2)]],
            dtype=_DTYPE,
        )
        for a in angles
    ]
    return (
        fq.Circuit(13, dtype=_DTYPE)
        .any(*data, matrix=reflector)
        .any(*data, matrix=transform)
        .compose(qpe)
        .any(*phase, indicator, matrix=torch.block_diag(*blocks))
        .compose(reverse)
        .any(*data, matrix=transform.mH)
    )


def compare_original() -> tuple[float, object]:
    path = _original()
    program = synth.synthesize(
        path, max_qubits=13, max_workspace=0, max_operations=6000
    )
    assert program.registers == {
        "xy_variable": tuple(range(6)),
        "phase_var": tuple(range(6, 12)),
        "indicator": (12,),
    }
    actual = fq.run(program).statevector().flatten()
    expected = fq.run(reference(path.read_text())).statevector().flatten()
    return torch.max(torch.abs(actual - expected)).item(), program.report["resources"]


def test_original_poisson_matches_full_state() -> None:
    error, _ = compare_original()
    assert error < 1e-10


@pytest.mark.parametrize(
    ("before", "after", "message"),
    [
        (
            "PHASE((-pi) / 16, q);",
            "PHASE(0.0, q);",
            "borrowed selector does not return",
        ),
        (
            "qct_type2_expanded___0(xy_variable[1]);",
            "qct_type2_expanded___0(xy_variable[2]);",
            "numeric array index is out of bounds",
        ),
    ],
)
def test_proof_and_view_reject_unsafe_source(
    before: str, after: str, message: str
) -> None:
    source = _original().read_text()
    assert before in source
    with pytest.raises(InteropConversionError, match=message):
        synth.synthesize(source.replace(before, after, 1), source_format="qmod")
