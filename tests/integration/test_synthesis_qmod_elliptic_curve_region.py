"""Verify closed Qmod point arithmetic against a separate affine Circuit."""

import hashlib
import json
from itertools import product
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
_DTYPE = torch.complex128
_P = 7
_A = 5
_GENERATOR = (0, 5)
_Point = tuple[int, int]


def _original() -> Path:
    manifest = json.loads(
        (_FIXTURES / "elliptic_curve_region_sources.json").read_text()
    )
    [item] = manifest["fixtures"]
    path = _FIXTURES / item["file"]
    raw = path.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == item["git_blob_sha1"]
    )
    assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    license_bytes = (_FIXTURES / manifest["license_file"]).read_bytes()
    assert hashlib.sha256(license_bytes).hexdigest() == manifest["license_sha256"]
    return path


def _curve_add(left: _Point | None, right: _Point | None) -> _Point | None:
    if left is None:
        return right
    if right is None:
        return left
    x, y = left
    u, v = right
    if x == u and (y + v) % _P == 0:
        return None
    if left == right:
        slope = ((3 * x * x + _A) * pow(2 * y, -1, _P)) % _P
    else:
        slope = ((v - y) * pow((u - x) % _P, -1, _P)) % _P
    x_out = (slope * slope - x - u) % _P
    return x_out, (slope * (x - x_out) - y) % _P


def _independent_circuit(seed: _Point) -> fq.Circuit:
    multiples = [None, _GENERATOR]
    for _ in range(2, 6):
        multiples.append(_curve_add(multiples[-1], _GENERATOR))
    assert multiples[-1] is None
    addends = multiples[1], multiples[2], multiples[4]
    assert addends == ((0, 5), (2, 1), (0, 2))
    words = []
    for left, right in product(range(8), repeat=2):
        point = seed
        for scalar in (left, right):
            for bit, addend in enumerate(addends):
                if not scalar >> bit & 1:
                    continue
                assert point is not None and addend is not None
                assert (point[0] - addend[0]) % _P
                point = _curve_add(point, addend)
                assert point is not None and (addend[0] - point[0]) % _P
        assert point is not None and point == _curve_add(
            seed, multiples[(left + right) % 5]
        )
        words.append((point[1] << 3) | point[0])
    circuit = fq.Circuit(12, dtype=_DTYPE)
    for wire in range(6):
        circuit.h(wire)
    for bit in range(6):
        matrix = torch.eye(128, dtype=_DTYPE)
        for row, value in enumerate(words):
            if value >> bit & 1:
                indices = [2 * row, 2 * row + 1]
                matrix[indices] = matrix[list(reversed(indices))]
        circuit.any(*range(6), 11 - bit, matrix=matrix)
    labels = torch.arange(8, dtype=torch.float64)
    inverse = torch.exp(-2j * torch.pi * (labels[:, None] * labels[None, :]) / 8) / (
        8**0.5
    )
    return circuit.any(0, 1, 2, matrix=inverse).any(3, 4, 5, matrix=inverse)


@pytest.mark.parametrize("seed", [(4, 2), (3, 2)])
def test_closed_point_arithmetic_preserves_absolute_fourier_amplitudes(
    seed: _Point,
) -> None:
    path = _original()
    limits = dict(max_qubits=14, max_workspace=2, max_operations=15_000)
    if seed == (4, 2):
        program = synth.synthesize(path, **limits)
    else:
        source = path.read_text().replace("ecp.x ^= 4;", "ecp.x ^= 3;")
        program = synth.synthesize(source, source_format="qmod", **limits)
    assert program.report["qmod_fractional_bits"] == {"x1": 3, "x2": 3}
    assert program.registers == {
        "x1": (0, 1, 2),
        "x2": (3, 4, 5),
        "ecp.x": (9, 10, 11),
        "ecp.y": (6, 7, 8),
    }
    result = fq.run(program)
    expected = fq.run(_independent_circuit(seed)).statevector().flatten()
    returned = result.statevector().reshape(expected.numel(), -1)
    torch.testing.assert_close(returned[:, 0], expected, rtol=0, atol=3e-12)
    assert returned[:, 1:].abs().max().item() < 1e-12
    probabilities = program.probabilities(
        result, registers=("x1", "x2", "ecp.x", "ecp.y")
    )
    assert probabilities.get((0, 0, 5, 0), 0) == pytest.approx(
        float(expected[5].abs().square()), abs=1e-12
    )
    assert sum(probabilities.values()) == pytest.approx(1, abs=2e-12)


def test_quantum_kernel_refuses_unproved_phase_and_dirty_work() -> None:
    source = _original().read_text()
    marker = "PHASE(qft_const_adder_phase(index, value, 4), phi_b[index]);"
    assert marker in source
    phased = source.replace(marker, marker + " PHASE(pi / 7, phi_b[0]);", 1)
    with pytest.raises(
        InteropConversionError,
        match="reversible proof requires zero|not a basis map|no relative phase",
    ):
        synth.synthesize(phased, source_format="qmod")
    dirty = source.replace("free(slope);", "X(slope[0]); free(slope);", 1)
    with pytest.raises(InteropConversionError, match="reversible proof requires zero"):
        synth.synthesize(dirty, source_format="qmod")


def test_closed_point_arithmetic_enforces_resource_budget() -> None:
    with pytest.raises(ValueError, match="budget|operation"):
        synth.synthesize(_original(), max_operations=100)
