"""Reference exact quantized samples and their joint output purification."""

import hashlib
import json
import math
import re
from fractions import Fraction
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
_DTYPE = torch.complex128


def _original() -> Path:
    manifest = json.loads((_FIXTURES / "brownian_quantized_sources.json").read_text())
    [row] = manifest["fixtures"]
    path = _FIXTURES / row["file"]
    raw = path.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == row["git_blob_sha1"]
    )
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    license_bytes = (_FIXTURES / manifest["license_file"]).read_bytes()
    assert hashlib.sha256(license_bytes).hexdigest() == manifest["license_sha256"]
    return path


def _integer_reference(source: str) -> list[tuple[int, int]]:
    [cosine] = re.findall(r"out = \((-\d+\.\d+)\) \* \(x - 0\.5\)", source)
    [price] = re.findall(r"\((\d+\.\d+) \* sin_pi_t\)", source)
    # Independent fixed integer arithmetic. The scale is documented by the
    # Classiq SDK release accompanying the pinned source snapshot.
    unit = 256
    c = math.floor(float(cosine) * unit)
    k = math.floor(float(price) * unit)
    samples = []
    environment = set()
    for row in range(32):
        t, a, b, d, e = (row >> bit & 1 for bit in range(5))
        u1 = c * (2 * t - 1) // 2
        u2 = u1 * u1 // unit - unit
        u3 = u1 * u2 // unit - u1
        sine = -unit // 4
        coefficient = k * sine // unit
        series = a * t * unit + coefficient * (b * u1 + d * u2 + e * u3) // unit
        samples.append((t, 2 * series - t * unit + unit))
        environment.add((a, b, d, e, u1))
    assert len(environment) == len(samples)  # Explicitly dropped source words.
    return samples


def _circuit_bits(samples: list[tuple[int, int]], width: int) -> torch.Tensor:
    """Independently write each truth column in a small Circuit on basis inputs."""
    rows = torch.zeros((32, 64), dtype=_DTYPE)
    indices = torch.arange(32)
    rows[indices, 2 * indices] = 1
    packed = torch.zeros(32, dtype=torch.long)
    for bit in range(width + 1):
        columns = torch.eye(64, dtype=_DTYPE)
        for index, (t, g) in enumerate(samples):
            word = (t << width) | (g & ((1 << width) - 1))
            if word >> bit & 1:
                low = 2 * index
                columns[[low, low + 1]] = columns[[low + 1, low]]
        circuit = fq.Circuit(6, bsz=32, dtype=_DTYPE, inputs=rows)
        states = circuit.any(*range(6), matrix=columns).state()
        words = states.abs().argmax(dim=1)
        assert torch.equal(words >> 1, indices)
        packed |= (words & 1) << bit
    return packed


@pytest.mark.parametrize("changed", [False, True])
def test_brownian_all_signed_words_and_absolute_amplitudes(changed: bool) -> None:
    source_path = _original()
    source = source_path.read_text()
    limits = dict(max_qubits=23, max_workspace=2, max_operations=3000)
    if changed:
        source = source.replace("0.450158158078553", "0.431573")
        program = synth.synthesize(source, source_format="qmod", **limits)
    else:
        program = synth.synthesize(source_path, **limits)
    assert program.report["qmod_fractional_bits"] == {"G": 8}
    assert program.report["qmod_signed_registers"] == ("G",)
    assert program.registers == {"t": (0,), "G": tuple(range(1, 16))}
    assert program.report["workspace_qubits"] == 2
    expected_samples = _integer_reference(source)
    if not changed:
        assert expected_samples[3] == (1, 512)
        assert min(value for _, value in expected_samples) == -1874
        assert max(value for _, value in expected_samples) == 2132
    result = fq.run(program)
    state = result.statevector().flatten()
    packed = _circuit_bits(expected_samples, 15)
    occupied = (packed << 7) | (torch.arange(32) << 2)
    nonzero = torch.nonzero(state.abs() > 1e-10).flatten()
    assert torch.equal(nonzero, occupied.sort().values)
    torch.testing.assert_close(
        state[occupied],
        torch.full((32,), 1 / math.sqrt(32), dtype=_DTYPE),
        atol=1e-10,
        rtol=0,
    )
    assert torch.count_nonzero(nonzero & 3) == 0  # Both clean work bits.
    probability = program.probabilities(
        result, registers=("t", "G"), max_amplitudes=1 << 23
    )
    expected = {}
    for t, signed_word in expected_samples:
        key = t, signed_word & ((1 << 15) - 1)
        expected[key] = expected.get(key, Fraction()) + Fraction(1, 32)
    assert dict(probability) == pytest.approx(
        {key: float(value) for key, value in expected.items()}, abs=1e-10
    )


def test_quantized_profile_refuses_insufficient_discard_evidence() -> None:
    source = """qperm f(const x:qnum, output y:qnum) { y = x*0.101; }
    qfunc main(output x:qnum, output y:qnum) {
        witness:qnum; allocate(1,witness); allocate(1,x);
        hadamard_transform(x); f(x,y); drop(witness);
    }"""
    with pytest.raises(
        InteropConversionError, match="must distinguish every prepared branch"
    ):
        synth.synthesize(source, source_format="qmod")
