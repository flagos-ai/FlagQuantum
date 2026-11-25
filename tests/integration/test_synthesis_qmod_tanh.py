"""Prepare an independent reference for the pinned piecewise-linear example."""

from __future__ import annotations

import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError


def xor_words(source: str) -> tuple[int, ...]:
    [coefficient_a, coefficient_b] = re.findall(
        r"\[([0-9.,\s]+)\]",
        source.split("compute_piecewise_linear( // tanh coefficients", 1)[1],
    )
    a, b = (
        tuple(Fraction(entry.strip()) for entry in values.split(","))
        for values in (coefficient_a, coefficient_b)
    )
    assert len(a) == len(b) == 4
    words = []
    for raw in range(16):
        label = raw >> 2
        value = a[label] * Fraction(raw, 16) + b[label]
        words.append((value.numerator * 16 // value.denominator) & 15)
    return tuple(words)


def tanh_reference(source: str, initial: int = 0) -> fq.Circuit:
    words = xor_words(source)
    reference = fq.Circuit(8, dtype=torch.complex128)
    for wire in range(4):
        reference.h(wire)
        if initial & 1 << wire:
            reference.x(7 - wire)
    matrix = torch.zeros((256, 256), dtype=torch.complex128)
    for word in range(256):
        matrix[word ^ words[word >> 4], word] = 1
    return reference.any(*range(8), matrix=matrix)


pytestmark = pytest.mark.integration
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "classiq_qmod"


def _original() -> Path:
    manifest = json.loads((_FIXTURES / "tanh_sources.json").read_text())
    [row] = manifest["fixtures"]
    path = _FIXTURES / row["file"]
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == row["git_blob_sha1"]
    )
    assert (
        hashlib.sha256((_FIXTURES / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )
    return path


@pytest.mark.parametrize("initial", [0, 5, 10])
def test_tanh_fixed_point_xor_and_clean_workspace(initial: int) -> None:
    source = _original()
    raw = source.read_text()
    if initial:
        allocation = "allocate(4, false, 4, y);"
        assert allocation in raw
        program = synth.synthesize(
            raw.replace(allocation, allocation + f" y ^= {initial / 16};"),
            source_format="qmod",
            max_qubits=10,
            max_workspace=2,
        )
    else:
        program = synth.synthesize(source, max_qubits=10, max_workspace=2)
    expected = fq.run(tanh_reference(raw, initial)).statevector().reshape(-1)
    workspace = program.report["workspace_qubits"]
    assert workspace == 2
    actual = fq.run(program).statevector().reshape(256, 1 << workspace)
    assert program.report["qmod_fractional_bits"] == {"x": 4, "y": 4}
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    assert actual[:, 1:].count_nonzero().item() == 0


@pytest.mark.parametrize(
    ("before", "after", "diagnostic"),
    [
        ("log(As.len,2)", "log(3,2)", "requires a positive power of two"),
        ("v.len-num_segs", "v.len-num_segs-16", "selector slice is out of bounds"),
        ("y ^= As", "x ^= As", "reads must be disjoint"),
        (
            "allocate(4, false, 4, y)",
            "allocate(9, false, 9, y)",
            "target needs at most eight bits",
        ),
        ("0.9375", "0.9", "must be exact binary rationals"),
    ],
)
def test_tanh_refuses_unproved_fixed_point_updates(
    before: str, after: str, diagnostic: str
) -> None:
    raw = _original().read_text()
    assert before in raw
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(raw.replace(before, after, 1), source_format="qmod")
