"""Independent exact full-state Circuit for the unchanged GQSP source."""

import ast
import cmath
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


def reference(source: str) -> fq.Circuit:
    table, tail = source.split("}, ind,", 1)[1].rsplit(",", 1)
    triples = ast.literal_eval(table.strip())
    [negative] = re.findall(r"^\s*(\d+)\);", tail)
    [coefficient] = re.findall(r"phase \(([-\d.]+) \* x\);", source)
    [width] = re.findall(r"output x: qnum<(\d+), UNSIGNED", source)
    width = int(width)
    split = len(triples) - int(negative)
    dtype = torch.complex128
    x = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
    z = torch.diag(torch.tensor([1, -1], dtype=dtype))

    def rotation(theta: float, phi: float, lam: float) -> torch.Tensor:
        c, s = math.cos(theta), math.sin(theta)
        unitary = torch.tensor(
            [
                [c, -s * cmath.exp(1j * lam)],
                [s * cmath.exp(1j * phi), c * cmath.exp(1j * (phi + lam))],
            ],
            dtype=dtype,
        )
        return z @ x @ unitary @ x

    circuit = fq.Circuit(width + 1, dtype=dtype)
    for wire in range(width):
        circuit.h(wire)
    circuit.any(width, matrix=rotation(*triples[0]))
    for index, angles in enumerate(triples[1:], start=1):
        sign = 1 if index < split else -1
        active = 0 if sign == 1 else 1
        diagonal = torch.tensor(
            [
                (
                    cmath.exp(1j * sign * float(coefficient) * word / (1 << width))
                    if branch == active
                    else 1.0
                )
                for word in range(1 << width)
                for branch in (0, 1)
            ],
            dtype=dtype,
        )
        circuit.any(*range(width + 1), matrix=torch.diag(diagonal)).any(
            width, matrix=rotation(*angles)
        )
    return circuit


pytestmark = pytest.mark.integration
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "classiq_qmod"


def _original() -> Path:
    manifest = json.loads((_FIXTURES / "gqsp_sources.json").read_text())
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


def test_original_gqsp_uses_signed_powers_and_fixed_point_signal() -> None:
    path = _original()
    program = synth.synthesize(path, max_workspace=0, max_qubits=8)
    assert dict(program.registers) == {"x": tuple(range(7)), "ind": (7,)}
    assert program.report["qmod_fractional_bits"] == {"x": 7}
    assert program.report["workspace_qubits"] == 0
    assert program.report["resources"].n_operations == 41
    expected = fq.run(reference(path.read_text())).statevector()
    torch.testing.assert_close(
        fq.run(program).statevector(), expected, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    ("before", "after", "diagnostic"),
    [
        ("phases[0][2]", "phases[0][3]", "real matrix cell is out of bounds"),
        ("], 3);", "], 8);", "real matrix row slice is out of bounds"),
        (
            "foreach (theta, phi, _lambda:",
            "foreach (theta, theta, _lambda:",
            "row names must be distinct",
        ),
        ("u_func(x);", "", "expected a nonempty zero-argument callback"),
        ("u_func(x);", "X(ind);", "selected gate cannot write its control"),
        ("u_func(x);", "allocate(ind);", "requires pure effects on existing storage"),
    ],
)
def test_gqsp_refuses_invalid_bindings_and_callback_effects(
    before: str, after: str, diagnostic: str
) -> None:
    raw = _original().read_text()
    assert before in raw
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(raw.replace(before, after, 1), source_format="qmod")


@pytest.mark.parametrize("negative_power", [0, 2])
def test_static_matrix_ranges_keep_explicit_power_split(negative_power: int) -> None:
    source = _original().read_text().replace("], 3);", f"], {negative_power});")
    program = synth.synthesize(source, source_format="qmod", max_workspace=0)
    expected = fq.run(reference(source)).statevector()
    torch.testing.assert_close(
        fq.run(program).statevector(), expected, atol=1e-12, rtol=0
    )
