"""Validate unchanged HHL sources against dense, phase-sensitive Circuits."""

import ast
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
DTYPE = torch.complex128
ACCEPTED = (
    "hhl_workshop.qmod",
    "portfolio_optimization_with_hhl.qmod",
    "HHL_portfolio.qmod",
)


def original_file(name: str) -> Path:
    manifest = json.loads((FIXTURES / "hhl_sources.json").read_text())
    original = next(row for row in manifest["fixtures"] if row["file"] == name)
    path = FIXTURES / name
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == original["sha256"]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == original["git_blob_sha1"]
    )
    assert (
        hashlib.sha256((FIXTURES / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )
    return path


def literal_arrays(pattern: str, source: str) -> list[Any]:
    return [ast.literal_eval(item) for item in re.findall(pattern, source, re.S)]


def preparation(vector: list[float]) -> torch.Tensor:
    amplitudes = torch.tensor(vector, dtype=DTYPE)
    torch.testing.assert_close(
        torch.linalg.vector_norm(amplitudes), torch.tensor(1.0, dtype=torch.float64)
    )
    identity = torch.eye(len(vector), dtype=DTYPE)
    direction = identity[:, 0] - amplitudes
    return identity - 2 * torch.outer(direction, direction.conj()) / direction.vdot(
        direction
    )


def ry(angle: float) -> torch.Tensor:
    cos, sin = math.cos(angle / 2), math.sin(angle / 2)
    return torch.tensor(((cos, -sin), (sin, cos)), dtype=DTYPE)


def hhl_reference(path: Path) -> torch.Tensor:
    source = path.read_text()
    [matrix] = literal_arrays(r"unitary\((\[.*?\n  \])", source)
    [angles] = literal_arrays(r"select_rotation\(Pauli::Y, (\[.*?\])", source)
    main = source[source.index("qfunc main(") :]
    hybrid = path.name == ACCEPTED[1]
    portfolio = path.name == ACCEPTED[2]
    if hybrid:
        b, other = literal_arrays(r"prepare_amplitudes\((\[.*?\])", main)
        n_qubits, phase, data, indicator = 10, (2, 3, 4, 5), (6, 7), 0
    elif portfolio:
        [b] = literal_arrays(r"prepare_amplitudes\((\[.*?\])", main)
        n_qubits, phase, data, indicator = 14, tuple(range(5, 11)), (1, 2, 3, 4), 0
    else:
        [b] = literal_arrays(r"my_hhl_expanded___0\((\[.*?\])", main)
        n_qubits, phase, data, indicator = 7, (2, 3, 4, 5), (0, 1), 6

    unitary = torch.tensor(matrix, dtype=DTYPE)
    qpe = fq.Circuit(n_qubits, dtype=DTYPE)
    for wire in phase:
        qpe.h(wire)
    powers = []
    for bit, wire in enumerate(reversed(phase)):
        block = torch.block_diag(
            torch.eye(1 << len(data), dtype=DTYPE),
            torch.linalg.matrix_power(unitary, 1 << bit),
        )
        qpe.any(wire, *data, matrix=block)
        powers.append((wire, block))
    size = 1 << len(phase)
    axis = torch.arange(size, dtype=torch.float64)
    inverse_fourier = torch.exp(
        -2j * math.pi * torch.outer(axis, axis) / size
    ) / math.sqrt(size)
    qpe.any(*phase, matrix=inverse_fourier)

    circuit = fq.Circuit(n_qubits, dtype=DTYPE)
    circuit.any(*data, matrix=preparation(b))
    circuit.compose(qpe)
    circuit.any(*phase, indicator, matrix=torch.block_diag(*(ry(a) for a in angles)))
    circuit.any(*phase, matrix=inverse_fourier.mH)
    for wire, block in reversed(powers):
        circuit.any(wire, *data, matrix=block.mH)
    for wire in phase:
        circuit.h(wire)
    if hybrid:
        circuit.any(8, 9, matrix=preparation(other))
        circuit.h(1).gate("cswap", (1, 6, 8)).gate("cswap", (1, 7, 9)).h(1)
    return fq.run(circuit).statevector()


@pytest.mark.parametrize("name", ACCEPTED)
def test_original_hhl_matches_all_complex_amplitudes(name: str) -> None:
    path = original_file(name)
    workspace = 3 if name == ACCEPTED[2] else 0
    program = synth.synthesize(path, max_workspace=workspace)
    actual = fq.run(program).statevector()
    torch.testing.assert_close(actual, hhl_reference(path), atol=1e-12, rtol=0)
    if name == ACCEPTED[0]:
        assert program.report["qmod_fractional_bits"] == {"phase": 4}
    elif name == ACCEPTED[1]:
        assert dict(program.registers) == {"indicator": (0,), "test": (1,)}
    else:
        assert program.report["qmod_fractional_bits"] == {"rescaled_eig": 6}
        dirty = actual.reshape(-1, 1 << workspace)[:, 1:]
        torch.testing.assert_close(dirty, torch.zeros_like(dirty), atol=1e-12, rtol=0)
    assert program.report["workspace_qubits"] == workspace
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(path, max_operations=2)


def test_zero_error_original_requires_normalized_amplitudes() -> None:
    path = original_file("hhl_example.qmod")
    [data] = literal_arrays(r"my_hhl_expanded___0\((\[.*?\])", path.read_text())
    assert abs(math.hypot(*data) - 1) > 1e-12
    with pytest.raises(InteropConversionError, match="normalized finite real vector"):
        synth.synthesize(path)


@pytest.mark.parametrize("axis", "XYZ")
def test_selector_rotation_axes_and_lsb_order(axis: str) -> None:
    angles = (0.0, 0.39, -0.12, 1.28)
    source = f"""
        qfunc select(const bits:qbit[2], target:qbit) {{
          select_rotation(Pauli::{axis}, {list(angles)}, bits, target);
        }}
        qfunc main(output q:qnum<2,False,2>, output t:qbit) {{
          allocate(q); allocate(t);
          prepare(q);
          H(t); S(t); select(q,t);
        }}
        qfunc prepare(bits:qbit[2]) {{ H(bits[0]); H(bits[1]); S(bits[1]); }}
    """
    program = synth.synthesize(source, source_format="qmod")
    pauli = {
        "X": [[0, 1], [1, 0]],
        "Y": [[0, -1j], [1j, 0]],
        "Z": [[1, 0], [0, -1]],
    }[axis]
    matrix = torch.tensor(pauli, dtype=DTYPE)
    reference = fq.Circuit(3, dtype=DTYPE).h(0).h(1).s(0).h(2).s(2)
    reference.any(
        0,
        1,
        2,
        matrix=torch.block_diag(
            *(torch.linalg.matrix_exp(-0.5j * a * matrix) for a in angles)
        ),
    )
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("setup", "diagnostic"),
    [
        ("select_rotation(Pauli::Y,[0.0,0.2],q,t);", "rotation table length"),
        (
            "select_rotation(Pauli::Y,[0.0,0.1,0.2,0.3],q,q[0]);",
            "locations must be distinct",
        ),
        ("CPHASE(0.2,q[1],q[1]);", "distinct qubit locations"),
        (
            "control(t) { repeat(i:2) { SWAP(q[i],t); } }",
            "controlled SWAP requires distinct",
        ),
    ],
)
def test_selector_alias_and_table_refusals(setup: str, diagnostic: str) -> None:
    source = (
        "qfunc main(output q:qbit[2],output t:qbit) {allocate(q);allocate(t);"
        + setup
        + "}"
    )
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize(
    ("helpers", "body", "diagnostic"),
    [
        (
            "qfunc prep(q:qnum<3,False,3>) {}",
            "allocate(q);allocate(t);prep(q);",
            "helper quantum array length disagrees",
        ),
        (
            "qfunc change(p:qbit[2]) {X(p[0]);} "
            "qfunc read(const p:qnum<2,False,2>, t:qbit) {change(p);}",
            "allocate(q);allocate(t);read(q,t);",
            "fixed-point helper ports require a checked bit view or matching type",
        ),
        (
            "qfunc fresh(q:qbit, output t:qbit) {allocate(t);}",
            "allocate(q);allocate(t);fresh(t,t);",
            "helper arguments must be disjoint",
        ),
        (
            "qfunc fresh(q:qbit, output t:qbit) {}",
            "allocate(q); control_bit:qbit; allocate(control_bit); fresh(control_bit,t); allocate(t);drop(control_bit);",
            "bound helper must initialize each Output",
        ),
        (
            "",
            "allocate(2,True,2,q);allocate(t);",
            "signedness must agree",
        ),
    ],
)
def test_fixed_point_width_borrow_and_output_contract(
    helpers: str, body: str, diagnostic: str
) -> None:
    source = (
        helpers + "qfunc main(output q:qnum<2,False,2>,output t:qbit) {" + body + "}"
    )
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")
