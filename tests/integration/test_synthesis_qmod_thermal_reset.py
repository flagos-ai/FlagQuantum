"""Independent thermal-channel construction from the original unitary blocks."""

import ast
import hashlib
import json
import math
import re
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from flagquantum.errors import PlanningError
from flagquantum.runtime.execution_plan_contract import plan_program
from flagquantum.simulation.density_matrix import density_matrix_from_ir

DTYPE = torch.complex128
pytestmark = pytest.mark.integration
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
STATE = (0, 1)
BOLTZMANN = 2
BOHR = (3, 4, 5, 6)
JUMP = (7, 8)


def block_encoding(source: str) -> fq.Circuit:
    [gaussian_text] = re.findall(
        r"inplace_prepare_amplitudes\((\[[^\]]+\]), 0, t\);", source
    )
    gaussian = torch.tensor(ast.literal_eval(gaussian_text), dtype=DTYPE)
    assert abs(torch.linalg.vector_norm(gaussian).item() - 1) < 1e-13
    gaussian /= torch.linalg.vector_norm(gaussian)
    identity = torch.eye(16, dtype=DTYPE)
    axis = identity[:, 0] - gaussian
    preparation = identity - 2 * torch.outer(axis, axis.conj()) / axis.vdot(axis)
    block = fq.Circuit(9, dtype=DTYPE).any(*BOHR, matrix=preparation)

    energy = torch.tensor([2, 0, -2, 0], dtype=torch.float64)
    elapsed = 0.2 * math.pi
    assert "0.2 * pi" in source
    evolution = fq.Circuit(9, dtype=DTYPE)
    for bit, control in enumerate(reversed(BOHR)):
        phases = torch.diag(torch.exp(-1j * elapsed * (2**bit) * energy))
        evolution.any(
            control, *STATE, matrix=torch.block_diag(torch.eye(4, dtype=DTYPE), phases)
        )
    block.compose(evolution)

    x = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE)
    eye = torch.eye(2, dtype=DTYPE)
    select_jump = torch.block_diag(
        torch.kron(eye, x), torch.kron(eye, y), torch.kron(x, eye), torch.kron(y, eye)
    )
    block.h(JUMP[0]).h(JUMP[1]).any(*JUMP, *STATE, matrix=select_jump)
    block.compose(evolution.adjoint())

    qft = fq.Circuit(9, dtype=DTYPE)
    bohr_bits = tuple(reversed(BOHR))
    for bit in range(len(bohr_bits) // 2):
        qft.swap(bohr_bits[bit], bohr_bits[-1 - bit])
    for bit, target in enumerate(bohr_bits):
        qft.h(target)
        for other in range(bit + 1, len(bohr_bits)):
            qft.cphase(bohr_bits[other], target, theta=math.pi / 2 ** (other - bit))
    block.compose(qft.adjoint())

    [angle_text] = re.findall(r"select_rotation\(Pauli::Y,\s*(\[.*?\])", source, re.S)
    angles = ast.literal_eval(angle_text)
    assert len(angles) == 16
    rotations = []
    for angle in angles:
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        rotations.append(torch.tensor([[c, -s], [s, c]], dtype=DTYPE))
    block.any(*BOHR, BOLTZMANN, matrix=torch.block_diag(*rotations))
    return block.x(BOLTZMANN)


def kraus_operators(source: str, delta: float) -> torch.Tensor:
    block = block_encoding(source)
    inverse = block.adjoint()
    theta = math.asin(2 * math.sqrt(delta))
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    # The highest scratch bit in this layout is boltzmann_weight.
    low = ((torch.arange(512) >> 6) & 1) == 0
    columns = []
    for basis in range(4):
        circuit = fq.Circuit(9, dtype=DTYPE)
        for bit, wire in enumerate(reversed(STATE)):
            if basis >> bit & 1:
                circuit.x(wire)
        encoded = fq.run(circuit.compose(block)).statevector().reshape(-1)
        weighted = encoded * torch.where(low, torch.tensor(c, dtype=torch.float64), 1.0)
        recovered = (
            fq.Circuit(9, dtype=DTYPE, inputs=weighted.reshape(1, -1))
            .compose(inverse)
            .state()
            .reshape(-1)
        )
        jump = encoded * low * s
        columns.append(torch.stack((recovered.reshape(4, 128), jump.reshape(4, 128))))
    return torch.stack(columns, dim=-1).permute(0, 2, 1, 3).reshape(256, 4, 4)


def evolve(operators: torch.Tensor, rounds: int) -> torch.Tensor:
    rho = torch.zeros((4, 4), dtype=DTYPE)
    rho[0, 0] = 1
    for _ in range(rounds):
        rho = torch.einsum("kij,jl,kml->im", operators, rho, operators.conj())
    return rho


def _original() -> Path:
    manifest = json.loads((_FIXTURES / "thermal_reset_sources.json").read_text())
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


def _state_density(program, result) -> torch.Tensor:
    state = result.state
    assert program.registers["be.state"] == (0, 1)
    matrix = state.reshape(4, -1, 4, 1 << (program.circuit.n_wires - 2))
    density = matrix[:, 0, :, 0]
    # All seven published reset bits and the hidden delta and work bits are zero.
    torch.testing.assert_close(
        torch.einsum("aibi->ab", matrix),
        density,
        atol=1e-11,
        rtol=0,
    )
    assert torch.linalg.vector_norm(matrix[:, 1:, :, :]) < 1e-11
    assert torch.linalg.vector_norm(matrix[:, :, :, 1:]) < 1e-11
    return density


@pytest.mark.parametrize(("delta", "rounds"), [(0.1, 50), (0.2, 2)])
def test_thermal_reset_matches_independent_complete_density(
    delta: float, rounds: int
) -> None:
    path = _original()
    source = path.read_text()
    limits = dict(max_qubits=11, max_workspace=1, max_operations=25_000)
    if rounds == 50:
        program = synth.synthesize(path, **limits)
        assert program.report["resources"].n_operations == 22200
        assert program.report["resources"].depth == 19901
    else:
        modified = source.replace("repeat (i: 50)", f"repeat (i: {rounds})")
        modified = modified.replace(
            "delta_step_expanded___0(0.1,", f"delta_step_expanded___0({delta},"
        )
        program = synth.synthesize(modified, source_format="qmod", **limits)
    assert program.execution.state_mode == "density_matrix"
    assert program.report["resources"].channel_count == 8 * rounds
    assert program.report["workspace_qubits"] == 1
    result = fq.run(program)
    density = _state_density(program, result)
    operators = kraus_operators(source, delta)
    identity = torch.einsum("kji,kjl->il", operators.conj(), operators)
    torch.testing.assert_close(identity, torch.eye(4, dtype=DTYPE), atol=1e-11, rtol=0)
    expected = evolve(operators, rounds)
    torch.testing.assert_close(density, expected, atol=1e-10, rtol=0)
    probabilities = program.probabilities(result, registers=("be.state",))
    assert dict(probabilities) == pytest.approx(
        {(index,): expected[index, index].real.item() for index in range(4)},
        abs=1e-10,
    )


RESET_BELL = """
qfunc main(output q:qbit[2]) {
    allocate(q);
    H(q[0]); CX(q[0],q[1]); RESET(q[0]);
}
"""


def test_reset_bell_keeps_exact_mixed_state() -> None:
    program = synth.synthesize(RESET_BELL, source_format="qmod")
    result = fq.run(program)
    expected = torch.diag(torch.tensor([0.5, 0.0, 0.5, 0.0], dtype=DTYPE))
    torch.testing.assert_close(result.state.reshape(4, 4), expected, atol=1e-12, rtol=0)
    assert dict(program.probabilities(result, registers=("q",))) == pytest.approx(
        {(0,): 0.5, (2,): 0.5}
    )


@pytest.mark.parametrize(
    "body", ["invert {RESET(q[0]);}", "control(q[1]) {RESET(q[0]);}"]
)
def test_reset_refuses_nonunitary_composition(body: str) -> None:
    with pytest.raises(InteropConversionError, match="invert|control|reversible|scope"):
        synth.synthesize(RESET_BELL.replace("RESET(q[0]);", body), source_format="qmod")


def test_original_respects_operation_and_density_memory_limits() -> None:
    path = _original()
    with pytest.raises(ValueError, match="budget|operation"):
        synth.synthesize(path, max_operations=20_000)
    with pytest.raises(PlanningError, match="memory|budget"):
        synth.synthesize(path, max_workspace=1, memory_limit_bytes=1024**2)


def test_repeated_reset_kernel_matches_uncompressed_cpu_density() -> None:
    source = """qfunc main(output q:qbit[2]) {
        allocate(q);
        repeat (i: 3) { H(q[0]); CX(q[0],q[1]); RESET(q[0]); }
    }"""
    program = synth.synthesize(source, source_format="qmod")
    ir = plan_program(program.execution)
    assert ir.metadata["construction_reset_repetitions"] == 3
    dense = density_matrix_from_ir(replace(ir, metadata={}), dtype=DTYPE)
    torch.testing.assert_close(fq.run(program).state, dense, atol=1e-12, rtol=0)
