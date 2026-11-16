"""Independent Fourier/block-encoding reference for the full 192-phase original."""

import hashlib
import json
import re
from math import atan2, cos, hypot, pi, sin, sqrt
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from flagquantum.simulation.statevector.basis_workspace import compact_workspace_state

pytestmark = pytest.mark.integration
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"

BASIS = torch.arange(1 << 15)
AUX = 0
BLOCK = tuple(range(1, 7))
Y = tuple(range(7, 11))
X = tuple(range(11, 15))
_x = (BASIS >> X[0]) & 15
_y = (BASIS >> Y[0]) & 15
_select = (BASIS >> BLOCK[4]) & 3
_axis = (BASIS >> BLOCK[1]) & 1
_direction = (BASIS >> BLOCK[0]) & 1
_SCATTER = ((_y >> 1) == 3) & ((_x >> 1) != 2) & ((_x >> 1) != 5)
_BOUNDARY_X = (_x == 0) | (_x == 15)
_BOUNDARY_Y = (_y == 0) | (_y == 15)


def mask_phase(active, theta, state):
    return state * torch.exp(1j * theta * active.to(torch.float64))


def flip_if(state, wire, active):
    return state.index_select(0, BASIS ^ (active.to(torch.int64) << wire))


def rotate(state, wire, active, angle):
    cosine, sine = cos(angle / 2), sin(angle / 2)
    target = (BASIS >> wire) & 1
    return torch.where(
        torch.as_tensor(active, dtype=torch.bool),
        state * cosine
        + state.index_select(0, BASIS ^ (1 << wire)) * sine * (2 * target - 1),
        state,
    )


def hadamard(state, wire, active=True):
    return torch.where(
        torch.as_tensor(active, dtype=torch.bool),
        (
            state * (1 - 2 * ((BASIS >> wire) & 1))
            + state.index_select(0, BASIS ^ (1 << wire))
        )
        / sqrt(2),
        state,
    )


def prepare_selector(state, amplitudes, adjoint=False):
    a, b, c, d = amplitudes
    angles = (2 * atan2(hypot(c, d), hypot(a, b)), 2 * atan2(b, a), 2 * atan2(d, c))
    steps = (
        (BLOCK[5], True, angles[0]),
        (BLOCK[4], _select < 2, angles[1]),
        (BLOCK[4], _select >= 2, angles[2]),
    )
    for wire, active, angle in reversed(steps) if adjoint else steps:
        state = rotate(state, wire, active, -angle if adjoint else angle)
    return state


def shift(state, adjoint=False):
    for axis, values, wires in ((0, _x, X), (1, _y, Y)):
        mask = (_select == 0) & (_axis == axis)
        delta = torch.where(_direction.bool(), -1, 1)
        inverse_index = (values - (-delta if adjoint else delta)) % 16
        state = state.index_select(
            0, BASIS ^ torch.where(mask, (values ^ inverse_index) << wires[0], 0)
        )
    return state


def semi_laplacian(state, adjoint=False):
    active = _select == 0
    boundary = active & (_SCATTER | _BOUNDARY_X | _BOUNDARY_Y)
    scatter = active & _SCATTER
    if adjoint:
        state = flip_if(state, BLOCK[3], boundary)
    else:
        state = flip_if(state, BLOCK[2], scatter)
    state = hadamard(state, BLOCK[1], active)
    state = hadamard(state, BLOCK[0], active)
    state = shift(state, adjoint)
    state = hadamard(state, BLOCK[0], active)
    state = hadamard(state, BLOCK[1], active)
    return flip_if(
        state, BLOCK[2] if adjoint else BLOCK[3], scatter if adjoint else boundary
    )


def semi_absorbing(state, adjoint=False):
    active = _select == 1
    parity = active & ~(_BOUNDARY_X ^ _BOUNDARY_Y)
    if adjoint:
        state = flip_if(state, BLOCK[2], parity)
    state = hadamard(state, BLOCK[1], active)
    for axis, wire, boundary in ((0, X[0], _BOUNDARY_X), (1, Y[0], _BOUNDARY_Y)):
        selected = active & (_axis == axis)
        if adjoint:
            state = flip_if(state, BLOCK[0], selected & ~boundary)
            state = flip_if(state, wire, selected)
        else:
            state = flip_if(state, wire, selected)
            state = flip_if(state, BLOCK[0], selected & ~boundary)
    state = hadamard(state, BLOCK[1], active)
    return state if adjoint else flip_if(state, BLOCK[2], parity)


def hamiltonian(state, amplitudes, phase, adjoint=False):
    state = prepare_selector(state, amplitudes)
    state = semi_laplacian(state, adjoint)
    state = semi_absorbing(state, adjoint)
    sign = -1 if adjoint else 1
    state = mask_phase(_select == 1, sign * phase, state)
    state = mask_phase(_select == 2, sign * pi, state)
    return prepare_selector(state, amplitudes, adjoint=True)


def project(state, angle):
    aux_after_compute = (BASIS & 1) ^ (((BASIS >> 1) & 63) == 0)
    return state * torch.exp(1j * angle * (aux_after_compute.to(torch.float64) - 0.5))


def reference(source: str, cycles: int = 95) -> torch.Tensor:
    phases = [
        float(x.strip().strip("()"))
        for x in re.search(r"qsvt_inversion_expanded___0\(\[([^]]+)\]", source)
        .group(1)
        .split(",")
    ]
    assert len(phases) == 192
    amplitudes = [
        float(x)
        for x in re.search(
            r"inplace_prepare_amplitudes\(\[([^]]+)\], 0.0, block2", source
        )
        .group(1)
        .split(",")
    ]
    [absorbing_phase] = re.findall(r"U\(0, 0, 0, ([\d.]+), block1\[0\]\)", source)
    state = torch.zeros(1 << 15, dtype=torch.complex128)
    for word in (0, 1):
        mask = (word ^ (word >> 7)) & 1
        data = word ^ mask ^ (mask << 7)
        data ^= 112 if data & 128 else 0
        data ^= 130
        state[data << 7] = 1 / sqrt(2)
    state = project(hadamard(state, AUX), phases[0])
    for index in range(cycles):
        state = project(
            hamiltonian(state, amplitudes, float(absorbing_phase), adjoint=True),
            phases[2 * index + 1],
        )
        state = project(
            hamiltonian(state, amplitudes, float(absorbing_phase)),
            phases[2 * index + 2],
        )
    state = project(
        hamiltonian(state, amplitudes, float(absorbing_phase), adjoint=True),
        phases[191],
    )
    return hadamard(state, AUX)


def _original() -> Path:
    manifest = json.loads(
        (_FIXTURES / "double_slit_gate_tape_sources.json").read_text()
    )
    [entry] = manifest["fixtures"]
    original = _FIXTURES / entry["file"]
    raw = original.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == entry["git_blob_sha1"]
    )
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    assert (
        hashlib.sha256((_FIXTURES / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )
    return original


@pytest.mark.parametrize("cycles", [95, 1])
def test_double_slit_matches_complete_independent_state(cycles: int) -> None:
    original = _original()
    source = original.read_text()
    budgets = dict(max_qubits=24, max_workspace=9, max_operations=100_000)
    if cycles == 95:
        program = synth.synthesize(original, **budgets)
        resources = program.report["resources"]
        assert resources.n_operations == 66509
        assert resources.n_qubits == 24
    else:
        source = source.replace("repeat (index: 95)", "repeat (index: 1)")
        source = source.replace("-598.477076962426", "-0.271")
        program = synth.synthesize(source, source_format="qmod", **budgets)
    assert program.registers == {"res": tuple(range(8)), "block": tuple(range(8, 15))}
    result = fq.run(program)
    actual = result.statevector().reshape(1 << 15, 1 << 9)
    expected = reference(source, cycles)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-11, rtol=0)
    assert torch.count_nonzero(actual[:, 1:]) == 0
    assert result.plan.state_bytes == 256 * 1024**2


def test_compact_scope_rejects_dirty_dependencies_and_budgets() -> None:
    short = _original().read_text().replace("repeat (index: 95)", "repeat (index: 1)")
    mutated = short.replace(
        "equals_expanded___3({temp_x", "X(e_field.x[2]); equals_expanded___3({temp_x", 1
    )
    with pytest.raises(InteropConversionError, match="computed work dependency"):
        synth.synthesize(mutated, source_format="qmod")
    with pytest.raises(ValueError, match="budget|operation"):
        synth.synthesize(short, source_format="qmod", max_operations=100)
    with pytest.raises(ValueError, match="memory"):
        synth.synthesize(short, source_format="qmod", memory_limit_bytes=128 * 1024**2)


def test_workspace_optimizer_checks_actual_gate_invariants() -> None:
    from dataclasses import replace

    circuit = fq.Circuit(3, dtype=torch.complex128)
    circuit.h(0).cx(0, 2).cz(1, 2).cx(0, 2).h(0)
    ir = replace(circuit.to_ir(), metadata={"construction_basis_workspace_prefix": 2})
    returned = compact_workspace_state(ir)
    assert returned is not None
    torch.testing.assert_close(returned, circuit.state(), atol=1e-12, rtol=0)
    dirty = replace(ir, instructions=ir.instructions[:-2])
    assert compact_workspace_state(dirty) is None
    unsafe = replace(ir, instructions=(*ir.instructions[:2], ir.instructions[-1]))
    assert compact_workspace_state(unsafe) is None
