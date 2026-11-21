"""Prepare the source 8-bit state and compile its full 127-bit extension."""

import math
from dataclasses import dataclass
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate, estimate_resources
from flagquantum.compiler.basis_conversion import convert_basis
from flagquantum.core.ir import CircuitIR

from .quantum import gaussian_tree

SIMULATED_WIDTH = 8
SYNTHESIS_WIDTH = 127
ROTATION_FLOOR = 1e-6
PHASE_FLOOR = 1e-2
TWEAK = -math.pi / 42
BASIS = ("cx", "u")


@dataclass(frozen=True)
class Compiled:
    raw: fq.Circuit
    program: CircuitIR
    resources: ResourceEstimate


@dataclass(frozen=True)
class Simulation:
    grid: tuple[float, ...]
    probabilities: tuple[float, ...]
    target: tuple[float, ...]
    mse: float
    source_state_error: float
    converted_state_error: float
    global_phase: float
    sample: float


def _angle(index: int) -> float:
    return 2 * math.atan(math.exp(-2.5 * index**2))


def native_program(resolution: int) -> synth.Program:
    """Build a bounded executable stage through named logical qfunc ports."""
    if type(resolution) is not int or not 2 <= resolution <= 18:
        raise ValueError("native executable stages support 2 through 18 qubits")
    return synth.synthesize(
        gaussian_tree,
        classical_inputs={
            "width": resolution,
            "rotations": [_angle(i) for i in range(resolution)],
            "phases": [math.ldexp(math.tau, -(i + 2)) for i in range(8)],
            "tweak": TWEAK,
        },
        max_qubits=resolution,
        max_workspace=0,
        memory_limit_bytes=16 * 2**18,
    )


def reference_circuit(resolution: int) -> fq.Circuit:
    """Retain the independent Circuit schedule and original compile-only stage."""
    output = fq.Circuit(resolution, dtype=torch.complex128)
    little_endian = tuple(range(resolution - 1, -1, -1))
    for i, wire in enumerate(little_endian):
        angle = _angle(i)
        if angle > ROTATION_FLOOR:
            output.ry(wire, theta=angle)
    output.cry(little_endian[0], little_endian[1], theta=TWEAK)

    for left in range(resolution // 2):
        output.swap(little_endian[left], little_endian[-1 - left])
    for j, wire in enumerate(little_endian):
        output.h(wire)
        for k in range(j + 1, resolution):
            phase = math.ldexp(math.tau, -(k - j + 1))
            if phase > PHASE_FLOOR:
                output.cphase(little_endian[k], wire, theta=phase)
    output.x(little_endian[0])
    return output


def build(resolution: int, *, ceiling: int) -> Compiled:
    """Use qfunc for executable widths; retain the explicit large Circuit stage."""
    if type(resolution) is not int or not 2 <= resolution <= ceiling <= SYNTHESIS_WIDTH:
        raise ValueError(
            "the original solution needs a width within its declared stage cap"
        )
    output = (
        fq.Circuit.from_ir(native_program(resolution).circuit, dtype=torch.complex128)
        if resolution <= 18
        else reference_circuit(resolution)
    )
    program = convert_basis(output, gates=BASIS, max_added_operations=20_000).program
    return Compiled(output, program, estimate_resources(program))


def _reference(width: int) -> torch.Tensor:
    """Evaluate all binary-path amplitudes independently of Circuit gate methods."""
    words = 2**width
    result = torch.zeros(words, dtype=torch.complex128)
    for basis in range(words):
        amplitude = 1.0
        for bit in range(width):
            coefficient = math.exp(-2.5 * bit**2)
            if _angle(bit) <= ROTATION_FLOOR:
                coefficient = 0.0
            amplitude *= (coefficient if (basis >> bit) & 1 else 1) / math.hypot(
                1, coefficient
            )
        result[basis] = amplitude
    for basis in range(words):
        if basis & 3 != 1:
            continue
        left, right = float(result[basis].real), float(result[basis | 2].real)
        cosine, sine = math.cos(TWEAK / 2), math.sin(TWEAK / 2)
        result[basis] = cosine * left - sine * right
        result[basis | 2] = sine * left + cosine * right

    reversals = torch.tensor(
        [int(f"{word:0{width}b}"[::-1], 2) for word in range(words)]
    )
    previous = result[reversals]
    for bit in range(width):
        stride = 1 << bit
        for low in range(words):
            if low & stride:
                continue
            high = low | stride
            even, odd = previous[low].clone(), previous[high].clone()
            previous[low] = (even + odd) / math.sqrt(2)
            previous[high] = (even - odd) / math.sqrt(2)
        for index in range(words):
            if index & stride:
                phase = math.fsum(
                    math.ldexp(math.tau, -(other - bit + 1))
                    for other in range(bit + 1, width)
                    if (index >> other) & 1
                    and math.ldexp(math.tau, -(other - bit + 1)) > PHASE_FLOOR
                )
                if phase:
                    previous[index] = previous[index] * torch.exp(
                        torch.tensor(1j * phase, dtype=torch.complex128)
                    )
    return previous[torch.arange(words) ^ 1]


def simulate(compiled: Compiled) -> Simulation:
    """Compare all 256 amplitudes and the signed Gaussian probabilities."""
    if compiled.program.n_wires != SIMULATED_WIDTH:
        raise ValueError("only the 8-qubit source stage is executed")
    width = SIMULATED_WIDTH
    expected = _reference(width)
    actual = compiled.raw.state().flatten()
    converted = fq.run(compiled.program).statevector().flatten()
    difference = float((actual - expected).abs().max())
    phase = float(torch.angle(actual.vdot(converted)))
    aligned = converted * torch.exp(torch.tensor(-1j * phase, dtype=torch.complex128))
    conversion_error = float((aligned - actual).abs().max())
    assert difference < 3e-12 and conversion_error < 3e-12
    probabilities = actual.abs().square()
    draw = int(
        torch.multinomial(
            probabilities, 1, generator=torch.Generator().manual_seed(19)
        ).item()
    )
    raw = torch.arange(2**width, dtype=torch.long)
    divisor = 2 ** (width - 2)
    signed = torch.where(raw >= (1 << (width - 1)), raw - (1 << width), raw)
    grid = signed.to(torch.float64) / divisor
    order = grid.argsort()
    grid = grid[order]
    probabilities = probabilities[order]
    target = torch.exp(-grid.square())
    target /= target.sum()
    mse = float((probabilities - target).square().mean())
    value = (draw - 2**width if draw >= 2 ** (width - 1) else draw) / divisor
    return Simulation(
        tuple(grid.tolist()),
        tuple(probabilities.tolist()),
        tuple(target.tolist()),
        mse,
        difference,
        conversion_error,
        phase,
        value,
    )


def save_program(compiled: Compiled, directory: Path, stage: int) -> Path:
    """Write the native portable circuit artifact used by each tutorial stage."""
    directory.mkdir(parents=True, exist_ok=True)
    path = (
        directory
        / f"Quantum_Tree__stage_{stage}__resolution_{compiled.program.n_wires}.json"
    )
    path.write_text(compiled.program.to_json() + "\n")
    restored = CircuitIR.from_json(path.read_text())
    assert restored.content_hash == compiled.program.content_hash
    return path
