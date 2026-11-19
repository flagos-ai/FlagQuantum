"""Original Oracle Workshop truth tables, phases and actual local observations."""

import cmath
import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum import synth

from .ansatz import (
    auxiliary_kickback,
    bit_oracle,
    deutsch_jozsa,
    direct_kickback,
    grover,
)

STAGES = (
    "bit_oracle",
    "auxiliary_kickback",
    "direct_kickback",
    "deutsch_jozsa",
    "grover",
)


@dataclass(frozen=True)
class StateCheck:
    name: str
    program: synth.Program
    maximum_state_error: float
    dirty_workspace_probability: float


@dataclass(frozen=True)
class PhaseRow:
    x: int
    y: int
    magnitude: float
    angle_pi: float
    solution: bool
    amplitude: complex


@dataclass(frozen=True)
class Observation:
    values: tuple[int, ...]
    count: int
    frequency: float


def build_program(name: str) -> synth.Program:
    definitions = dict(
        bit_oracle=bit_oracle,
        auxiliary_kickback=auxiliary_kickback,
        direct_kickback=direct_kickback,
        deutsch_jozsa=deutsch_jozsa,
        grover=grover,
    )
    return synth.synthesize(definitions[name], memory_limit_bytes=64 * 1024**2)


def reference_circuit(name: str) -> fq.Circuit:
    if name == "bit_oracle":
        circuit = fq.Circuit(4, dtype=torch.complex128)
        for bit in range(3):
            circuit.h(bit)
        permutation = torch.zeros((16, 16), dtype=torch.complex128)
        for x in range(2):
            for y in range(4):
                for z in range(2):
                    permutation[
                        (x << 3) | (y << 1) | (z ^ int(2 * x + y == 4)),
                        (x << 3) | (y << 1) | z,
                    ] = 1
        return circuit.any(*range(4), matrix=permutation)
    if name in {"auxiliary_kickback", "direct_kickback"}:
        circuit = fq.Circuit(3, dtype=torch.complex128)
        for bit in range(3):
            circuit.h(bit)
        diagonal = torch.tensor(
            [-1 if 2 * (w >> 2) + (w & 3) == 4 else 1 for w in range(8)],
            dtype=torch.complex128,
        )
        return circuit.any(*range(3), matrix=torch.diag(diagonal))
    if name == "deutsch_jozsa":
        circuit = fq.Circuit(3, dtype=torch.complex128)
        for bit in range(3):
            circuit.h(bit)
        circuit.any(
            *range(3),
            matrix=torch.diag(
                torch.tensor(
                    [-1 if w % 2 == 0 else 1 for w in range(8)], dtype=torch.complex128
                )
            ),
        )
        for bit in range(3):
            circuit.h(bit)
        return circuit
    if name != "grover":
        raise ValueError("unknown Oracle Workshop stage")
    circuit = fq.Circuit(4, dtype=torch.complex128)
    for bit in range(4):
        circuit.h(bit)
    diagonal = torch.diag(
        torch.tensor(
            [-1 if (w >> 2) - (w & 3) == 2 else 1 for w in range(16)],
            dtype=torch.complex128,
        )
    )
    reflection = torch.ones((16, 16), dtype=torch.complex128) / 8 - torch.eye(
        16, dtype=torch.complex128
    )
    for _ in range(2):
        circuit.any(*range(4), matrix=diagonal)
        circuit.any(*range(4), matrix=reflection)
    return circuit


def declared_amplitudes(program: synth.Program) -> torch.Tensor:
    state = fq.run(program).statevector()
    assert isinstance(state, torch.Tensor)
    width = state.numel().bit_length() - 1
    logical = tuple(wire for wires in program.registers.values() for wire in wires)
    hidden = tuple(wire for wire in range(width) if wire not in logical)
    return (
        state.reshape((2,) * width)
        .permute((*logical, *hidden))
        .reshape(1 << len(logical), -1)
    )


def verify_program(name: str, program: synth.Program) -> StateCheck:
    amplitudes = declared_amplitudes(program)
    expected = fq.run(reference_circuit(name)).statevector()
    assert isinstance(expected, torch.Tensor)
    error = float((amplitudes[:, 0] - expected.flatten()).abs().max())
    dirty = float(amplitudes[:, 1:].abs().square().sum())
    if error > 2e-11 or dirty > 1e-22:
        raise AssertionError(
            "native absolute state or hidden workspace differs from independent Circuit"
        )
    return StateCheck(name, program, error, dirty)


def verify(name: str) -> StateCheck:
    return verify_program(name, build_program(name))


def phase_table(program: synth.Program) -> tuple[PhaseRow, ...]:
    if tuple(program.registers) != ("x", "y") or tuple(
        len(w) for w in program.registers.values()
    ) != (1, 2):
        raise ValueError("phase table needs one-bit x and two-bit y")
    amplitudes = declared_amplitudes(program)
    if float(amplitudes[:, 1:].abs().square().sum()) > 1e-22:
        raise AssertionError(
            "phase kickback auxiliary and compiler workspace must be zero"
        )
    return tuple(
        PhaseRow(x, y, abs(v), cmath.phase(v) / math.pi, 2 * x + y == 4, v)
        for x in range(2)
        for y in range(4)
        if abs(v := complex(amplitudes[4 * x + y, 0])) >= 1e-12
    )


def sample(
    program: synth.Program, *, shots: int = 1000, seed: int = 1970
) -> tuple[Observation, ...]:
    if type(shots) is not int or not 1 <= shots <= 100000:
        raise ValueError("shots must lie in 1..100000")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    probabilities = program.probabilities(
        fq.run(program), registers=tuple(program.registers)
    )
    keys = tuple(probabilities)
    counts = torch.bincount(
        torch.multinomial(
            torch.tensor(tuple(probabilities.values()), dtype=torch.float64),
            shots,
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        ),
        minlength=len(keys),
    )
    return tuple(
        Observation(values, int(counts[i]), int(counts[i]) / shots)
        for i, values in enumerate(keys)
        if int(counts[i])
    )


def classify_deutsch_jozsa(observations: tuple[Observation, ...]) -> str:
    if any(len(row.values) != 1 for row in observations):
        raise ValueError("Deutsch-Jozsa readout needs one named word")
    words = {row.values[0] for row in observations if row.count > 0}
    if len(words) == 1:
        return "constant" if 0 in words else "balanced"
    return "undetermined"
