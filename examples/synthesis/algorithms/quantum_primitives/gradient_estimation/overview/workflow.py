"""Original gradient models,independent Circuits and measured success statistics."""

import cmath
import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum import synth

from .ansatz import gradient_2d

M_VALUES = (
    0.05,
    0.1,
    0.33,
    0.5,
    0.67,
    1.0,
    1.33,
    1.67,
    2.0,
    2.33,
    2.67,
    3.0,
    3.33,
    3.67,
    4.0,
    6.0,
    10.0,
)
L_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0, 3.0)


@dataclass(frozen=True)
class Model:
    quadratic: float
    linear: float
    constant: float
    window: float = 0.5
    m: float = 2.0

    def __post_init__(self) -> None:
        if (
            not all(
                type(v) in (int, float) and math.isfinite(v)
                for v in (
                    self.quadratic,
                    self.linear,
                    self.constant,
                    self.window,
                    self.m,
                )
            )
            or self.window <= 0
            or self.m <= 0
        ):
            raise ValueError("finite coefficients and positive l,m are required")

    def normalized(self, index: float) -> float:
        x = self.window / 8 * index
        return (
            (self.quadratic * x * x + self.linear * x + self.constant)
            * 8
            / (self.window * self.m)
        )

    @property
    def coefficients(self) -> tuple[float, ...]:
        return (
            self.quadratic * self.window / (8 * self.m),
            self.linear / self.m,
            self.constant * 8 / (self.window * self.m),
        )


@dataclass(frozen=True)
class StateCheck:
    maximum_state_error: float
    dirty_workspace_probability: float


@dataclass(frozen=True)
class Observation:
    values: tuple[int, ...]
    count: int
    frequency: float


@dataclass(frozen=True)
class Success:
    rate: float
    successes: int
    shots: int
    majority_gradient: tuple[float, ...]
    analytical_gradient: tuple[float, ...]
    majority_correct: bool


@dataclass(frozen=True)
class PhaseRow:
    x: int
    theoretical: float
    unwrapped_phase: float


def signed(word: int) -> int:
    return word if word < 4 else word - 8


def declared_amplitudes(program: synth.Program) -> torch.Tensor:
    state = fq.run(program).statevector()
    assert isinstance(state, torch.Tensor)
    width = state.numel().bit_length() - 1
    logical = tuple(w for wires in program.registers.values() for w in wires)
    hidden = tuple(w for w in range(width) if w not in logical)
    return (
        state.reshape((2,) * width)
        .permute((*logical, *hidden))
        .reshape(1 << len(logical), -1)
    )


def reference_circuit(model: Model, *, inverse_fourier: bool) -> fq.Circuit:
    circuit = fq.Circuit(3, dtype=torch.complex128)
    for bit in range(3):
        circuit.h(bit)
    circuit.any(
        0,
        1,
        2,
        matrix=torch.diag(
            torch.tensor(
                [
                    cmath.exp(2j * math.pi * model.normalized(signed(w)))
                    for w in range(8)
                ],
                dtype=torch.complex128,
            )
        ),
    )
    if inverse_fourier:
        axis = torch.arange(8, dtype=torch.float64)
        circuit.any(
            0,
            1,
            2,
            matrix=torch.exp(-2j * math.pi * torch.outer(axis, axis) / 8)
            / math.sqrt(8),
        )
    return circuit


def auxiliary_reference() -> fq.Circuit:
    circuit = fq.Circuit(6, dtype=torch.complex128)
    for wire in range(3):
        circuit.h(wire)
    for wire in range(3, 6):
        circuit.x(wire)
    axis = torch.arange(8, dtype=torch.float64)
    fourier = torch.exp(2j * math.pi * torch.outer(axis, axis) / 8) / math.sqrt(8)
    circuit.any(3, 4, 5, matrix=fourier)
    permutation = torch.zeros((64, 64), dtype=torch.complex128)
    for x in range(8):
        delta = 2 * signed(x) + 16
        for ancilla in range(8):
            permutation[8 * x + (ancilla + delta) % 8, 8 * x + ancilla] = 1
    circuit.any(*range(6), matrix=permutation)
    circuit.any(3, 4, 5, matrix=fourier.mH)
    for wire in range(3, 6):
        circuit.x(wire)
    return circuit


def verify_program(
    program: synth.Program, model: Model, *, inverse_fourier: bool
) -> StateCheck:
    a = declared_amplitudes(program)
    reference = fq.run(
        reference_circuit(model, inverse_fourier=inverse_fourier)
    ).statevector()
    assert isinstance(reference, torch.Tensor)
    error = float((a[:, 0] - reference.flatten()).abs().max())
    dirty = float(a[:, 1:].abs().square().sum())
    if error > 2e-11 or dirty > 1e-22:
        raise AssertionError(
            "absolute phase or clean hidden workspace differs from independent Circuit"
        )
    return StateCheck(error, dirty)


def verify_auxiliary(program: synth.Program) -> StateCheck:
    actual = declared_amplitudes(program)
    reference = fq.run(auxiliary_reference()).statevector()
    assert isinstance(reference, torch.Tensor)
    error = float((actual - reference.reshape(8, 8)).abs().max())
    dirty = float(actual[:, 1:].abs().square().sum())
    if error > 2e-11 or dirty > 1e-22:
        raise AssertionError(
            "auxiliary kickback or cleanup differs from modular-add Circuit"
        )
    return StateCheck(error, dirty)


def sample(
    program: synth.Program, *, shots: int = 1000, seed: int = 2026
) -> tuple[Observation, ...]:
    if (
        type(shots) is not int
        or not 1 <= shots <= 100000
        or type(seed) is not int
        or not 0 <= seed < 2**63
    ):
        raise ValueError("use 1..100000 shots and a nonnegative 63-bit seed")
    probabilities = program.numeric_probabilities(
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
        sorted(
            (
                Observation(
                    tuple(int(v) for v in key), int(counts[i]), int(counts[i]) / shots
                )
                for i, key in enumerate(keys)
                if int(counts[i])
            ),
            key=lambda row: (-row.count, row.values),
        )
    )


def success_rate(
    observations: tuple[Observation, ...],
    model: Model,
    analytical: tuple[float, ...],
    *,
    tolerance: float | None = None
) -> Success:
    threshold = model.m / 16 if tolerance is None else tolerance
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("tolerance must be positive and finite")
    if any(len(row.values) != len(analytical) for row in observations):
        raise ValueError("derivative dimensions must match observed named words")
    shots = sum(row.count for row in observations)
    successes = sum(
        row.count
        for row in observations
        if all(
            abs(v * model.m / 8 - a) < threshold
            for v, a in zip(row.values, analytical, strict=True)
        )
    )
    majority = (
        tuple(v * model.m / 8 for v in observations[0].values) if observations else ()
    )
    correct = bool(majority) and all(
        abs(v - a) < model.m / 16 for v, a in zip(majority, analytical, strict=True)
    )
    return Success(
        successes / shots if shots else 0.0,
        successes,
        shots,
        majority,
        analytical,
        correct,
    )


def phase_table(program: synth.Program, model: Model) -> tuple[PhaseRow, ...]:
    state = declared_amplitudes(program)[:, 0]
    raw = [
        round(cmath.phase(complex(state[word])) / (2 * math.pi), 5)
        for word in (4, 5, 6, 7, 0, 1, 2, 3)
    ]
    unwrapped = [raw[0]]
    correction = 0.0
    for previous, current in zip(raw, raw[1:], strict=False):
        delta = current - previous
        adjusted = (delta + 0.5) % 1 - 0.5
        if adjusted == -0.5 and delta > 0:
            adjusted = 0.5
        if abs(delta) >= 0.5:
            correction += adjusted - delta
        unwrapped.append(current + correction)
    return tuple(
        PhaseRow(
            x, model.normalized(x) - model.normalized(0), unwrapped[i] - unwrapped[4]
        )
        for i, x in enumerate(range(-4, 4))
    )


def build_two_dimensional() -> synth.Program:
    window, m = 0.1, 2.0
    coefficients = (
        0.6 * window / (8 * m),
        -0.4 * window / (8 * m),
        0.25 * window / (8 * m),
        0.25 / m,
        -0.5 / m,
        0.1 * 8 / (window * m),
    )
    return synth.synthesize(
        gradient_2d,
        classical_inputs={"coefficients": list(coefficients)},
        max_workspace=0,
        memory_limit_bytes=64 * 1024**2,
    )


def verify_two_dimensional(program: synth.Program) -> StateCheck:
    expected = fq.Circuit(6, dtype=torch.complex128)
    for bit in range(6):
        expected.h(bit)
    phases = []
    for x in range(8):
        for y in range(8):
            u, v = 0.1 / 8 * signed(x), 0.1 / 8 * signed(y)
            phases.append(
                (0.6 * u * u - 0.4 * v * v + 0.25 * u * v + 0.25 * u - 0.5 * v + 0.1)
                * 8
                / (0.1 * 2)
            )
    expected.any(
        *range(6),
        matrix=torch.diag(
            torch.exp(2j * math.pi * torch.tensor(phases, dtype=torch.float64))
        )
    )
    axis = torch.arange(8, dtype=torch.float64)
    fourier = torch.exp(-2j * math.pi * torch.outer(axis, axis) / 8) / math.sqrt(8)
    expected.any(0, 1, 2, matrix=fourier).any(3, 4, 5, matrix=fourier)
    reference = fq.run(expected).statevector()
    assert isinstance(reference, torch.Tensor)
    a = declared_amplitudes(program)
    error = float((a[:, 0] - reference.flatten()).abs().max())
    dirty = float(a[:, 1:].abs().square().sum())
    if error > 2e-11 or dirty > 1e-22:
        raise AssertionError(
            "two-dimensional absolute state differs from independent Circuit"
        )
    return StateCheck(error, dirty)
