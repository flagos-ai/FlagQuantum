"""Source-size matrix QPE with exact and power-dependent Suzuki evolution."""

import itertools
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, Literal

import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    QNum,
    allocate,
    drop,
    prepare_amplitudes,
    qfunc,
    qpe,
    suzuki_trotter,
    unitary,
)

Route = Literal["exact", "approximate"]
MATRIX = np.array(
    [
        [0.38891555, 0.23315811, 0.21499372, 0.06119557],
        [0.23315811, 0.44435328, 0.25197881, -0.13087919],
        [0.21499372, 0.25197881, 0.44116509, -0.01961855],
        [0.06119557, -0.13087919, -0.01961855, 0.32556608],
    ],
    dtype=np.float64,
)
PAULIS = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.diag(np.array([1, -1], dtype=np.complex128)),
}
PAULI_TERMS = tuple(
    (a + b, float(np.trace(np.kron(PAULIS[a], PAULIS[b]) @ MATRIX).real / 4))
    for a, b in itertools.product("IXYZ", repeat=2)
    if abs(np.trace(np.kron(PAULIS[a], PAULIS[b]) @ MATRIX) / 4) > 1e-15
)
NORMALIZATION = 2 * sum(abs(coefficient) for _, coefficient in PAULI_TERMS)
PHASE_BITS = math.ceil(math.log2(NORMALIZATION / 0.02))
ENERGY_RESOLUTION = 2 * NORMALIZATION / (1 << PHASE_BITS)
EIGENVALUES, _eigenvectors = np.linalg.eigh(MATRIX)
_exact = (
    _eigenvectors * np.exp(2j * math.pi * EIGENVALUES / NORMALIZATION)
) @ _eigenvectors.conj().T
UNITARY = tuple(tuple(complex(value) for value in row) for row in _exact)
_initial = np.random.RandomState(8).rand(4)
INITIAL_AMPLITUDES = tuple(
    float(value) for value in _initial / np.linalg.norm(_initial)
)
REPETITIONS = tuple(math.ceil(2 * 1.5**bit) for bit in range(PHASE_BITS))


def normalized_operator() -> fq.Observable:
    """Translate ordered Pauli words to native logical low-to-high indices."""
    axes = {"I": fq.I, "X": fq.X, "Y": fq.Y, "Z": fq.Z}
    operator = 0 * fq.I(0)
    for word, coefficient in PAULI_TERMS:
        operator += (coefficient / NORMALIZATION) * (
            axes[word[0]](1) @ axes[word[1]](0)
        )
    return operator


NORMALIZED_OPERATOR = normalized_operator()


@qfunc
def exact(
    amplitudes: list[float], phase_result: Output[Annotated[QNum, (7, True, 7)]]
) -> None:
    state: QArray[QBit] = QArray()
    prepare_amplitudes(amplitudes, 0, state)
    allocate(phase_result)
    qpe(lambda: unitary(UNITARY, state), phase_result)
    drop(state)


@qfunc
def approximate(
    amplitudes: list[float],
    repetitions: list[int],
    phase_result: Output[Annotated[QNum, (7, True, 7)]],
) -> None:
    state: QArray[QBit] = QArray()
    prepare_amplitudes(amplitudes, 0, state)
    allocate(phase_result)
    qpe(
        lambda power: suzuki_trotter(
            NORMALIZED_OPERATOR, -2 * math.pi * power, 2, repetitions[power], state
        ),
        phase_result,
    )
    drop(state)


@dataclass(frozen=True)
class Outcome:
    """One observed signed phase and its energy in original matrix units."""

    phase: Fraction
    energy: float
    count: int
    probability: float


@dataclass(frozen=True)
class Estimate:
    """One actual native execution and its complete source readout workflow."""

    route: Route
    seed: int
    shots: int
    outcomes: tuple[Outcome, ...]
    resources: ResourceEstimate
    maximum_state_error: float

    @property
    def solutions(self) -> tuple[float, ...]:
        """Return the source's two most observed energy estimates."""
        return tuple(row.energy for row in self.outcomes[:2])

    @property
    def intervals(self) -> tuple[tuple[float, float], ...]:
        """Apply the source's plus/minus energy-resolution display convention."""
        return tuple(
            (value - ENERGY_RESOLUTION, value + ENERGY_RESOLUTION)
            for value in self.solutions
        )


def build_program(
    route: Route, amplitudes: tuple[float, ...] = INITIAL_AMPLITUDES
) -> synth.Program:
    """Synthesize the original seven-bit signed phase problem with no clean work.

    Examples:
        >>> build_program("approximate").circuit.n_wires
        9
    """
    if route not in ("exact", "approximate"):
        raise ValueError("use exact or approximate evolution")
    if (
        len(amplitudes) != 4
        or any(not math.isfinite(value) for value in amplitudes)
        or not math.isclose(math.hypot(*amplitudes), 1.0, abs_tol=1e-12)
    ):
        raise ValueError("use four finite normalized real amplitudes")
    if route == "exact":
        return synth.synthesize(
            exact,
            classical_inputs={"amplitudes": list(amplitudes)},
            max_qubits=9,
            max_workspace=0,
            memory_limit_bytes=64 * 1024**2,
        )
    table = [1] * 65
    for bit, repetitions in enumerate(REPETITIONS):
        table[1 << bit] = repetitions
    return synth.synthesize(
        approximate,
        classical_inputs={"amplitudes": list(amplitudes), "repetitions": table},
        max_qubits=9,
        max_workspace=0,
        memory_limit_bytes=64 * 1024**2,
    )


def reference_circuit(
    route: Route, amplitudes: tuple[float, ...] = INITIAL_AMPLITUDES
) -> fq.Circuit:
    """Assemble dense controlled matrices independently of synth lowering.

    Pauli exponentials use their analytic involution identity, avoiding repeated
    small-angle general matrix exponentials in the reference. Public phase
    storage occupies wires 0..6; the retained private data occupies wires 7,8.
    """
    if route not in ("exact", "approximate"):
        raise ValueError("use exact or approximate evolution")
    circuit = fq.Circuit(9, dtype=torch.complex128)
    vector = torch.tensor(amplitudes, dtype=torch.complex128)
    first = torch.zeros(4, dtype=torch.complex128)
    first[0] = 1
    difference = first - vector
    identity = torch.eye(4, dtype=torch.complex128)
    norm = torch.vdot(difference, difference)
    preparation = (
        identity
        if abs(norm) < 1e-30
        else identity - 2 * torch.outer(difference, difference.conj()) / norm
    )
    circuit.any(7, 8, matrix=preparation)
    for wire in range(7):
        circuit.h(wire)
    for bit, repetitions in enumerate(REPETITIONS):
        power = 1 << bit
        if route == "exact":
            eigenvalues, vectors = torch.linalg.eigh(
                torch.tensor(MATRIX, dtype=torch.complex128)
            )
            evolution = (
                vectors * torch.exp(2j * math.pi * power * eigenvalues / NORMALIZATION)
            ) @ vectors.mH
        else:
            evolution = identity.clone()
            for _ in range(repetitions):
                for word, coefficient in (*PAULI_TERMS, *reversed(PAULI_TERMS)):
                    pauli = torch.tensor(
                        np.kron(PAULIS[word[0]], PAULIS[word[1]]),
                        dtype=torch.complex128,
                    )
                    angle = (
                        math.pi * power * coefficient / (NORMALIZATION * repetitions)
                    )
                    evolution = (
                        math.cos(angle) * identity + 1j * math.sin(angle) * pauli
                    ) @ evolution
        controlled = torch.eye(8, dtype=torch.complex128)
        controlled[4:, 4:] = evolution
        circuit.any(6 - bit, 7, 8, matrix=controlled)
    indices = torch.arange(128, dtype=torch.float64)
    inverse = torch.exp(
        -2j * math.pi * torch.outer(indices, indices) / 128
    ) / math.sqrt(128)
    circuit.any(*range(7), matrix=inverse)
    return circuit


def run_estimate(route: Route, *, shots: int = 10000, seed: int = 1400) -> Estimate:
    """Validate all 512 amplitudes, then sample and decode named numeric output."""
    if type(shots) is not int or not 2 <= shots <= 100000:
        raise ValueError("use 2..100000 shots")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("use an integer seed in 0..2**32-1")
    program = build_program(route)
    result = fq.run(program)
    actual = result.statevector().flatten()
    expected = fq.run(reference_circuit(route)).statevector().flatten()
    maximum_error = float((actual - expected).abs().max())
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=0)
    # Sample the physical joint distribution, including the retained environment.
    draws = torch.multinomial(
        actual.abs().square(),
        shots,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(draws // 4, minlength=128)
    numeric = dict(program.numeric_probabilities(result, registers=("phase_result",)))
    rows = []
    for word, count in enumerate(counts.tolist()):
        if count:
            phase = Fraction(word if word < 64 else word - 128, 128)
            if (phase,) not in numeric:
                raise AssertionError(
                    "sampled phase is absent from named numeric readout"
                )
            rows.append(
                Outcome(phase, float(phase) * NORMALIZATION, count, count / shots)
            )
    rows.sort(key=lambda row: (-row.count, row.phase))
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    return Estimate(route, seed, shots, tuple(rows), resources, maximum_error)


def plot_estimates(exact_result: Estimate, approximate_result: Estimate) -> Figure:
    """Plot both sampled distributions and every classical eigenvalue window."""
    figure = Figure(figsize=(9, 4.5), layout="constrained")
    axis = figure.subplots()
    height = max(row.probability for row in exact_result.outcomes)
    for eigenvalue in EIGENVALUES:
        axis.add_patch(
            Rectangle(
                (float(eigenvalue) - ENERGY_RESOLUTION / 2, 0),
                ENERGY_RESOLUTION,
                height,
                color="gray",
                alpha=0.3,
            )
        )
        axis.plot([eigenvalue, eigenvalue], [0, height], "--r")
    for result in (exact_result, approximate_result):
        axis.plot(
            [row.energy for row in result.outcomes],
            [row.probability for row in result.outcomes],
            "o",
            label=result.route + " exponentiation",
        )
    axis.set_xlabel("Eigenvalue")
    axis.set_ylabel("Observed probability")
    axis.legend()
    return figure
