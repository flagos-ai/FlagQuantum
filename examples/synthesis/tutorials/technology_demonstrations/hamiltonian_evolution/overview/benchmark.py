"""Run the seven original-size water Hamiltonian evolution benchmark cases."""

import math
from dataclasses import dataclass
from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.pauli_evolution import (
    Term,
    load_terms,
    reference_circuit,
    reference_sequence,
)
from examples.synthesis.tutorials.basics.exponentiation.example_exponentiation.overview.model import (
    WIDTH,
    build_model,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate, estimate_resources
from flagquantum.compiler.basis_conversion import convert_basis
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    allocate,
    qdrift,
    qfunc,
    suzuki_trotter,
)

TARGET_BASIS = ("cx", "u")
HISTORICAL_QISKIT_CX = (25164, 33508, 41882, 186067, 248232, 3581, 7404)
HISTORICAL_QISKIT_DEPTH = (30627, 42879, 53581, 300924, 402125, 3463, 7141)
SEED = 0
CONTROL = WIDTH
SOURCE_HAMILTONIAN = Path(__file__).resolve().parents[1] / "hamiltonian.json"
MAX_NATIVE_OPERATIONS = 100_000
MAX_ADDED_OPERATIONS = 800_000


@dataclass(frozen=True)
class Model:
    observable: fq.Observable
    terms: tuple[Term, ...]
    rhf_energy: float
    coefficient_error: float


@dataclass(frozen=True)
class Result:
    label: str
    native: ResourceEstimate
    target: ResourceEstimate
    reference_error: float
    conversion_error: float


def fresh_model(reference_path: Path = SOURCE_HAMILTONIAN) -> Model:
    """Regenerate coefficients and align the source's reversed logical label order."""
    generated = build_model()
    width, expected = load_terms(reference_path)
    terms = tuple(
        (coefficient, tuple(sorted((WIDTH - 1 - bit, axis) for bit, axis in support)))
        for coefficient, support in generated.terms
    )
    if width != WIDTH or len(terms) != len(expected):
        raise ValueError("The benchmark uses the full 551-term 12-bit Hamiltonian")
    if any(a[1] != tuple(sorted(b[1])) for a, b in zip(terms, expected, strict=True)):
        raise ValueError("The regenerated term order must match the benchmark source")
    error = max(abs(a[0] - b[0]) for a, b in zip(terms, expected, strict=True))
    if error > 1e-6:
        raise ValueError("The regenerated coefficients exceed the source tolerance")
    axes = {"X": fq.X, "Y": fq.Y, "Z": fq.Z}
    observable = 0.0 * fq.I()
    for coefficient, support in terms:
        operator = fq.I()
        for bit, axis in support:
            operator = operator @ axes[axis](bit)
        observable += coefficient * operator
    return Model(observable, terms, generated.rhf_energy, error)


def build_step(observable: fq.Observable, time: float, order: int) -> fq.Circuit:
    @qfunc
    def step(time: float, order: int, state: Output[QArray[QBit]]) -> None:
        allocate(12, state)
        suzuki_trotter(observable, time, order, 1, state)

    program = synth.synthesize(
        step,
        classical_inputs={"time": time, "order": order},
        max_qubits=WIDTH,
        max_workspace=0,
        max_operations=50_000,
        memory_limit_bytes=64 * 1024**2,
    )
    return fq.Circuit.from_ir(program.circuit, dtype=torch.complex128)


def build_qdrift(observable: fq.Observable, samples: int) -> fq.Circuit:
    @qfunc
    def evolution(samples: int, state: Output[QArray[QBit]]) -> None:
        allocate(12, state)
        qdrift(observable, 1.0, samples, state, 0)

    program = synth.synthesize(
        evolution,
        classical_inputs={"samples": samples},
        max_qubits=WIDTH,
        max_workspace=0,
        max_operations=50_000,
        memory_limit_bytes=64 * 1024**2,
    )
    return fq.Circuit.from_ir(program.circuit, dtype=torch.complex128)


def fourth_order_times() -> tuple[float, ...]:
    """Return Suzuki's symmetric five-part fourth-order composition at time one."""
    factor = 1 / (4 - math.pow(4, 1 / 3))
    return factor, factor, 1 - 4 * factor, factor, factor


def repeated_product(step: fq.Circuit, repetitions: int) -> fq.Circuit:
    full = fq.Circuit(WIDTH, dtype=torch.complex128)
    for _ in range(repetitions):
        full.compose(step)
    return full


def trotter_reference(
    terms: tuple[Term, ...],
    times: tuple[float, ...],
    *,
    symmetric: bool,
) -> tuple[tuple[int, float], ...]:
    """Expand the mathematical product independently of synthesis scheduling."""
    factors: list[tuple[int, float]] = []
    for time in times:
        if symmetric:
            half = tuple(
                (index, time * term[0] / 2) for index, term in enumerate(terms)
            )
            factors.extend(half)
            factors.extend(reversed(half))
        else:
            factors.extend((index, time * term[0]) for index, term in enumerate(terms))
    return tuple(factors)


def controlled_product(
    terms: tuple[Term, ...],
    factors: tuple[tuple[int, float], ...],
) -> fq.Circuit:
    """Control each diagonal Pauli phase between unconditional parity changes."""
    circuit = fq.Circuit(WIDTH + 1, dtype=torch.complex128)
    for index, angle in factors:
        support = tuple((WIDTH - 1 - bit, axis) for bit, axis in terms[index][1])
        if not support:
            circuit.phase(CONTROL, theta=-angle)
            continue
        anchor = support[0][0]
        for wire, axis in support:
            if axis == "X":
                circuit.h(wire)
            elif axis == "Y":
                circuit.rx(wire, theta=math.pi / 2)
        for wire, _ in support[1:]:
            circuit.cx(wire, anchor)
        circuit.crz(CONTROL, anchor, theta=2 * angle)
        for wire, _ in reversed(support[1:]):
            circuit.cx(wire, anchor)
        for wire, axis in reversed(support):
            if axis == "X":
                circuit.h(wire)
            elif axis == "Y":
                circuit.rx(wire, theta=-math.pi / 2)
    return circuit


def checked(
    label: str,
    circuit: fq.Circuit,
    expected: torch.Tensor,
    *,
    controlled: bool = False,
) -> Result:
    native = estimate_resources(circuit)
    if (
        native.n_qubits != WIDTH + int(controlled)
        or native.n_operations > MAX_NATIVE_OPERATIONS
    ):
        raise ValueError("The benchmark circuit exceeded its declared resource bounds")
    converted = convert_basis(
        circuit, gates=TARGET_BASIS, max_added_operations=MAX_ADDED_OPERATIONS
    )
    target = estimate_resources(converted.program)
    in_basis = fq.Circuit.from_ir(converted.program, dtype=torch.complex128)
    expected = expected.flatten()
    if controlled:
        zero = torch.zeros_like(expected)
        zero[0] = 1
        expected = torch.stack((zero, expected), dim=1).flatten() / math.sqrt(2)
        # Execute both branches coherently to catch a wrong controlled global phase.
        prep = fq.Circuit(WIDTH + 1, dtype=torch.complex128).h(CONTROL)
        circuit = (
            fq.Circuit(WIDTH + 1, dtype=torch.complex128).compose(prep).compose(circuit)
        )
        in_basis = (
            fq.Circuit(WIDTH + 1, dtype=torch.complex128)
            .compose(prep)
            .compose(in_basis)
        )

    actual = circuit.state().flatten()
    lowered = in_basis.state().flatten()
    phase = torch.vdot(lowered, actual)
    if float(phase.abs()) < 0.99:
        raise ValueError("Basis conversion lost the reference state")
    reference_error = float((actual - expected).abs().max())
    conversion_error = float((lowered * (phase / phase.abs()) - actual).abs().max())
    if reference_error > 2e-10 or conversion_error > 2e-10:
        raise ValueError(
            f"Full-state verification failed for {label}: {reference_error=}, {conversion_error=}"
        )
    return Result(label, native, target, reference_error, conversion_error)


def run_trotter(model: Model) -> tuple[tuple[Result, ...], tuple[Result, ...]]:
    products = []
    controls = []
    for order, reps in ((1, 6), (2, 4)):
        native = repeated_product(build_step(model.observable, 1 / reps, order), reps)
        factors = trotter_reference(
            model.terms, (1 / reps,) * reps, symmetric=order == 2
        )
        expected = reference_circuit(WIDTH, model.terms, factors).state()
        label = f"TS{order} ({reps} reps)"
        products.append(checked(label, native, expected))
        controls.append(
            checked(
                f"Controlled {label}",
                controlled_product(model.terms, factors),
                expected,
                controlled=True,
            )
        )

    times = fourth_order_times()
    positive = build_step(model.observable, times[0], 2)
    negative = build_step(model.observable, times[2], 2)
    fourth = fq.Circuit(WIDTH, dtype=torch.complex128)
    for fragment in (positive, positive, negative, positive, positive):
        fourth.compose(fragment)
    factors = trotter_reference(model.terms, times, symmetric=True)
    expected = reference_circuit(WIDTH, model.terms, factors).state()
    products.append(checked("TS4 (1 rep)", fourth, expected))
    return tuple(products), tuple(controls)


def run_qdrift(model: Model) -> tuple[Result, ...]:
    rows = []
    for samples in (1000, 2000):
        native = build_qdrift(model.observable, samples)
        schedule = reference_sequence(model.terms, 1.0, samples, seed=SEED)
        expected = reference_circuit(WIDTH, model.terms, schedule).state()
        rows.append(checked(f"qDRIFT ({samples} samples)", native, expected))
    return tuple(rows)
