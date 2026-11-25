"""Native single-shot Pauli and finite-depth Clifford shadow tomography."""

import math
from dataclasses import dataclass

import numpy as np
import torch
from matplotlib.figure import Figure

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.synth import (
    CX,
    RZ,
    H,
    Output,
    QArray,
    QBit,
    R,
    S,
    prepare_bell_state,
    qfunc,
)


@dataclass(frozen=True)
class Gate:
    """One bounded random H, S or CX gate on the two logical data bits."""

    kind: int
    first: int
    second: int


@dataclass(frozen=True)
class Shadow:
    """Actual single-shot rows with their exact basis or Clifford provenance."""

    bits: torch.Tensor
    bases: torch.Tensor | None
    sequences: tuple[tuple[Gate, ...], ...]
    density: torch.Tensor
    zz_estimate: float
    frobenius_error: float
    resources: tuple[ResourceEstimate, ...]


@dataclass(frozen=True)
class Workflow:
    """Source-size Pauli and finite-depth global Clifford shadow estimates."""

    pauli: Shadow
    clifford: Shadow
    pauli_bound: tuple[int, int]
    clifford_bound: tuple[int, int]


@qfunc
def pauli_measurement(
    thetas: list[float], phis: list[float], q: Output[QArray[QBit]]
) -> None:
    prepare_bell_state(1, q)
    for i in range(2):
        RZ(phis[i], q[i])
        R(thetas[i], math.pi / 2, q[i])


@qfunc
def clifford_measurement(
    kinds: list[int], first: list[int], second: list[int], q: Output[QArray[QBit]]
) -> None:
    prepare_bell_state(1, q)
    for i in range(20):
        if kinds[i] == 0:
            H(q[first[i]])
        elif kinds[i] == 1:
            S(q[first[i]])
        else:
            CX(q[first[i]], q[second[i]])


def _matrices() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    identity = torch.eye(2, dtype=torch.complex128)
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    s = torch.diag(torch.tensor([1, 1j], dtype=torch.complex128))
    return identity, h, s


def _bits(program: synth.Program, generator: torch.Generator) -> tuple[int, int]:
    values = dict(program.probabilities(fq.run(program), registers=("q",)))
    distribution = torch.tensor(
        [values.get((i,), 0.0) for i in range(4)], dtype=torch.float64
    )
    word = int(torch.multinomial(distribution, 1, generator=generator))
    return word & 1, word >> 1


def _checked_rows(bits: torch.Tensor) -> None:
    if bits.ndim != 2 or bits.shape[1] != 2 or bits.shape[0] == 0:
        raise ValueError("use a nonempty two-bit snapshot matrix")
    if bits.dtype != torch.int64 or not bool(((bits == 0) | (bits == 1)).all()):
        raise ValueError("snapshot entries must be int64 binary values")


def pauli_estimators(bits: torch.Tensor, bases: torch.Tensor) -> torch.Tensor:
    """Reconstruct each snapshot with the inverse local Pauli channel."""
    _checked_rows(bits)
    if bases.shape != bits.shape or bases.dtype != torch.int64:
        raise ValueError("use one int64 Pauli basis id for each snapshot bit")
    if not bool(((bases >= 0) & (bases <= 2)).all()):
        raise ValueError("Pauli basis ids must be 0, 1 or 2")
    identity, h, s = _matrices()
    unitaries = (h, h @ s, identity)
    rows = []
    for pair, ids in zip(bits.tolist(), bases.tolist(), strict=True):
        factors = []
        for bit, basis in zip(pair, ids, strict=True):
            state = torch.diag(torch.tensor([1 - bit, bit], dtype=torch.complex128))
            u = unitaries[basis]
            factors.append(3 * u.mH @ state @ u - identity)
        rows.append(torch.kron(*factors))
    return torch.stack(rows)


def _checked_sequence(sequence: tuple[Gate, ...]) -> None:
    if len(sequence) != 20:
        raise ValueError("use the original depth-20 Clifford sequence")
    for gate in sequence:
        if any(
            type(value) is not int for value in (gate.kind, gate.first, gate.second)
        ):
            raise ValueError("Clifford gate kinds and indices must be Python integers")
        if gate.kind not in (0, 1, 2) or gate.first not in (0, 1):
            raise ValueError("use H, S or CX on the two logical bits")
        if gate.kind < 2 and gate.second != -1:
            raise ValueError("single-qubit gates require second=-1")
        if gate.kind == 2 and (gate.second not in (0, 1) or gate.second == gate.first):
            raise ValueError("CX requires two distinct logical bits")


def build_pauli_program(bases: tuple[int, int]) -> synth.Program:
    """Bind the two Pauli basis choices to a two-bit native quantum program."""
    if len(bases) != 2 or any(type(b) is not int or b not in (0, 1, 2) for b in bases):
        raise ValueError("use exactly two integer Pauli basis ids in 0..2")
    angles = ((math.pi / 2, math.pi), (math.pi / 2, 3 * math.pi / 2), (0.0, 0.0))
    return synth.synthesize(
        pauli_measurement,
        classical_inputs={
            "thetas": [angles[b][0] for b in bases],
            "phis": [angles[b][1] for b in bases],
        },
        max_qubits=2,
        max_workspace=0,
        memory_limit_bytes=1024**2,
    )


def build_clifford_program(sequence: tuple[Gate, ...]) -> synth.Program:
    """Bind one checked depth-20 sequence to a two-bit native quantum program."""
    _checked_sequence(sequence)
    return synth.synthesize(
        clifford_measurement,
        classical_inputs={
            "kinds": [g.kind for g in sequence],
            "first": [g.first for g in sequence],
            "second": [g.second for g in sequence],
        },
        max_qubits=2,
        max_workspace=0,
        memory_limit_bytes=1024**2,
    )


def sequence_unitary(sequence: tuple[Gate, ...]) -> torch.Tensor:
    """Use logical q[0] as the left tensor factor for matrix reconstruction."""
    _checked_sequence(sequence)
    identity, h, s = _matrices()
    unitary = torch.eye(4, dtype=torch.complex128)
    for gate in sequence:
        if gate.kind < 2:
            g = h if gate.kind == 0 else s
            embedded = torch.kron(
                g if gate.first == 0 else identity, g if gate.first == 1 else identity
            )
        else:
            embedded = torch.zeros((4, 4), dtype=torch.complex128)
            for word in range(4):
                output = word ^ (
                    (1 << (1 - gate.second)) if word & (1 << (1 - gate.first)) else 0
                )
                embedded[output, word] = 1
        unitary = embedded @ unitary
    return unitary


def clifford_estimators(
    bits: torch.Tensor, sequences: tuple[tuple[Gate, ...], ...]
) -> torch.Tensor:
    """Reconstruct global Clifford snapshots without clipping negative eigenvalues."""
    _checked_rows(bits)
    if len(sequences) != len(bits):
        raise ValueError("each snapshot needs its own recorded Clifford sequence")
    rows = []
    for pair, sequence in zip(bits.tolist(), sequences, strict=True):
        row = sequence_unitary(sequence)[2 * pair[0] + pair[1]]
        rows.append(
            5 * row.conj().unsqueeze(1) * row.unsqueeze(0)
            - torch.eye(4, dtype=torch.complex128)
        )
    return torch.stack(rows)


def median_of_means(estimators: torch.Tensor, divisions: int) -> float:
    """Estimate ZZ with equal, nonempty blocks and the conventional even median."""
    if estimators.ndim != 3 or estimators.shape[1:] != (4, 4):
        raise ValueError("use four-by-four snapshot estimators")
    if estimators.dtype != torch.complex128 or not bool(
        torch.isfinite(estimators).all()
    ):
        raise ValueError("use finite complex128 snapshot estimators")
    if not torch.allclose(estimators, estimators.mH, atol=1e-12, rtol=0):
        raise ValueError("snapshot estimators must be Hermitian")
    if (
        isinstance(divisions, bool)
        or not isinstance(divisions, int)
        or not 1 <= divisions <= len(estimators)
        or len(estimators) % divisions
    ):
        raise ValueError("divisions must evenly partition the nonempty snapshots")
    zz = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.complex128))
    values = torch.einsum("ij,nji->n", zz, estimators).real
    means = values.reshape(divisions, -1).mean(1)
    return float(torch.quantile(means, 0.5))


def sample_bound(
    error: float, delta: float, shadow_norm_squared: float
) -> tuple[int, int]:
    """Compute the source's illustrative single-ZZ-observable sample bound."""
    if (
        any(isinstance(v, bool) for v in (error, delta, shadow_norm_squared))
        or not all(math.isfinite(v) for v in (error, delta, shadow_norm_squared))
        or error <= 0
        or not 0 < delta < 1
        or shadow_norm_squared <= 0
    ):
        raise ValueError("use finite positive error/norm and 0 < delta < 1")
    divisions = 2 * math.log(2 / delta)
    return math.ceil(34 * shadow_norm_squared / error**2 * divisions), int(divisions)


def _summary(
    bits: torch.Tensor,
    bases: torch.Tensor | None,
    sequences: tuple[tuple[Gate, ...], ...],
    estimators: torch.Tensor,
    divisions: int,
    resources: tuple[ResourceEstimate, ...],
) -> Shadow:
    bell = torch.tensor([1, 0, 0, -1], dtype=torch.complex128) / math.sqrt(2)
    density = estimators.mean(0)
    error = float(torch.linalg.matrix_norm(density - torch.outer(bell, bell.conj())))
    return Shadow(
        bits,
        bases,
        sequences,
        density,
        median_of_means(estimators, divisions),
        error,
        resources,
    )


def collect_pauli(rng: np.random.RandomState) -> Shadow:
    """Collect the source's 400 random Pauli-basis snapshots, one shot each."""
    bases = torch.tensor(rng.randint(3, size=(400, 2)), dtype=torch.int64)
    generator = torch.Generator().manual_seed(int(rng.randint(1000000)))
    rows = []
    resources = []
    for pair in bases.tolist():
        program = build_pauli_program((pair[0], pair[1]))
        estimate = program.report["resources"]
        assert isinstance(estimate, ResourceEstimate)
        resources.append(estimate)
        rows.append(_bits(program, generator))
    bits = torch.tensor(rows, dtype=torch.int64)
    return _summary(bits, bases, (), pauli_estimators(bits, bases), 2, tuple(resources))


def collect_clifford(rng: np.random.RandomState) -> Shadow:
    """Collect 50 depth-20 random H/S/CX snapshots, one shot each."""
    sequences = []
    rows = []
    resources = []
    for _ in range(50):
        gates = []
        for _ in range(20):
            kind = int(rng.randint(3))
            if kind < 2:
                first, second = int(rng.randint(2)), -1
            else:
                first, second = map(int, rng.choice(2, size=2, replace=False))
            gates.append(Gate(kind, first, second))
        sequence = tuple(gates)
        program = build_clifford_program(sequence)
        estimate = program.report["resources"]
        assert isinstance(estimate, ResourceEstimate)
        resources.append(estimate)
        rows.append(
            _bits(program, torch.Generator().manual_seed(int(rng.randint(1000000))))
        )
        sequences.append(sequence)
    bits = torch.tensor(rows, dtype=torch.int64)
    recorded = tuple(sequences)
    return _summary(
        bits, None, recorded, clifford_estimators(bits, recorded), 10, tuple(resources)
    )


def run_workflow() -> Workflow:
    """Collect both source-size shadows locally with seed 552.

    Examples:
        >>> report = run_workflow()
        >>> len(report.pauli.bits), len(report.clifford.bits)
        (400, 50)

    A finite-depth Clifford random walk is an approximate ensemble. The ideal
    Clifford sample bound is illustrative and does not certify this walk or
    the error observed from only 50 snapshots.
    """
    rng = np.random.RandomState(552)
    pauli_shadow = collect_pauli(rng)
    clifford_shadow = collect_clifford(rng)
    return Workflow(
        pauli_shadow,
        clifford_shadow,
        sample_bound(0.2, 0.01, 9),
        sample_bound(0.2, 0.01, 12),
    )


def plot_workflow(workflow: Workflow) -> Figure:
    """Compare both measured reconstruction matrices and ZZ estimates."""
    figure = Figure(figsize=(12, 3.8), constrained_layout=True)
    axes = figure.subplots(1, 3)
    limit = max(
        float(workflow.pauli.density.real.abs().max()),
        float(workflow.clifford.density.real.abs().max()),
    )
    for axis, shadow, title in zip(
        axes[:2],
        (workflow.pauli, workflow.clifford),
        ("Pauli: 400 single shots", "Clifford: 50 single shots"),
        strict=True,
    ):
        image = axis.imshow(
            shadow.density.real.numpy(), vmin=-limit, vmax=limit, cmap="RdBu_r"
        )
        axis.set(
            title=f"{title}\nFrobenius error {shadow.frobenius_error:.3f}",
            xlabel="Logical basis column",
            ylabel="Logical basis row",
            xticks=range(4),
            yticks=range(4),
        )
        figure.colorbar(image, ax=axis, label="Real matrix entry")
    axes[2].bar(
        ("Pauli", "Clifford"),
        (workflow.pauli.zz_estimate, workflow.clifford.zz_estimate),
    )
    axes[2].axhline(1, color="red", linestyle="--", label="Exact Bell ZZ = 1")
    axes[2].set(title="Measured median of means", ylabel="ZZ estimate")
    axes[2].legend()
    return figure
