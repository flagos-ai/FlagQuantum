"""Both original discrete-log experiments, exact labels and source illustrations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated

import numpy as np
import torch
from matplotlib.figure import Figure

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.synth import (
    Output,
    QNum,
    adjoint,
    allocate,
    hadamard_transform,
    modular_exp,
    qft,
    qfunc,
)

from ..comparisons.circuit import analytic_state as analytic_mod13
from ..comparisons.circuit import reference_circuit as reference_mod13
from ..main import main as modulo13


@dataclass(frozen=True)
class SourceCase:
    """The two fixed problem sizes and observation counts in the source."""

    modulus: int
    generator: int
    element: int
    phase_bits: int
    target_bits: int
    shots: int
    width_limit: int

    @property
    def order(self) -> int:
        return self.modulus - 1

    @property
    def scale(self) -> int:
        return 1 if self.modulus == 5 else 1 << self.phase_bits

    @property
    def logical_bits(self) -> int:
        return 2 * self.phase_bits + self.target_bits


def source_case(modulus: int) -> SourceCase:
    """Select an original scenario rather than silently changing its dimensions."""
    if type(modulus) is not int:
        raise ValueError("use original modulus 5 or 13")
    if modulus == 5:
        return SourceCase(5, 3, 2, 2, 3, 4000, 13)
    if modulus == 13:
        return SourceCase(13, 7, 3, 5, 4, 10000, 23)
    raise ValueError("use original modulus 5 or 13")


@qfunc
def modulo5(
    element: int,
    generator: int,
    alpha: Output[Annotated[QNum, 2]],
    beta: Output[Annotated[QNum, 2]],
    func_res: Output[Annotated[QNum, 3]],
) -> None:
    allocate(alpha)
    allocate(beta)
    func_res |= 1
    hadamard_transform(alpha)
    hadamard_transform(beta)
    modular_exp(5, element, func_res, alpha)
    modular_exp(5, generator, func_res, beta)
    with adjoint():
        qft(alpha)
        qft(beta)


def _bases(
    case: SourceCase, element: int | None, generator: int | None
) -> tuple[int, int]:
    a = case.element if element is None else element
    g = case.generator if generator is None else generator
    if any(
        type(base) is not int
        or not 1 <= base < case.modulus
        or math.gcd(base, case.modulus) != 1
        for base in (a, g)
    ):
        raise ValueError("bases must be positive invertible residues below the modulus")
    return a, g


def build_program(
    modulus: int, *, element: int | None = None, generator: int | None = None
) -> synth.Program:
    """Reuse native modular powers, inverse QFT and the existing mod-13 program."""
    case = source_case(modulus)
    a, g = _bases(case, element, generator)
    return synth.synthesize(
        modulo5 if modulus == 5 else modulo13,
        classical_inputs={"element": a, "generator": g},
        max_qubits=case.width_limit,
        max_workspace=2,
        memory_limit_bytes=64 * 1024**2,
    )


def reference_circuit(
    modulus: int, *, element: int | None = None, generator: int | None = None
) -> fq.Circuit:
    """Use an independent whole-power permutation and Fourier reference."""
    case = source_case(modulus)
    a, g = _bases(case, element, generator)
    if modulus == 13:
        return reference_mod13(a, g)
    circuit = fq.Circuit(7, dtype=torch.complex128).x(6)
    for bit in range(4):
        circuit.h(bit)
    for offset, base in ((0, a), (2, g)):
        matrix = torch.zeros((32, 32), dtype=torch.complex128)
        for column in range(32):
            power, value = divmod(column, 8)
            target = pow(base, power, 5) * value % 5 if value < 5 else value
            matrix[8 * power + target, column] = 1
        circuit.any(offset, offset + 1, 4, 5, 6, matrix=matrix)
    axis = torch.arange(4, dtype=torch.float64)
    inverse = torch.exp(-2j * math.pi * torch.outer(axis, axis) / 4) / 2
    return circuit.any(0, 1, matrix=inverse).any(2, 3, matrix=inverse)


def analytic_state(
    modulus: int, *, element: int | None = None, generator: int | None = None
) -> torch.Tensor:
    """Evaluate the modular function then independently Fourier-transform both axes."""
    case = source_case(modulus)
    a, g = _bases(case, element, generator)
    if modulus == 13:
        return analytic_mod13(a, g)
    state = torch.zeros((4, 4, 8), dtype=torch.complex128)
    for alpha in range(4):
        for beta in range(4):
            state[alpha, beta, pow(a, alpha, 5) * pow(g, beta, 5) % 5] = 1 / 4
    result = torch.fft.fftn(state, dim=(0, 1), norm="ortho")
    assert isinstance(result, torch.Tensor)
    return result.flatten()


@dataclass(frozen=True)
class Observation:
    """A measured pair of exact numeric labels with its actual shot count."""

    alpha: Fraction
    beta: Fraction
    count: int


@dataclass(frozen=True)
class Candidate:
    """A retained source row after rounding, coprime filtering and inversion."""

    observation: Observation
    alpha_rounded: int
    beta_rounded: int
    logarithm: int


def select_candidates(
    modulus: int, observations: tuple[Observation, ...]
) -> tuple[Candidate, ...]:
    """Retain every source-selected row, including low-count approximate tails."""
    case = source_case(modulus)
    limit = Fraction(1 << case.phase_bits, case.scale)
    for row in observations:
        if (
            type(row.count) is not int
            or row.count < 1
            or any(
                not isinstance(value, Fraction)
                or not 0 <= value < limit
                or (value * case.scale).denominator != 1
                for value in (row.alpha, row.beta)
            )
        ):
            raise ValueError(
                "observations require positive counts and exact in-range register labels"
            )
    candidates = []
    for row in sorted(observations, key=lambda row: (-row.count, row.alpha, row.beta)):
        if modulus == 5:
            a, b = int(row.alpha), int(row.beta)
            if b not in (1, 3):
                continue
            if math.gcd(a, 4) != 1:
                raise ValueError("selected alpha has no inverse modulo four")
            logarithm = b * pow(a, -1, 4)
        else:
            a, b = round(row.alpha * case.order), round(row.beta * case.order)
            if math.gcd(b, case.order) != 1:
                continue
            logarithm = a * pow(b, -1, case.order) % case.order
        candidates.append(Candidate(row, a, b, logarithm))
    return tuple(candidates)


@dataclass(frozen=True)
class Estimate:
    """Source observations and classical recovery with independent state evidence."""

    case: SourceCase
    seed: int
    observations: tuple[Observation, ...]
    candidates: tuple[Candidate, ...]
    maximum_state_error: float
    dirty_workspace_probability: float
    resources: ResourceEstimate

    @property
    def shots(self) -> int:
        return sum(row.count for row in self.observations)

    @property
    def logarithm(self) -> int:
        return self.candidates[0].logarithm


def run_estimate(modulus: int, *, seed: int = 0, shots: int | None = None) -> Estimate:
    """Check full states and same-seed samples, then apply the original assertions."""
    case = source_case(modulus)
    count = case.shots if shots is None else shots
    if type(count) is not int or not 1 <= count <= 100000:
        raise ValueError("use 1..100000 shots")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("use an integer seed in 0..2**32-1")
    program = build_program(modulus)
    result = fq.run(program)
    actual = result.statevector().flatten().reshape(1 << case.logical_bits, -1)
    expected = fq.run(reference_circuit(modulus)).statevector().flatten()
    torch.testing.assert_close(expected, analytic_state(modulus), atol=2e-11, rtol=0)
    torch.testing.assert_close(actual[:, 0], expected, atol=2e-11, rtol=0)
    error = float((actual[:, 0] - expected).abs().max())
    dirty = float(actual[:, 1:].abs().square().sum())
    if dirty > 1e-24:
        raise AssertionError("discrete-log workspace did not return to zero")
    numeric: dict[tuple[Fraction, ...], float]
    if modulus == 5:
        numeric = {
            (Fraction(a), Fraction(b)): probability
            for (a, b), probability in program.probabilities(
                result, registers=("alpha", "beta")
            ).items()
        }
    else:
        numeric = dict(
            program.numeric_probabilities(result, registers=("alpha", "beta"))
        )
    dimension = 1 << case.phase_bits
    dense = torch.zeros(dimension * dimension, dtype=torch.float64)
    for (alpha, beta), probability in numeric.items():
        dense[int(alpha * case.scale) * dimension + int(beta * case.scale)] = (
            probability
        )
    draws = torch.multinomial(
        dense, count, replacement=True, generator=torch.Generator().manual_seed(seed)
    )
    independent = expected.reshape(dimension * dimension, -1).abs().square().sum(dim=-1)
    independent_draws = torch.multinomial(
        independent,
        count,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    if not torch.equal(draws, independent_draws):
        raise AssertionError("same-seed independent Circuit observations disagree")
    counts = torch.bincount(draws, minlength=dimension * dimension)
    observations = tuple(
        sorted(
            (
                Observation(
                    Fraction(index // dimension, case.scale),
                    Fraction(index % dimension, case.scale),
                    int(value),
                )
                for index, value in enumerate(counts.tolist())
                if value
            ),
            key=lambda row: (-row.count, row.alpha, row.beta),
        )
    )
    candidates = select_candidates(modulus, observations)
    if not candidates:
        raise ValueError("no sampled row passed the source inverse filter")
    if modulus == 5:
        if any(row.logarithm != 3 for row in candidates):
            raise AssertionError("mod-five source logarithm assertion failed")
    elif any(
        pow(case.generator, row.logarithm, case.modulus) != case.element
        for row in candidates[:10]
    ):
        raise AssertionError("mod-thirteen source top-ten assertion failed")
    if pow(case.generator, candidates[0].logarithm, case.modulus) != case.element:
        raise AssertionError("recovered logarithm does not reconstruct the element")
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    return Estimate(case, seed, observations, candidates, error, dirty, resources)


@dataclass(frozen=True)
class IllustrativeCurves:
    """The source's real-log-ratio curves, distinct from modular-log recovery."""

    bits: int
    alpha: tuple[float, ...]
    beta: tuple[float, ...]
    real_log_ratio: float


def illustrative_curves(bits: int) -> IllustrativeCurves:
    """Retain the two original analytic illustrations with stable sinc limits."""
    if type(bits) is not int or bits not in (5, 8):
        raise ValueError("use source illustration size 5 or 8")
    size = 1 << bits
    x = np.arange(size, dtype=np.float64)
    ratio = math.log(3) / math.log(7)
    curves = []
    for center in (5 * size / 12, ratio * 5 * size / 12):
        delta = x - center
        values = (np.sinc(delta) / np.sinc(delta / size)) ** 2
        values /= values.sum()
        curves.append(tuple(float(value) for value in values))
    return IllustrativeCurves(bits, curves[0], curves[1], ratio)


def plot_curves(curves: IllustrativeCurves) -> Figure:
    """Plot the supplied illustrative distributions without presenting them as shots."""
    figure = Figure(figsize=(8, 4.5), layout="constrained")
    axis = figure.subplots()
    axis.plot(range(len(curves.alpha)), curves.alpha, label="alpha")
    axis.plot(range(len(curves.beta)), curves.beta, "r-.", label="beta")
    axis.set_xlabel("mu and nu values")
    axis.set_ylabel("Illustrative probability")
    axis.set_title(f"Source real-log-ratio illustration, t={curves.bits}")
    axis.legend()
    return figure
