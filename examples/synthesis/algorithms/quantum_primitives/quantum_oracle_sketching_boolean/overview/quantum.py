"""Synthesize commuting Boolean sketches and apply the fresh QSVT correction."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qsvt.overview.even_phases import (
    fit_even,
)
from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qsvt.overview.workflow import (
    selected_response,
    selection_weights,
    sequential,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.phases import (
    CTYPE,
    DTYPE,
    PhaseFit,
    response,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.workflow import (
    shifted_phases,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate, estimate_resources
from flagquantum.synth import (
    CX,
    H,
    Output,
    QNum,
    allocate,
    apply,
    control,
    hadamard_transform,
    phase,
    qfunc,
    within,
)

SHOTS = 10_000
P_MIN = 0.10
P_MAX = 0.45
THRESHOLD = math.sin(P_MIN / P_MAX)
EPS_QSVT = 0.02
SCALE = 0.95
DEGREE = 2 * math.ceil(math.log(1 / EPS_QSVT) / (2 * THRESHOLD))


@qfunc
def apply_phase(angle: float, word: int, data: QNum) -> None:
    control(data == word, lambda: phase(1, angle))


@qfunc
def diagonal(angles: list[float], data: QNum) -> None:
    for i in range(len(angles)):
        apply_phase(angles[i], i, data)


@qfunc
def uniform_sketch(width: int, angles: list[float], data: Output[QNum]) -> None:
    allocate(width, data)
    hadamard_transform(data)
    diagonal(angles, data)


@qfunc
def bell_sketch(angles: list[float], data: Output[QNum]) -> None:
    allocate(2, data)
    with within():
        H(data[0])
        CX(data[0], data[1])
    with apply():
        diagonal(angles, data)


def sketch(angles: tuple[float, ...], bell: bool = False) -> synth.Program:
    if len(angles) not in {4, 8} or bell and len(angles) != 4:
        raise ValueError("Use the original two- or three-bit sketch size")
    if any(not math.isfinite(angle) for angle in angles):
        raise ValueError("The accumulated phases must be finite")
    width = len(angles).bit_length() - 1
    key: dict[str, int | list[float]] = {"angles": list(angles)}
    if not bell:
        key["width"] = width
    return synth.synthesize(
        bell_sketch if bell else uniform_sketch,
        classical_inputs=key,
        max_qubits=width + 2,
        max_workspace=2,
        memory_limit_bytes=4 * 1024**2,
    )


def target_response(x: torch.Tensor) -> torch.Tensor:
    value: torch.Tensor = -SCALE * torch.tanh(4 * (x.abs() - THRESHOLD / 2) / THRESHOLD)
    return value


def bounded_minimax(design: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fit a small bounded Chebyshev polynomial with a feasible Newton barrier."""
    count = design.size(1)
    ones = torch.ones((design.size(0), 1), dtype=DTYPE)
    zeros = torch.zeros_like(ones)
    constraints = torch.cat(
        (
            torch.cat((design, -ones), dim=1),
            torch.cat((-design, -ones), dim=1),
            torch.cat((design, zeros), dim=1),
            torch.cat((-design, zeros), dim=1),
        )
    )
    upper = torch.cat((target, -target, 0.99 * ones[:, 0], 0.99 * ones[:, 0]))
    guess = torch.zeros(count + 1, dtype=DTYPE)
    guess[-1] = 1.25
    objective = torch.zeros_like(guess)
    objective[-1] = 1
    for exponent in range(1, 10):
        barrier = 10.0**-exponent
        for _ in range(48):
            slack = upper - constraints @ guess
            normalized = constraints / slack[:, None]
            gradient = objective + barrier * normalized.sum(dim=0)
            hessian = barrier * (normalized.T @ normalized)
            step = torch.linalg.solve(hessian, -gradient)
            slope = float(gradient @ step)
            if -slope < 1e-13:
                break
            base = float(guess[-1] - barrier * slack.log().sum())
            scale = 1.0
            for _ in range(64):
                candidate = guess + scale * step
                remaining = upper - constraints @ candidate
                if bool((remaining > 0).all()):
                    value = float(candidate[-1] - barrier * remaining.log().sum())
                    if value <= base + 0.01 * scale * slope:
                        guess = candidate
                        break
                scale *= 0.5
            else:
                raise ValueError(
                    "The bounded threshold fit failed to find a feasible step"
                )
    return guess[:-1]


def threshold_fit() -> tuple[PhaseFit, float]:
    theta = torch.linspace(0, math.pi / 2, 2049, dtype=DTYPE)
    target = target_response(theta.cos())
    design = torch.cos(theta[:, None] * torch.arange(0, DEGREE + 1, 2, dtype=DTYPE))
    coefficients = bounded_minimax(design, target)
    fitted = fit_even(coefficients)
    grid = torch.linspace(0, 1, 4097, dtype=DTYPE)
    values = response(grid, fitted.phases)
    error = float((values - target_response(grid)).abs().max())
    if float(values.abs().max()) >= 1 or error > 0.19:
        raise ValueError(
            "The degree-18 threshold response must stay in its checked error range"
        )
    return fitted, error


def hadamard_block(angles: tuple[float, ...]) -> torch.Tensor:
    """Execute the source Hadamard-test sequence on its entire input basis."""
    phase_operator = torch.diag(torch.exp(1j * torch.tensor(angles, dtype=DTYPE)))
    controlled = torch.eye(8, dtype=CTYPE)
    controlled[4:, 4:] = phase_operator
    inputs = torch.eye(8, dtype=CTYPE)
    circuit = fq.Circuit(3, dtype=CTYPE, bsz=8, inputs=inputs)
    circuit.h(0).any(0, 1, 2, matrix=controlled).x(0)
    circuit.any(0, 1, 2, matrix=controlled.mH).x(0).h(0).s(0).x(0)
    actual: torch.Tensor = circuit.state().reshape(8, 8).T.contiguous()
    expect = torch.diag(torch.tensor(angles, dtype=DTYPE).sin()).to(CTYPE)
    torch.testing.assert_close(actual[:4, :4], expect, atol=1e-12, rtol=0)
    return actual


@dataclass(frozen=True)
class Corrected:
    success: float
    conditional_fidelity: float
    spectral_error: float
    oracle_error: float
    full_state_error: float
    resources: ResourceEstimate
    sampled_success: float
    sampled_fidelity: float


def sampled(probabilities: torch.Tensor, seed: int) -> torch.Tensor:
    draws = torch.multinomial(
        probabilities,
        SHOTS,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return torch.bincount(draws, minlength=probabilities.numel())


def corrected_reference(source: torch.Tensor, fit: PhaseFit) -> torch.Tensor:
    hadamard = torch.tensor([[1, 1], [1, -1]], dtype=CTYPE) / math.sqrt(2)
    permutation = torch.eye(4, dtype=CTYPE)[[0, 3, 2, 1]]
    bell = permutation @ torch.kron(torch.eye(2, dtype=CTYPE), hadamard)
    vector = torch.zeros(8, dtype=CTYPE)
    vector[:4] = bell[:, 0]
    signs = torch.tensor([1] * 4 + [-1] * 4, dtype=DTYPE)
    phases = shifted_phases(fit)
    a, b = selection_weights(phases.numel() - 1)
    plus = a * sequential(source, signs, phases, vector)
    minus = b * sequential(source, signs, -phases, vector)
    combined = torch.cat((plus + minus, plus - minus)).reshape(4, 4) / 2
    output: torch.Tensor = combined @ bell.conj()
    return output.flatten()


def corrected(angles: tuple[float, ...], fit: PhaseFit) -> Corrected:
    source = hadamard_block(angles)
    filtered = selected_response(source, fit)
    circuit = fq.Circuit(4, dtype=CTYPE).h(3).cx(3, 2)
    circuit.any(*range(4), matrix=filtered).cx(3, 2).h(3)
    actual = fq.run(circuit).statevector().flatten()
    full_reference = corrected_reference(source, fit)
    torch.testing.assert_close(actual, full_reference, atol=1e-12, rtol=0)
    signed = response(torch.tensor(angles, dtype=DTYPE).sin(), fit.phases)
    basis = torch.eye(16, dtype=CTYPE)[:4]
    compiled = fq.Circuit(4, dtype=CTYPE, bsz=4, inputs=basis)
    columns = compiled.any(*range(4), matrix=filtered).state()[:, :4]
    expected_oracle = torch.diag(signed).to(CTYPE)
    torch.testing.assert_close(columns, expected_oracle, atol=1e-12, rtol=0)
    p, q = signed[0], signed[-1]
    expected = torch.tensor([(p + q) / 2, (p - q) / 2, 0, 0], dtype=CTYPE)
    torch.testing.assert_close(actual[:4], expected, atol=1e-12, rtol=0)
    frequencies = actual.abs().square()
    success = float(frequencies[:4].sum())
    count = sampled(frequencies, 246)
    block_draws = int(count[:4].sum())
    return Corrected(
        success,
        float(frequencies[1]) / success,
        float((actual[:4] - expected).abs().max()),
        float((columns - expected_oracle).abs().max()),
        float((actual - full_reference).abs().max()),
        estimate_resources(circuit),
        block_draws / SHOTS,
        int(count[1]) / block_draws,
    )
