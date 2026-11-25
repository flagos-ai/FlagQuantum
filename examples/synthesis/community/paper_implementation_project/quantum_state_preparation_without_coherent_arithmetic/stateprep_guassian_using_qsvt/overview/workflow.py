"""Refit the two Gaussian QSVT branches and calibrate exact amplification."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qsvt.overview.even_phases import (
    fit_even,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.phases import (
    CTYPE,
    DTYPE,
    PhaseFit,
    fit,
)
from examples.synthesis.community.paper_implementation_project.quantum_state_preparation_without_coherent_arithmetic.stateprep_guassian_using_qsvt.main import (
    main as amplify_program,
)
from examples.synthesis.community.paper_implementation_project.quantum_state_preparation_without_coherent_arithmetic.stateprep_guassian_using_qsvt.main import (
    prepare,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate, estimate_resources
from flagquantum.compiler.basis_conversion import convert_basis
from flagquantum.synth import Output, QArray, QBit, allocate, qfunc

WIDTH = 8
SIZE = 1 << WIDTH
SCALE = 0.99
EVEN_DEGREE = 28
ODD_DEGREE = 27
ANGLES = tuple(2 ** (bit + 1 - WIDTH) for bit in range(WIDTH))


@dataclass(frozen=True)
class Result:
    even: PhaseFit
    odd: PhaseFit
    fit_error: tuple[float, float]
    continuum_overlap: float
    measured_overlap: float
    prep_error: float
    amplified_error: float
    accepted_mass: float
    work_probability: float
    distribution: torch.Tensor
    expected_distribution: torch.Tensor
    mse: float
    resources: ResourceEstimate
    target_resources: ResourceEstimate


@qfunc
def prepare_gaussian(
    even: list[float],
    odd: list[float],
    angles: list[float],
    data: Output[QArray[QBit]],
    signal: Output[QBit],
    aux: Output[QBit],
    selector: Output[QBit],
) -> None:
    allocate(WIDTH, data)
    allocate(signal)
    allocate(aux)
    allocate(selector)
    prepare(even, odd, angles, data, signal, aux, selector)


def gaussian(x: torch.Tensor) -> torch.Tensor:
    value: torch.Tensor = torch.exp(-x.square())
    return value


def parity_fit(parity: int, degree: int) -> tuple[PhaseFit, float]:
    count = 4096
    theta = math.pi * (torch.arange(count, dtype=DTYPE) + 0.5) / count
    x = torch.cos(theta)
    positive = gaussian(4 * x.asin() - 2)
    negative = gaussian(-4 * x.asin() - 2)
    target = SCALE * (positive + (-1) ** parity * negative)
    orders = torch.arange(parity, degree + 1, 2, dtype=DTYPE)
    basis = torch.cos(theta[:, None] * orders)
    coefficients = 2 / count * (basis.T @ target)
    if parity == 0:
        coefficients[0] /= 2
    error = float((basis @ coefficients - target).abs().max())
    fitted = fit_even(coefficients) if parity == 0 else fit(-coefficients)
    return fitted, error


def projector_angles(
    even: PhaseFit, odd: PhaseFit
) -> tuple[torch.Tensor, torch.Tensor]:
    # Translate Wx phases to the source's real projector response.
    a = even.phases - math.pi / 2
    b = odd.phases - math.pi / 2
    a[0] -= math.pi / 4
    a[-1] += math.pi / 4
    b[0] += math.pi / 4
    b[-1] += math.pi / 4
    return 2 * a, 2 * b


def reference_preparation(even: torch.Tensor, odd: torch.Tensor) -> torch.Tensor:
    """Compute all branches from the controlled sine blocks in basis order."""
    x = torch.arange(SIZE, dtype=DTYPE) / SIZE
    sine, cosine = x.sin(), x.cos()
    signals = (
        torch.stack((sine, cosine, cosine, -sine), dim=-1).reshape(SIZE, 2, 2).to(CTYPE)
    )
    start = torch.zeros((SIZE, 2, 1), dtype=CTYPE)
    start[:, 0, 0] = 1
    signs = torch.tensor([0.5, -0.5], dtype=DTYPE)
    selected = []
    for steps in (even, odd):
        branches = []
        for direction in (1, -1):
            state = start
            for index, phase in enumerate(steps):
                if index:
                    state = signals @ state
                rotation = (
                    (1j * direction * float(phase) * signs).exp().reshape(1, 2, 1)
                )
                state = state * rotation
            branches.append(state[..., 0])
        selected.append(
            torch.stack((branches[0] + branches[1], branches[0] - branches[1]), dim=-1)
        )
    return torch.stack(selected, dim=-1) / (2 * math.sqrt(2 * SIZE))


def reference_amplification(initial: torch.Tensor, beta: float) -> torch.Tensor:
    attenuation = torch.tensor([math.cos(beta / 2), math.sin(beta / 2)], dtype=CTYPE)
    reference = (initial[..., None] * attenuation).reshape(SIZE, 16)
    state = reference.clone()
    for _ in range(2):
        state[:, 0] *= -1
        state = 2 * reference.flatten().vdot(state.flatten()) * reference - state
    return state


def continuum_amplitude() -> float:
    integral = math.sqrt(math.pi / 2) * math.erf(2 * math.sqrt(2))
    return math.sqrt(integral / 8)


def run() -> Result:
    even, even_error = parity_fit(0, EVEN_DEGREE)
    odd, odd_error = parity_fit(1, ODD_DEGREE)
    a, b = projector_angles(even, odd)
    config = dict(
        even=list(map(float, a)),
        odd=list(map(float, b)),
        angles=list(map(float, ANGLES)),
    )
    state_preparation = synth.synthesize(
        prepare_gaussian,
        classical_inputs=config,
        max_qubits=18,
        max_workspace=5,
        memory_limit_bytes=64 * 1024**2,
    )
    initial = fq.run(state_preparation).statevector().reshape(SIZE, 2, 2, 2, -1)
    valid = initial[..., 0]
    ref = reference_preparation(a, b)
    torch.testing.assert_close(valid, ref, atol=1e-10, rtol=0)
    if float(initial[..., 1:].abs().square().sum()) > 1e-22:
        raise ValueError("The preparation must restore all compiler workspace")
    amplitude = float(valid[:, 0, 0, 0].norm())
    beta = 2 * math.acos(math.sin(math.pi / 10) / amplitude)
    bound: dict[str, list[float] | float] = {**config, "beta": beta}
    amplified = synth.synthesize(
        amplify_program,
        classical_inputs=bound,
        max_qubits=18,
        max_workspace=5,
        memory_limit_bytes=64 * 1024**2,
    )
    executed = fq.run(amplified).statevector().reshape(SIZE, 16, 2, -1)
    visible = executed[:, :, 0, 0]
    independent = reference_amplification(ref, beta)
    torch.testing.assert_close(visible, independent, atol=1e-10, rtol=0)
    work = float(
        executed[:, :, 1, :].abs().square().sum()
        + executed[:, :, 0, 1:].abs().square().sum()
    )
    probabilities = visible[:, 0].abs().square()
    target_x = 4 * torch.arange(SIZE, dtype=DTYPE) / SIZE - 2
    expected = gaussian(target_x).square()
    expected /= expected.sum()
    basis = convert_basis(
        amplified.circuit, gates=("cx", "u"), max_added_operations=100_000
    )
    return Result(
        even,
        odd,
        (even_error, odd_error),
        continuum_amplitude(),
        amplitude,
        float((valid - ref).abs().max()),
        float((visible - independent).abs().max()),
        float(probabilities.sum()),
        work,
        probabilities,
        expected,
        float((probabilities - expected).square().mean()),
        estimate_resources(amplified.circuit),
        estimate_resources(basis.program),
    )
