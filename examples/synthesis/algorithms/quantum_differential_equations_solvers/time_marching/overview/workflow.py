"""Run both native integrators and retain their finite postselection outcomes."""

from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.phases import (
    response,
)
from flagquantum.compiler import ResourceEstimate, estimate_resources

from .approximation import (
    REAL,
    Amplifier,
    Exponential,
    amplitude_polynomial,
    exponential,
    physical_phases,
)
from .quantum import END, SLICE_COUNT, assemble, build_slice, exact_ode, expected_block
from .reference import reference_state

SHOTS = 1_000_000


@dataclass(frozen=True)
class March:
    resources: ResourceEstimate
    state_error: float
    exact_prob: float
    accepted_counts: tuple[int, ...]
    sampled_magnitudes: torch.Tensor
    normalized_error: float


@dataclass(frozen=True)
class Result:
    times: torch.Tensor
    classical: torch.Tensor
    demo: Exponential
    exponential: Exponential
    amplifier: Amplifier
    scalar_fit_error: float
    magnus_block_error: float
    naive: March
    amplified: March


def march(
    circuit: fq.Circuit,
    expected: torch.Tensor,
    normalized_classical: torch.Tensor,
    *,
    seed: int,
) -> March:
    resources = estimate_resources(circuit)
    if resources.n_qubits != 11 or resources.n_operations > 200_000:
        raise ValueError("The original-sized integrator exceeded its resource budget")
    state: torch.Tensor = circuit.state().flatten()
    error = float((state - expected).abs().max())
    if error > 1e-10:
        raise ValueError("The complete state disagrees with the independent integrator")
    joint = state.reshape(4, 64, 8).abs().square().sum(dim=1)
    success = float(joint[:, 0].sum())
    generator = torch.Generator().manual_seed(seed)
    observed = torch.multinomial(
        joint.flatten(), SHOTS, replacement=True, generator=generator
    )
    counts = torch.bincount(observed, minlength=32).reshape(4, 8)
    accepted = counts[:, 0]
    amplitudes = (accepted.to(REAL) / SHOTS).sqrt()
    if int(accepted.sum()) == 0:
        raise ValueError("The finite run did not observe a completed path")
    norm = amplitudes.norm()
    difference = float((amplitudes / norm - normalized_classical).norm())
    return March(
        resources,
        error,
        success,
        tuple(int(v) for v in accepted),
        amplitudes,
        difference,
    )


def run() -> Result:
    times = torch.linspace(0, END, 100, dtype=REAL)
    classical = exact_ode(times)
    demo = exponential(2)
    approximation = exponential(END / SLICE_COUNT)
    amplifier = amplitude_polynomial()
    even = physical_phases(approximation.even_fit)
    odd = physical_phases(approximation.odd_fit)
    amp = physical_phases(amplifier.phase_fit)
    slices = tuple(
        build_slice(i / 2, (i + 1) / 2, even, odd) for i in range(SLICE_COUNT)
    )
    maximum_block_error = 0.0
    for index, segment in enumerate(slices):
        circuit = fq.Circuit(7, dtype=torch.complex128).h(0).h(1).compose(segment)
        projected = 2 * circuit.state().reshape(4, 32)[:, 0]
        target = expected_block(index / 2, (index + 1) / 2)
        expected = 0.5 * (
            response(target, approximation.even_fit.phases)
            + response(target, approximation.odd_fit.phases)
        )
        maximum_block_error = max(
            maximum_block_error, float((projected - expected).abs().max())
        )
    if maximum_block_error > 1e-10:
        raise ValueError(
            "The Magnus projector blocks must realize the fitted polynomial"
        )
    dense = torch.linspace(-1, 1, 4097, dtype=REAL)
    fitted = response(dense, demo.even_fit.phases) + response(
        dense, demo.odd_fit.phases
    )
    scalar_fit_error = float((fitted - (2 * (dense - 1)).exp()).abs().max() / 2)

    normalized = classical[-1] / classical[-1].norm()
    naive = march(
        assemble(slices, None), reference_state(even, odd, None), normalized, seed=1725
    )
    amplified = march(
        assemble(slices, amp), reference_state(even, odd, amp), normalized, seed=1726
    )
    if amplified.normalized_error >= 0.1 or amplified.exact_prob <= naive.exact_prob:
        raise ValueError(
            "Amplified time marching does not meet the source accuracy outcome"
        )
    return Result(
        times,
        classical,
        demo,
        approximation,
        amplifier,
        scalar_fit_error,
        maximum_block_error,
        naive,
        amplified,
    )
