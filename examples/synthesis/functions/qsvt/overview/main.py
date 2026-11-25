"""Fit and execute the source's odd degree-five square-root QSVT example."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

import flagquantum as fq
from examples.synthesis.functions.qsvt.main import main
from flagquantum import synth
from flagquantum.algorithms import NelderMeadOptimizer

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class PhaseFit:
    """One bounded local fit, with independently sampled validation error."""

    phases: tuple[float, ...]
    seed: int
    evaluations: int
    max_error: float


@dataclass(frozen=True)
class Workflow:
    """Actual native execution and the source's sixteen projected values."""

    fit: PhaseFit
    program: synth.Program
    points: torch.Tensor
    measured: torch.Tensor
    target: torch.Tensor


def source_coefficients() -> torch.Tensor:
    """Retain degree five, seed one, odd parity and 500-point normalization."""
    coefficients = np.random.RandomState(1).uniform(-1, 1, 6)
    coefficients[0::2] = 0
    polynomial = np.polynomial.Chebyshev(coefficients)
    scale = np.max(np.abs(polynomial(np.linspace(-1, 1, 500)))) + 0.001
    return torch.tensor(coefficients / scale, dtype=torch.float64)


def polynomial(coefficients: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Evaluate six Chebyshev coefficients without invoking the phase model."""
    if coefficients.shape != (6,) or not bool(torch.isfinite(coefficients).all()):
        raise ValueError("Use six finite Chebyshev coefficients.")
    previous, current = torch.ones_like(values), values
    result = coefficients[0] * previous + coefficients[1] * current
    for index in range(2, 6):
        previous, current = current, 2 * values * current - previous
        result = result + coefficients[index] * current
    return result


def response(phases: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Scalar QSP response using native projector angles, twice the QSP phase."""
    if (
        phases.shape != (6,)
        or values.ndim != 1
        or not 1 <= values.numel() <= 65536
        or phases.dtype != torch.float64
        or values.dtype != torch.float64
        or phases.device.type != "cpu"
        or values.device.type != "cpu"
        or not bool(torch.isfinite(phases).all())
        or not bool(torch.isfinite(values).all())
        or bool((values.abs() > 1).any())
    ):
        raise ValueError(
            "Use six finite CPU float64 phases and 1..65536 real points in [-1,1]."
        )
    complement = (1 - values.square()).clamp_min(0).sqrt()
    walk = torch.stack((values, complement, complement, -values), dim=-1)
    walk = walk.reshape(-1, 2, 2).to(torch.complex128)
    result = torch.eye(2, dtype=torch.complex128).expand(len(values), 2, 2)
    signs = torch.tensor([1.0, -1.0], dtype=torch.float64)
    for index in range(6):
        result = torch.diag(torch.exp(0.5j * phases[index] * signs)) @ result
        if index < 5:
            result = walk @ result
    return result[:, 0, 0].real


def fit_phases(
    coefficients: torch.Tensor, *, max_iterations: int = 2500, starts: int = 4
) -> PhaseFit:
    """Fit this concrete six-angle profile; refuse an inaccurate local result."""
    if (
        type(max_iterations) is not int
        or not 1 <= max_iterations <= 2500
        or type(starts) is not int
        or not 1 <= starts <= 4
    ):
        raise ValueError("Use 1..2500 iterations and 1..4 seeded starts.")
    if (
        coefficients.shape != (6,)
        or coefficients.dtype != torch.float64
        or coefficients.device.type != "cpu"
        or not bool(torch.isfinite(coefficients).all())
        or bool((coefficients[0::2] != 0).any())
    ):
        raise ValueError("Use six finite CPU float64 odd-parity coefficients.")
    fit_grid = torch.cos(torch.pi * (torch.arange(64, dtype=torch.float64) + 0.5) / 64)
    validation_grid = torch.linspace(-1, 1, 4097, dtype=torch.float64)
    target = polynomial(coefficients, fit_grid)
    expected = polynomial(coefficients, validation_grid)
    if float(expected.abs().max()) >= 1:
        raise ValueError("This local fit requires a polynomial below one in magnitude.")

    def objective(parameters: torch.Tensor) -> torch.Tensor:
        return (response(parameters, fit_grid) - target).square().mean()

    for seed in range(starts):
        generator = torch.Generator().manual_seed(seed)
        initial = torch.rand(6, generator=generator, dtype=torch.float64) * 2 * torch.pi
        optimizer = NelderMeadOptimizer(
            maxiter=max_iterations, tolerance=1e-10, atol=1e-16
        )
        result = optimizer.minimize(objective, initial)
        error = float(
            (response(result.parameters, validation_grid) - expected).abs().max()
        )
        if error < 1e-7:
            return PhaseFit(
                tuple(float(p) for p in result.parameters),
                seed,
                result.evaluations,
                error,
            )
    raise ValueError(
        "Bounded phase fit failed numerical validation; no fixed-phase fallback."
    )


def build_program(phases: tuple[float, ...]) -> synth.Program:
    """Bind newly fitted angles to the existing native quantum composition."""
    if len(phases) != 6 or any(type(value) is not float for value in phases):
        raise ValueError("Use six finite fitted floating-point angles.")
    if not bool(torch.isfinite(torch.tensor(phases, dtype=torch.float64)).all()):
        raise ValueError("Use six finite fitted floating-point angles.")
    return synth.synthesize(
        main,
        classical_inputs={"phases": list(phases)},
        max_qubits=16,
        max_workspace=6,
        memory_limit_bytes=8 * 1024**2,
    )


def projected_values(program: synth.Program) -> torch.Tensor:
    """Read the odd-degree source projection through named logical outputs."""
    probabilities = program.probabilities(
        fq.run(program), registers=("state", "ref", "ind", "qsvt_aux")
    )
    selected = torch.zeros(16, dtype=torch.float64)
    for (state, _, indicator, auxiliary), probability in probabilities.items():
        if indicator == 0 and auxiliary == 0:
            selected[state] += probability
    return (16 * selected).sqrt()


def run_workflow() -> Workflow:
    coefficients = source_coefficients()
    fit = fit_phases(coefficients)
    program = build_program(fit.phases)
    points = (torch.arange(16, dtype=torch.float64) / 16).sqrt()
    measured = projected_values(program)
    target = polynomial(coefficients, points).abs()
    torch.testing.assert_close(measured, target, atol=1e-7, rtol=0)
    print("Chebyshev coefficients:", coefficients.tolist())
    print("Fitted native projector angles:", fit.phases)
    print("Fit:", fit)
    print("Measured:", measured.tolist())
    print("Target:", target.tolist())
    print("Maximum execution error:", float((measured - target).abs().max()))
    print("Resources:", program.report["resources"])
    return Workflow(fit, program, points, measured, target)


def plot_results(workflow: Workflow) -> "Figure":
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    axes.scatter(workflow.points, workflow.measured, label="measured", c="g")
    axes.plot(workflow.points, workflow.target, label="target")
    axes.set_xlabel(r"$\sqrt{x}$")
    axes.set_ylabel(r"$P(\sqrt{x})$")
    axes.legend()
    plt.show()
    return figure


def run_example() -> None:
    plot_results(run_workflow())


if __name__ == "__main__":
    run_example()
