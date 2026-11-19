"""Native quantum VQE and explicit parameter-shift NOT training."""

import math
from dataclasses import dataclass

import torch
from matplotlib.figure import Figure

import flagquantum as fq
from examples.synthesis.tutorials.basics.qml.main import main as classifier
from flagquantum import synth
from flagquantum.algorithms import NelderMeadOptimizer, NelderMeadResult
from flagquantum.synth import Output, QBit, U, allocate, qfunc


@qfunc
def ansatz(angles: list[float], q: Output[QBit]) -> None:
    allocate(q)
    U(angles[0], angles[1], angles[2], 0, q)


@dataclass(frozen=True)
class Variational:
    """Actual energy evaluations and the bounded optimizer's final program."""

    optimizer: NelderMeadResult
    trace: tuple[float, ...]
    program: synth.Program


@dataclass(frozen=True)
class Step:
    """One original full-batch MAE/SGD update using quantum shift derivatives."""

    weight_before: float
    loss: float
    gradient: float
    weight_after: float


@dataclass(frozen=True)
class Training:
    """Five-step NOT training observations, final predictions and native program."""

    steps: tuple[Step, ...]
    weight: float
    predictions: tuple[float, float]
    accuracy: float
    program: synth.Program


@dataclass(frozen=True)
class Workflow:
    """Both original QML guide sections with independent resource reports."""

    variational: Variational
    training: Training


def _finite(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite real value")


def build_variational(angles: tuple[float, ...]) -> synth.Program:
    """Specialize the original three-angle U ansatz within local resource limits."""
    if len(angles) != 3:
        raise ValueError("angles must contain exactly three values")
    for value in angles:
        _finite(value, "angle")
    return synth.synthesize(
        ansatz,
        classical_inputs={"angles": list(angles)},
        max_qubits=4,
        max_operations=100,
        memory_limit_bytes=1024**2,
    )


def energy(angles: tuple[float, ...]) -> float:
    """Evaluate 0.5 I + 0.5 Z - X from an actual native quantum state."""
    program = build_variational(angles)
    result = fq.run(program)
    logical = program.registers["q"]
    hidden = tuple(i for i in range(program.circuit.n_wires) if i not in logical)
    state = (
        result.statevector()
        .reshape((2,) * program.circuit.n_wires)
        .permute((*logical, *hidden))
        .reshape(2, -1)
    )
    density = state @ state.conj().T
    return float(
        (
            0.5 + 0.5 * (density[0, 0] - density[1, 1]) - density[0, 1] - density[1, 0]
        ).real
    )


def solve_energy(*, maxiter: int = 200) -> Variational:
    """Retain the source's zero initialization and at most 200 iterations."""
    if (
        isinstance(maxiter, bool)
        or not isinstance(maxiter, int)
        or not 1 <= maxiter <= 200
    ):
        raise ValueError("maxiter must be an integer in [1, 200]")
    trace: list[float] = []

    def objective(parameters: torch.Tensor) -> torch.Tensor:
        value = energy(tuple(float(v) for v in parameters))
        trace.append(value)
        return torch.tensor(value, dtype=torch.float64)

    optimizer = NelderMeadOptimizer(maxiter=maxiter).minimize(
        objective, torch.zeros(3, dtype=torch.float64)
    )
    program = build_variational(tuple(float(v) for v in optimizer.parameters))
    return Variational(optimizer, tuple(trace), program)


def build_classifier(input_angle: float, weight: float) -> synth.Program:
    """Reuse the frozen native encoding/mixing RX helper composition."""
    _finite(input_angle, "input_angle")
    _finite(weight, "weight")
    return synth.synthesize(
        classifier,
        classical_inputs={"input_0": input_angle, "weight_0": weight},
        max_qubits=1,
        max_workspace=0,
        max_operations=4,
        memory_limit_bytes=1024**2,
    )


def probability_zero(input_angle: float, weight: float) -> float:
    """Read the named one-bit output of an actual qfunc execution."""
    program = build_classifier(input_angle, weight)
    result = fq.run(program)
    return dict(program.probabilities(result, registers=("res",))).get((0,), 0.0)


def probability_gradient(input_angle: float, weight: float) -> float:
    """Differentiate the single RX weight with two actual shifted executions."""
    _finite(weight, "weight")
    return 0.5 * (
        probability_zero(input_angle, weight + math.pi / 2)
        - probability_zero(input_angle, weight - math.pi / 2)
    )


def train_not(
    *, epochs: int = 5, initial_weight: float = 3.0, learning_rate: float = 1.0
) -> Training:
    """Use quantum shift gradients with the source's MAE and PyTorch SGD.

    Classical qfunc bindings are static: gradients are explicitly evaluated and
    assigned to the optimizer-owned parameter. This example does not claim that
    synthesis through a float conversion is differentiable.
    """
    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or not 1 <= epochs <= 200
    ):
        raise ValueError("epochs must be an integer in [1, 200]")
    _finite(initial_weight, "initial_weight")
    _finite(learning_rate, "learning_rate")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    parameter = torch.nn.Parameter(torch.tensor(initial_weight, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=learning_rate)
    inputs = (math.pi, 0.0)
    labels = torch.tensor([1.0, 0.0], dtype=torch.float64)
    steps: list[Step] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        before = float(parameter.detach())
        predictions = torch.tensor(
            [probability_zero(x, before) for x in inputs], dtype=torch.float64
        )
        derivatives = torch.tensor(
            [probability_gradient(x, before) for x in inputs], dtype=torch.float64
        )
        # MAE's derivative is sign(prediction-label), including zero at equality.
        gradient = (torch.sign(predictions - labels) * derivatives).mean()
        parameter.grad = gradient
        loss = float(torch.nn.functional.l1_loss(predictions, labels))
        optimizer.step()
        steps.append(Step(before, loss, float(gradient), float(parameter.detach())))
    weight = float(parameter.detach())
    predictions_final = (
        probability_zero(inputs[0], weight),
        probability_zero(inputs[1], weight),
    )
    accuracy = float(
        torch.isclose(
            torch.tensor(predictions_final, dtype=torch.float64),
            labels,
            atol=1e-2,
            rtol=0,
        )
        .to(torch.float64)
        .mean()
    )
    return Training(
        tuple(steps),
        weight,
        predictions_final,
        accuracy,
        build_classifier(inputs[0], weight),
    )


def run_workflow() -> Workflow:
    """Run both original QML sections with the declared local adaptations.

    Examples:
        >>> workflow = run_workflow()
        >>> len(workflow.training.steps)
        5
    """
    return Workflow(solve_energy(), train_not())


def plot_workflow(workflow: Workflow) -> Figure:
    """Plot actual VQE evaluations and the five observed pre-update MAE values."""
    figure = Figure(figsize=(10, 4))
    energy_axes, training_axes = figure.subplots(1, 2)
    energy_axes.plot(range(len(workflow.variational.trace)), workflow.variational.trace)
    energy_axes.set(
        xlabel="Quantum objective evaluation", ylabel="Energy", title="Native VQE cost"
    )
    training_axes.plot(
        range(len(workflow.training.steps)),
        [row.loss for row in workflow.training.steps],
        marker="o",
    )
    training_axes.set(
        xlabel="Epoch", ylabel="Mean absolute error", title="Native NOT training"
    )
    figure.tight_layout()
    return figure
