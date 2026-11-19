"""Sampled native autoencoder training, channel decoding and anomaly detection."""

import math
from dataclasses import dataclass
from typing import Literal

import torch
from matplotlib.figure import Figure
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import flagquantum as fq
from examples.synthesis.algorithms.quantum_primitives.swap_test.main import overlap_test
from flagquantum import synth
from flagquantum.synth import (
    CX,
    RESET,
    Output,
    QArray,
    QBit,
    R,
    adjoint,
    allocate,
    drop,
    qfunc,
)

Cost = Literal["swap", "trash"]
SHOTS = 4096
DOMAIN_WALL = ((0, 0, 1, 1), (0, 0, 0, 1), (0, 1, 1, 1))
INITIAL_WEIGHTS = (
    1.5227,
    0.3588,
    0.6905,
    1.4777,
    1.5718,
    1.5615,
    1.5414,
    0.6021,
    0.1254,
    0.9903,
)


@qfunc
def ansatz(weights: list[float], q: QArray[QBit]) -> None:
    for layer in range(2):
        for i in range(4):
            R(weights[4 * layer + i], math.pi / 2, q[i])
        for i in range(3):
            CX(q[i], q[i + 1])
    for i in range(2):
        R(weights[8 + i], math.pi / 2, q[2 + i])


@qfunc
def encode(data: list[float], weights: list[float], q: QArray[QBit]) -> None:
    for i in range(4):
        R(math.pi * data[i], math.pi / 2, q[i])
    ansatz(weights, q)


@qfunc
def encoder(data: list[float], weights: list[float], q: Output[QArray[QBit]]) -> None:
    allocate(4, q)
    encode(data, weights, q)


@qfunc
def swap_encoder(
    data: list[float], weights: list[float], q: Output[QArray[QBit]], test: Output[QBit]
) -> None:
    reference: QArray[QBit] = QArray()
    allocate(2, reference)
    allocate(4, q)
    allocate(test)
    encode(data, weights, q)
    overlap_test(q[2:4], reference, test)
    drop(reference)


@qfunc
def decoder(data: list[float], weights: list[float], q: Output[QArray[QBit]]) -> None:
    allocate(4, q)
    encode(data, weights, q)
    RESET(q[2])
    RESET(q[3])
    with adjoint():
        ansatz(weights, q)


def build_program(
    data: torch.Tensor,
    weights: torch.Tensor,
    *,
    mode: Literal["swap", "trash", "decode"],
) -> synth.Program:
    """Bind four finite data values and ten angles to one bounded native program."""
    if data.shape != (4,) or weights.shape != (10,):
        raise ValueError("use four data values and ten encoder weights")
    if not bool(torch.isfinite(data).all() and torch.isfinite(weights).all()):
        raise ValueError("data and encoder weights must be finite")
    if mode not in {"swap", "trash", "decode"}:
        raise ValueError("mode must be swap, trash or decode")
    return synth.synthesize(
        {"swap": swap_encoder, "trash": encoder, "decode": decoder}[mode],
        classical_inputs={
            "data": data.detach().tolist(),
            "weights": weights.detach().tolist(),
        },
        max_qubits=7,
        max_workspace=0,
        memory_limit_bytes=64 * 1024**2,
    )


def initial_weights() -> torch.Tensor:
    """Preserve the source's float32 checkpoint rounding before float64 training."""
    return torch.tensor(INITIAL_WEIGHTS, dtype=torch.float32).to(torch.float64)


def _sample(probabilities: torch.Tensor, shots: int | None, seed: int) -> torch.Tensor:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if shots is not None:
        if (
            isinstance(shots, bool)
            or not isinstance(shots, int)
            or not 1 <= shots <= 65536
        ):
            raise ValueError("shots must be None or an integer between 1 and 65536")
        draws = torch.multinomial(
            probabilities,
            shots,
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        return (
            torch.bincount(draws, minlength=len(probabilities)).to(torch.float64)
            / shots
        )
    return probabilities


def readout(
    data: torch.Tensor,
    weights: torch.Tensor,
    *,
    cost: Cost,
    shots: int | None = SHOTS,
    seed: int = 0,
) -> torch.Tensor:
    """Read P(test=1) or the expected number of nonzero trash bits.

    Exact mode is explicit (shots=None); default measurements draw 4096 joint
    outcomes from the actual local quantum execution. Array bit i contributes
    2**i to the named integer q, so trash is bits 2 and 3.
    """
    program = build_program(data, weights, mode=cost)
    values = dict(
        program.probabilities(
            fq.run(program), registers=("test" if cost == "swap" else "q",)
        )
    )
    width = 2 if cost == "swap" else 16
    probabilities = _sample(
        torch.tensor(
            [values.get((i,), 0.0) for i in range(width)], dtype=torch.float64
        ),
        shots,
        seed,
    )
    if cost == "swap":
        return probabilities[1]
    counts = torch.tensor(
        [(i >> 2 & 1) + (i >> 3 & 1) for i in range(16)], dtype=torch.float64
    )
    return probabilities @ counts


def quantum_gradient(
    data: torch.Tensor,
    weights: torch.Tensor,
    *,
    cost: Cost,
    shots: int | None = SHOTS,
    seed: int = 0,
) -> torch.Tensor:
    """Differentiate all ten independently occurring RY angles by shifts."""
    columns = []
    for index in range(10):
        shift = torch.zeros(10, dtype=torch.float64)
        shift[index] = math.pi / 2
        plus = readout(
            data, weights + shift, cost=cost, shots=shots, seed=seed + 2 * index
        )
        minus = readout(
            data, weights - shift, cost=cost, shots=shots, seed=seed + 2 * index + 1
        )
        columns.append(0.5 * (plus - minus))
    return torch.stack(columns)


@dataclass(frozen=True)
class Step:
    """A full or final partial minibatch and its actual L1 SGD update."""

    data: torch.Tensor
    predictions: torch.Tensor
    loss: float
    weights_before: torch.Tensor
    gradient: torch.Tensor
    weights_after: torch.Tensor


@dataclass(frozen=True)
class Training:
    """One executed epoch, with source initialization, batch size and shot budget."""

    cost: Cost
    steps: tuple[Step, ...]
    weights: torch.Tensor


def train(cost: Cost, *, seed: int = 7) -> Training:
    """Train one epoch with batch 2, L1 loss, SGD lr=0.3 and quantum gradients."""
    weights = nn.Parameter(initial_weights())
    optimizer = torch.optim.SGD((weights,), lr=0.3)
    loader = DataLoader(
        TensorDataset(torch.tensor(DOMAIN_WALL, dtype=torch.float64)),
        batch_size=2,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )
    steps = []
    for batch_index, (data,) in enumerate(loader):
        optimizer.zero_grad()
        before = weights.detach().clone()
        batch_seed = seed + batch_index * 1000
        predictions = torch.stack(
            [
                readout(row, before, cost=cost, seed=batch_seed + index * 21)
                for index, row in enumerate(data)
            ]
        ).requires_grad_()
        loss = nn.functional.l1_loss(predictions, torch.zeros_like(predictions))
        sensitivity = torch.autograd.grad(loss, predictions)[0]
        gradient = torch.zeros(10, dtype=torch.float64)
        for index, row in enumerate(data):
            gradient += sensitivity[index] * quantum_gradient(
                row, before, cost=cost, seed=batch_seed + index * 21 + 1
            )
        weights.grad = gradient
        optimizer.step()
        steps.append(
            Step(
                data.clone(),
                predictions.detach(),
                float(loss.detach()),
                before,
                gradient.clone(),
                weights.detach().clone(),
            )
        )
    return Training(cost, tuple(steps), weights.detach().clone())


@dataclass(frozen=True)
class Decoded:
    """The actual mixed output channel and sampled modal logical data bits."""

    data: tuple[int, ...]
    density: torch.Tensor
    probabilities: torch.Tensor
    modal_bits: tuple[int, ...]


def decode(data: tuple[int, ...], weights: torch.Tensor, *, seed: int = 0) -> Decoded:
    """Reset trash, invert the encoder and sample the resulting mixed state."""
    program = build_program(
        torch.tensor(data, dtype=torch.float64), weights, mode="decode"
    )
    result = fq.run(program)
    density = result.state
    assert isinstance(density, torch.Tensor)
    values = dict(program.probabilities(result, registers=("q",)))
    probabilities = _sample(
        torch.tensor([values.get((i,), 0.0) for i in range(16)], dtype=torch.float64),
        SHOTS,
        seed,
    )
    word = int(probabilities.argmax())
    return Decoded(
        data, density.clone(), probabilities, tuple(word >> i & 1 for i in range(4))
    )


@dataclass(frozen=True)
class Anomaly:
    """Sampled swap cost and the source's strict 0.01 detection threshold."""

    data: tuple[int, ...]
    cost: float
    detected: bool


@dataclass(frozen=True)
class Workflow:
    """Both original training routes, three decoder channels and five anomalies."""

    swap: Training
    trash: Training
    decoded: tuple[Decoded, ...]
    anomalies: tuple[Anomaly, ...]
    quantum_evaluations: int


def run_workflow() -> Workflow:
    """Execute the complete source workflow locally with explicit native seeds.

    Examples:
        >>> report = run_workflow()
        >>> len(report.decoded), len(report.anomalies)
        (3, 5)
    """
    swap = train("swap")
    trash = train("trash")
    decoded = tuple(
        decode(row, swap.weights, seed=100 + i) for i, row in enumerate(DOMAIN_WALL)
    )
    rows = (*DOMAIN_WALL, (1, 0, 1, 0), (1, 1, 1, 1))
    order = torch.randperm(5, generator=torch.Generator().manual_seed(19)).tolist()
    anomalies = []
    for i in order:
        value = float(
            readout(
                torch.tensor(rows[i], dtype=torch.float64),
                swap.weights,
                cost="swap",
                seed=200 + i,
            )
        )
        anomalies.append(Anomaly(rows[i], value, abs(value) > 0.01))
    # 2 routes * 3 samples * (1 forward + 20 shifts), plus 3 decodes + 5 anomalies.
    return Workflow(swap, trash, decoded, tuple(anomalies), 134)


def plot_workflow(workflow: Workflow) -> Figure:
    """Plot the measured minibatch losses and all five measured anomaly scores."""
    figure = Figure(figsize=(11, 4), constrained_layout=True)
    losses, anomalies = figure.subplots(1, 2)
    for training in (workflow.swap, workflow.trash):
        losses.plot(
            (1, 2),
            [step.loss for step in training.steps],
            marker="o",
            label=training.cost,
        )
    losses.set(
        xlabel="Minibatch (sizes 2, 1)",
        ylabel="Sampled L1 loss",
        title="One epoch per cost",
    )
    losses.legend()
    anomalies.bar(
        ["".join(map(str, row.data)) for row in workflow.anomalies],
        [row.cost for row in workflow.anomalies],
    )
    anomalies.axhline(0.01, color="red", linestyle="--", label="Threshold 0.01")
    anomalies.set(
        xlabel="Logical data bits (index 0 first)",
        ylabel="Sampled P(test=1)",
        title="Domain walls and anomalies",
    )
    anomalies.legend()
    return figure
