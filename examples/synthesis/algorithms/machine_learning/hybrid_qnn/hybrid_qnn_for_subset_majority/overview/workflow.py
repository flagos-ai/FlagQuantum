"""Checkpoint-backed subset-majority training with sampled native quantum readout."""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from matplotlib.figure import Figure
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import flagquantum as fq
from examples.synthesis.algorithms.machine_learning.hybrid_qnn.hybrid_qnn_for_subset_majority.main import (
    build_program,
)
from flagquantum import synth

SHOTS = 4096
SUBSET = (1, 3, 4, 6, 7, 9)


@dataclass(frozen=True)
class Dataset:
    """The source's disjoint 340/512 split of 1024 shuffled ten-bit words."""

    training: torch.Tensor
    training_labels: torch.Tensor
    test: torch.Tensor
    test_labels: torch.Tensor


@dataclass(frozen=True)
class Step:
    """One full or final partial batch, including all parameter update evidence."""

    size: int
    loss: float
    parameters_before: dict[str, torch.Tensor]
    gradients: dict[str, torch.Tensor]
    parameters_after: dict[str, torch.Tensor]


@dataclass(frozen=True)
class Workflow:
    """One executed epoch and the actual sampled 512-example test result."""

    data: Dataset
    steps: tuple[Step, ...]
    predictions: torch.Tensor
    test_labels: torch.Tensor
    accuracy: float
    quantum_evaluations: int
    program: synth.Program


def make_dataset() -> Dataset:
    """Keep source bit order, shuffle seed, subset, strict threshold and split."""
    words = [[int(bit) for bit in f"{word:010b}"] for word in range(1024)]
    random.Random(1).shuffle(words)
    values = torch.tensor(words, dtype=torch.float64)
    labels = (values[:, SUBSET].sum(dim=1) > 3).to(torch.float64).unsqueeze(1)
    return Dataset(values[:340], labels[:340], values[340:852], labels[340:852])


class HybridNetwork(nn.Module):
    """Classical 10-to-4 Tanh and 2-to-1 ReLU layers around three quantum weights.

    The workflow explicitly transfers quantum shift gradients between these
    stages. Static float synthesis is not a generic differentiable QLayer.
    """

    def __init__(self) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.linear_1 = nn.Linear(10, 4, dtype=torch.float64)
            self.linear_2 = nn.Linear(2, 1, dtype=torch.float64)
        self.quantum_weight = nn.Parameter(torch.zeros(3, dtype=torch.float64))

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Compute the four classical Bloch-encoding inputs."""
        return torch.tanh(self.linear_1(data.flatten(start_dim=1)))

    def decode(self, readout: torch.Tensor) -> torch.Tensor:
        """Predict one nonnegative score from the two quantum P(1) values."""
        return torch.relu(self.linear_2(readout))


def load_network() -> HybridNetwork:
    """Load reviewed checkpoint tensor values from portable JSON, without pickle."""
    source: object = json.loads(Path(__file__).with_name("checkpoint.json").read_text())
    if not isinstance(source, dict):
        raise ValueError("checkpoint must contain named tensor arrays")
    network = HybridNetwork()
    parameters: dict[str, torch.Tensor] = {}
    for name, value in source.items():
        if not isinstance(name, str) or not isinstance(value, list):
            raise ValueError("checkpoint entries must be named tensor arrays")
        tensor = torch.tensor(value, dtype=torch.float64)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("checkpoint tensors must be finite")
        parameters["quantum_weight" if name == "qlayer.weight" else name] = tensor
    network.load_state_dict(parameters, strict=True)
    return network


def quantum_readout(
    features: torch.Tensor,
    weights: torch.Tensor,
    *,
    shots: int | None = SHOTS,
    seed: int = 0,
) -> torch.Tensor:
    """Execute native qfunc and return both logical P(1) marginals.

    Each sampled call observes 4096 joint outcomes by default. Explicit
    ``shots=None`` returns exact marginals for independent gradient acceptance.
    """
    if features.shape != (4,) or weights.shape != (3,):
        raise ValueError("readout requires four features and three weights")
    if not bool(torch.isfinite(features).all() and torch.isfinite(weights).all()):
        raise ValueError("features and weights must be finite")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if shots is not None and (
        isinstance(shots, bool) or not isinstance(shots, int) or not 1 <= shots <= 65536
    ):
        raise ValueError("shots must be None or an integer between 1 and 65536")
    program = build_program(features.detach().tolist(), weights.detach().tolist())
    values = dict(program.probabilities(fq.run(program), registers=("result",)))
    probabilities = torch.tensor(
        [values.get((word,), 0.0) for word in range(4)], dtype=torch.float64
    )
    if shots is not None:
        samples = torch.multinomial(
            probabilities,
            shots,
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        probabilities = torch.bincount(samples, minlength=4).to(torch.float64) / shots
    return torch.stack(
        (probabilities[1] + probabilities[3], probabilities[2] + probabilities[3])
    )


def quantum_jacobians(
    features: torch.Tensor,
    weights: torch.Tensor,
    *,
    shots: int | None = SHOTS,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate all four features and all three weights by quantum shifts."""
    feature_columns: list[torch.Tensor] = []
    weight_columns: list[torch.Tensor] = []
    for index in range(7):
        feature_shift = torch.zeros(4, dtype=torch.float64)
        weight_shift = torch.zeros(3, dtype=torch.float64)
        if index < 4:
            # Encoding gates use pi * feature: shift the feature by half a turn.
            feature_shift[index] = 0.5
            multiplier = math.pi / 2
        else:
            weight_shift[index - 4] = math.pi / 2
            multiplier = 0.5
        plus = quantum_readout(
            features + feature_shift,
            weights + weight_shift,
            shots=shots,
            seed=seed + 2 * index,
        )
        minus = quantum_readout(
            features - feature_shift,
            weights - weight_shift,
            shots=shots,
            seed=seed + 2 * index + 1,
        )
        column = multiplier * (plus - minus)
        (feature_columns if index < 4 else weight_columns).append(column)
    return torch.stack(feature_columns, dim=1), torch.stack(weight_columns, dim=1)


def _snapshot(network: HybridNetwork) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in network.named_parameters()}


def _step(
    network: HybridNetwork,
    optimizer: torch.optim.Optimizer,
    data: torch.Tensor,
    labels: torch.Tensor,
    seed: int,
) -> Step:
    optimizer.zero_grad()
    before = _snapshot(network)
    features = network.encode(data)
    readout = torch.stack(
        [
            quantum_readout(row, network.quantum_weight, seed=seed + 15 * index)
            for index, row in enumerate(features.detach())
        ]
    ).requires_grad_()
    predictions = network.decode(readout)
    loss = nn.functional.mse_loss(predictions, labels)
    output_parameters = tuple(network.linear_2.parameters())
    readout_gradient, *output_gradients = torch.autograd.grad(
        loss, (readout, *output_parameters)
    )
    for parameter, gradient in zip(output_parameters, output_gradients, strict=True):
        parameter.grad = gradient
    feature_gradients = []
    weight_gradient = torch.zeros(3, dtype=torch.float64)
    for index, row in enumerate(features.detach()):
        feature_jacobian, weight_jacobian = quantum_jacobians(
            row,
            network.quantum_weight.detach(),
            seed=seed + 15 * index + 1,
        )
        feature_gradients.append(feature_jacobian.T @ readout_gradient[index])
        weight_gradient += weight_jacobian.T @ readout_gradient[index]
    input_parameters = tuple(network.linear_1.parameters())
    input_gradients = torch.autograd.grad(
        features, input_parameters, grad_outputs=torch.stack(feature_gradients)
    )
    for parameter, gradient in zip(input_parameters, input_gradients, strict=True):
        parameter.grad = gradient
    network.quantum_weight.grad = weight_gradient
    gradients = {}
    for name, parameter in network.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"missing gradient for {name}")
        gradients[name] = parameter.grad.detach().clone()
    optimizer.step()
    return Step(len(data), float(loss.detach()), before, gradients, _snapshot(network))


def run_workflow(*, seed: int = 17) -> Workflow:
    """Train all 340 examples once, then evaluate 512 examples at 4096 shots.

    Examples:
        >>> workflow = run_workflow()
        >>> [step.size for step in workflow.steps]
        [64, 64, 64, 64, 64, 20]
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer between zero and 2**31 - 1")
    data = make_dataset()
    network = load_network()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.05)
    training = DataLoader(
        TensorDataset(data.training, data.training_labels),
        batch_size=64,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )
    steps = []
    evaluations = 0
    for values, labels in training:
        steps.append(_step(network, optimizer, values, labels, seed + evaluations))
        evaluations += 15 * len(values)
    testing = DataLoader(
        TensorDataset(data.test, data.test_labels),
        batch_size=64,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    network.eval()
    predictions = []
    observed_labels = []
    with torch.no_grad():
        for values, labels in testing:
            features = network.encode(values)
            readout = torch.stack(
                [
                    quantum_readout(
                        row, network.quantum_weight, seed=seed + evaluations + index
                    )
                    for index, row in enumerate(features)
                ]
            )
            evaluations += len(values)
            predictions.append(network.decode(readout))
            observed_labels.append(labels)
    observed = torch.cat(predictions)
    targets = torch.cat(observed_labels)
    accuracy = float((observed.round() == targets).to(torch.float64).mean())
    program = build_program(
        network.encode(data.test[:1]).detach()[0].tolist(),
        network.quantum_weight.detach().tolist(),
    )
    return Workflow(
        data, tuple(steps), observed, targets, accuracy, evaluations, program
    )


def plot_workflow(workflow: Workflow) -> Figure:
    """Plot the actual six minibatch losses and sampled held-out predictions."""
    figure = Figure(figsize=(10, 3.5), constrained_layout=True)
    losses, predictions = figure.subplots(1, 2)
    losses.plot(
        range(1, len(workflow.steps) + 1), [step.loss for step in workflow.steps], "o-"
    )
    losses.set(
        xlabel="Minibatch", ylabel="MSE", title="One epoch: 340 training examples"
    )
    for label in (0, 1):
        mask = workflow.test_labels[:, 0] == label
        predictions.hist(
            workflow.predictions[mask, 0].tolist(),
            bins=20,
            alpha=0.6,
            label=f"Label {label}",
        )
    predictions.set(
        xlabel="Sampled model score",
        ylabel="Examples",
        title=f"512 test examples; accuracy {workflow.accuracy:.3f}",
    )
    predictions.legend()
    return figure
