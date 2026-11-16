"""Train the source subset-majority network on the local exact CPU path."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import flagquantum as fq

SUBSET_INDICES = (1, 3, 4, 6, 7, 9)


def make_datasets() -> tuple[TensorDataset, TensorDataset]:
    """Reproduce the notebook's shuffle and disjoint 340/512 split."""
    rows = [[int(bit) for bit in f"{value:010b}"] for value in range(1024)]
    random.Random(1).shuffle(rows)
    features = torch.tensor(rows, dtype=torch.float64)
    labels = (features[:, SUBSET_INDICES].sum(dim=1) > 3).to(features.dtype)
    labels = labels.unsqueeze(1)
    return (
        TensorDataset(features[:340], labels[:340]),
        TensorDataset(features[340:852], labels[340:852]),
    )


def quantum_circuit(weights: torch.Tensor, features: torch.Tensor) -> fq.Circuit:
    """Preserve the source register order while keeping live autograd tensors."""
    circuit = fq.Circuit(2, bsz=features.shape[0], dtype=torch.complex128, device="cpu")
    for index, wire in enumerate((1, 0)):
        circuit.rx(wire, theta=math.pi * features[..., 2 * index])
        circuit.rz(wire, theta=math.pi * features[..., 2 * index + 1])
        circuit.ry(wire, theta=weights[index])
    circuit.rzz(1, 0, theta=weights[2])
    return circuit


class SubsetMajorityNetwork(nn.Module):
    """The notebook's Linear/Tanh/quantum/Linear/ReLU composition.

    Examples:
        >>> network = SubsetMajorityNetwork()
        >>> load_pretrained(network)
        >>> network(torch.zeros(2, 10, dtype=torch.float64)).shape
        torch.Size([2, 1])
    """

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(10, 4, dtype=torch.float64)
        self.activation_1 = nn.Tanh()
        self.qlayer = fq.Module(
            quantum_circuit,
            n_parameters=3,
            dtype=torch.float64,
            init=torch.zeros(3, dtype=torch.float64),
            policy=fq.RuntimePolicy(
                execution_options=fq.ExecutionOptions(
                    mode="statevector",
                    backend="pytorch",
                    device="cpu",
                    precision="complex128",
                    allow_backend_fallback=False,
                ),
                observable_qubits=(1, 0),
            ),
        )
        self.linear_2 = nn.Linear(2, 1, dtype=torch.float64)
        self.activation_2 = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.activation_1(self.linear_1(self.flatten(features)))
        prob_one = (1 - self.qlayer(encoded)) / 2
        output: torch.Tensor = self.activation_2(self.linear_2(prob_one))
        return output


def load_pretrained(network: SubsetMajorityNetwork) -> None:
    """Load the pinned upstream tensors stored as inspectable JSON, without pickle."""
    state = json.loads(Path(__file__).with_name("pretrained.json").read_text())
    with torch.no_grad():
        for name, parameter in network.named_parameters():
            source_name = (
                "qlayer.weight" if name == "qlayer.parameters_tensor" else name
            )
            parameter.copy_(torch.tensor(state[source_name], dtype=parameter.dtype))


@dataclass(frozen=True)
class Metrics:
    """Exact mean squared error and the source's rounded-label accuracy."""

    loss: float
    accuracy: float
    samples: int


def evaluate(network: nn.Module, dataset: TensorDataset) -> Metrics:
    """Evaluate every row, including the final incomplete batch."""
    network.eval()
    squared_error = 0.0
    correct = 0
    with torch.no_grad():
        for features, labels in DataLoader(dataset, batch_size=64):
            predictions = network(features)
            squared_error += float((predictions - labels).square().sum())
            correct += int((predictions.round() == labels).sum())
    return Metrics(squared_error / len(dataset), correct / len(dataset), len(dataset))


def train(
    network: nn.Module, dataset: TensorDataset, *, epochs: int = 1, seed: int = 0
) -> list[float]:
    """Run source SGD (lr=0.05), MSE and shuffled batches of 64.

    Returns one sample-weighted loss per epoch; this is a training trajectory,
    not a guarantee that SGD improves the objective at every step.
    """
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=generator)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.05)
    history = []
    network.train()
    for epoch in range(epochs):
        total = 0.0
        for features, labels in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(network(features), labels)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("subset-majority training produced a nonfinite loss")
            torch.autograd.backward(loss)
            optimizer.step()
            total += float(loss.detach()) * len(features)
        history.append(total / len(dataset))
        print(
            json.dumps({"epoch": epoch + 1, "training_loss": history[-1]}), flush=True
        )
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--from-scratch", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(1)
    training, testing = make_datasets()
    network = SubsetMajorityNetwork()
    if not args.from_scratch:
        load_pretrained(network)
    before = evaluate(network, testing)
    train(network, training, epochs=args.epochs)
    print(
        json.dumps(
            {"before": asdict(before), "after": asdict(evaluate(network, testing))}
        )
    )


if __name__ == "__main__":
    main()
