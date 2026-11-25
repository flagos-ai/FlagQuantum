"""CPU neural networks with a batched native four-qubit sampled layer."""

import hashlib
import math
from pathlib import Path

import torch
from torch import nn

import flagquantum as fq
from examples.synthesis.algorithms.machine_learning.qgan.qgan_bars_and_strips.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.algorithms.machine_learning.qgan.qgan_bars_and_strips.main import (
    build_program,
)

SHOTS = 4096
INPUT_WIRES = (3, 2, 1, 0)
CLASSICAL_ASSET = "generator_trained_model.pth"
QUANTUM_ASSET = "q_generator_trained_model.pth"
ASSET_HASHES = {
    CLASSICAL_ASSET: "21bae21bcb4e2abd763e8fd77adff64da9439562cc7758046327ca626f6e507e",
    QUANTUM_ASSET: "85a2a8ad1f7d3495857808a5cbab7f2e4d186c2cd1b89de53e33926bd72af19b",
}


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Sigmoid(),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return torch.round(self.model(data))


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 16),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        value = self.model(data.reshape(-1, 4))
        assert isinstance(value, torch.Tensor)
        return value


def batched_state(data: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Bind all rows as differentiable angles on one native batched Circuit."""
    if data.ndim != 2 or data.shape[1] != 4 or weights.shape != (8,):
        raise ValueError("the source layer needs four features and eight weights")
    circuit = fq.Circuit(4, bsz=data.shape[0], dtype=torch.complex128)
    for index, wire in enumerate(INPUT_WIRES):
        circuit.rx(wire, theta=math.pi * data[:, index].to(torch.float64))
    for index, wire in enumerate(INPUT_WIRES):
        circuit.rz(wire, theta=math.pi * data[:, index].to(torch.float64))
    for index, wire in enumerate(INPUT_WIRES):
        circuit.ry(wire, theta=weights[index].to(torch.float64))
    for index, (left, right) in enumerate(
        zip(INPUT_WIRES[:-1], INPUT_WIRES[1:], strict=True)
    ):
        circuit.rzz(left, right, theta=weights[4 + index].to(torch.float64))
    circuit.rzz(3, 2, theta=weights[7].to(torch.float64))
    state = circuit.state()
    assert isinstance(state, torch.Tensor)
    return state


def check_states(weights: torch.Tensor) -> float:
    """Compare nonzero bindings to both an independent Circuit and native qfunc."""
    features = torch.tensor(
        [(0.17, 0.38, 0.53, 0.79), (-0.43, -0.28, 0.63, 0.39)], dtype=torch.float64
    )
    full = batched_state(features, weights)
    values = weights.detach().tolist()
    error = 0.0
    for state, row in zip(full, features.tolist(), strict=True):
        independent = fq.run(reference_circuit(row, values)).statevector().flatten()
        qfunc_state = fq.run(build_program(row, values)).statevector().flatten()
        error = max(
            error,
            float((state - independent).abs().max().detach()),
            float((state - qfunc_state).abs().max().detach()),
        )
    if error > 3e-12:
        raise AssertionError(
            f"batch layer differs from the original quantum function by {error}"
        )
    return error


class SampledLayer(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(8))
        self.generator = torch.Generator().manual_seed(seed)
        self.query_rows = 0

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        probability = batched_state(data, self.weight).abs().square()
        outcome_bits = (torch.arange(16)[:, None] >> torch.arange(4)) & 1
        classical = outcome_bits.to(probability.dtype)
        exact_mean = probability @ classical
        words = torch.multinomial(
            probability.detach(), SHOTS, replacement=True, generator=self.generator
        )
        sample_mean = classical[words].mean(dim=1)
        self.query_rows += data.shape[0]
        # The finite-shot value is used in the forward pass. Use the exact
        # observable derivative locally before the original final round.
        return (sample_mean + (exact_mean - exact_mean.detach())).to(data.dtype)


class QuantumGenerator(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(4, 16)
        self.linear_2 = nn.Linear(16, 32)
        self.linear_3 = nn.Linear(32, 16)
        self.linear_4 = nn.Linear(16, 4)
        # The checkpoint includes this unused layer.
        self.linear_5 = nn.Linear(2, 4)
        self.qlayer = SampledLayer(seed)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = torch.sigmoid(self.linear_1(self.flatten(data)))
        data = torch.relu(self.linear_2(data))
        data = torch.relu(self.linear_3(data))
        data = torch.sigmoid(self.linear_4(data))
        return torch.round(torch.sigmoid(self.qlayer.forward(data)))


def load_checkpoint(module: nn.Module, directory: Path, name: str) -> None:
    """Load tensor weights only, after checking the requested original assets."""
    path = directory / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != ASSET_HASHES[name]:
        raise ValueError(f"checkpoint differs from the pinned original: {name}")
    module.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
