"""Fresh datasets and the original classical and hybrid adversarial schedules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .neural import (
    CLASSICAL_ASSET,
    QUANTUM_ASSET,
    SHOTS,
    Discriminator,
    Generator,
    QuantumGenerator,
    check_states,
    load_checkpoint,
)

if TYPE_CHECKING:
    from numpy.random import RandomState

CLASSICAL_ROWS = 1000
CLASSICAL_EPOCHS = 10
QUANTUM_ROWS = 250
QUANTUM_EPOCHS = 1
BATCH_SIZE = 64
SEED = 42


@dataclass(frozen=True)
class Training:
    losses: tuple[tuple[float, float, float], ...]
    changed_generator_tensors: int
    changed_discriminator_tensors: int
    maximum_generator_gradient: float


@dataclass(frozen=True)
class Scores:
    diagonal_rule: float
    bars_and_stripes: float
    distinct_patterns: int


@dataclass(frozen=True)
class ClassicalRun:
    initial_examples: torch.Tensor
    generated_examples: torch.Tensor
    fresh_comparison_examples: torch.Tensor
    training: Training
    scores: Scores


@dataclass(frozen=True)
class QuantumRun:
    examples: torch.Tensor
    training: Training
    scores: Scores
    query_rows: int
    finite_draws: int
    maximum_state_error: float


def dataset(rows: int, rng: RandomState) -> torch.Tensor:
    """Regenerate the two horizontal or vertical binary stripe coordinates."""
    import numpy as np

    samples = np.empty((rows, 2, 2), dtype=np.uint8)
    for i in range(rows):
        horizontal = rng.randint(0, 2) == 0
        stripes = rng.randint(0, 2, size=(2, 1) if horizontal else (1, 2))
        samples[i] = np.tile(stripes, (1, 2) if horizontal else (2, 1))
    return torch.from_numpy(samples).to(torch.float32)


def gaussian_noise(rows: int) -> torch.Tensor:
    return torch.randn(rows, 2)


def binary_noise(rows: int) -> torch.Tensor:
    return torch.bernoulli(torch.rand(rows, 4))


def _changes(module: nn.Module, before: dict[str, torch.Tensor]) -> int:
    return sum(
        not torch.equal(before[name], data)
        for name, data in module.state_dict().items()
    )


def train(
    generator: Generator | QuantumGenerator,
    discriminator: Discriminator,
    real_data: torch.Tensor,
    noise: Callable[[int], torch.Tensor],
    epochs: int,
) -> Training:
    """Run complete source batches with BCE, Adam and the original rounding."""
    loader = DataLoader(TensorDataset(real_data), batch_size=BATCH_SIZE, shuffle=True)
    loss_fn = nn.BCELoss()
    optimize_gen = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optimize_dis = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    prior_gen = {name: value.clone() for name, value in generator.state_dict().items()}
    prior_dis = {
        name: value.clone() for name, value in discriminator.state_dict().items()
    }
    loss_rows = []
    max_gen_gradient = 0.0
    for _ in range(epochs):
        for (real,) in loader:
            rows = real.shape[0]
            optimize_dis.zero_grad()
            real_output = discriminator.forward(real)
            real_loss = loss_fn(real_output, torch.ones_like(real_output))
            real_loss.backward()
            fake_output = discriminator.forward(generator.forward(noise(rows)).detach())
            fake_loss = loss_fn(fake_output, torch.zeros_like(fake_output))
            fake_loss.backward()
            optimize_dis.step()

            optimize_gen.zero_grad()
            generated = discriminator.forward(generator.forward(noise(rows)))
            generator_loss = loss_fn(generated, torch.ones_like(generated))
            generator_loss.backward()
            step_gradient = max(
                (
                    float(p.grad.abs().max())
                    for p in generator.parameters()
                    if p.grad is not None
                ),
                default=0.0,
            )
            max_gen_gradient = max(max_gen_gradient, step_gradient)
            optimize_gen.step()
            loss_rows.append(
                (
                    float(generator_loss.detach()),
                    float(real_loss.detach()),
                    float(fake_loss.detach()),
                )
            )
    return Training(
        tuple(loss_rows),
        _changes(generator, prior_gen),
        _changes(discriminator, prior_dis),
        max_gen_gradient,
    )


def score(data: torch.Tensor) -> Scores:
    """Keep the notebook's diagonal rule and show exact pattern membership."""
    images = data.reshape(-1, 2, 2)
    left = images[:, 0, 0] * images[:, 1, 1]
    right = images[:, 0, 1] * images[:, 1, 0]
    source_score = ~(left.bool() ^ right.bool())
    valid_rows = (images[:, 0, :] == images[:, 1, :]).all(dim=-1)
    valid_columns = (images[:, :, 0] == images[:, :, 1]).all(dim=-1)
    return Scores(
        float(source_score.to(torch.float64).mean()),
        float((valid_rows | valid_columns).to(torch.float64).mean()),
        len({tuple(row) for row in images.reshape(-1, 4).tolist()}),
    )


def run_classical(assets: Path, output: Path) -> ClassicalRun:
    """Run all classical minibatches and round-trip its trained state dictionary."""
    from numpy.random import RandomState

    output.mkdir(parents=True, exist_ok=True)
    rng = RandomState(SEED)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        source = dataset(CLASSICAL_ROWS, rng)
        initial_examples = dataset(9, rng)
        torch.randn(9, 2)  # Source visualization noise is generated before training.
        generator = Generator()
        discriminator = Discriminator()
        load_checkpoint(generator, assets, CLASSICAL_ASSET)
        classical_training = train(
            generator, discriminator, source, gaussian_noise, CLASSICAL_EPOCHS
        )
        classical_checkpoint = output / "generator_model.pth"
        torch.save(generator.state_dict(), classical_checkpoint)
        evaluated = Generator()
        evaluated.load_state_dict(torch.load(classical_checkpoint, weights_only=True))
        with torch.no_grad():
            classical_scores = score(evaluated.forward(gaussian_noise(100)))
        for_figures = Generator()
        for_figures.load_state_dict(torch.load(classical_checkpoint, weights_only=True))
        for_figures.eval()
        with torch.no_grad():
            classical_examples = for_figures.forward(gaussian_noise(9))
        fresh_examples = dataset(9, rng)

    assert len(classical_training.losses) == CLASSICAL_EPOCHS * (
        (CLASSICAL_ROWS + 63) // 64
    )
    return ClassicalRun(
        initial_examples,
        classical_examples,
        fresh_examples,
        classical_training,
        classical_scores,
    )


def run_quantum(assets: Path, output: Path) -> QuantumRun:
    """Run all hybrid minibatches with an isolated seeded CPU generator."""
    from numpy.random import RandomState

    output.mkdir(parents=True, exist_ok=True)
    rng = RandomState(SEED + 1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 1)
        quantum_generator = QuantumGenerator(seed=107)
        torch.randn(
            4
        )  # Keep the source visualization vector outside the training noise.
        quantum_source = dataset(QUANTUM_ROWS, rng)
        discriminator = Discriminator()
        load_checkpoint(quantum_generator, assets, QUANTUM_ASSET)
        error = check_states(quantum_generator.qlayer.weight)
        quantum_training = train(
            quantum_generator,
            discriminator,
            quantum_source,
            binary_noise,
            QUANTUM_EPOCHS,
        )
        quantum_checkpoint = output / "q_generator_model_bs64.pth"
        torch.save(quantum_generator.state_dict(), quantum_checkpoint)
        quantum_generator.load_state_dict(
            torch.load(quantum_checkpoint, weights_only=True)
        )
        with torch.no_grad():
            quantum_scores = score(quantum_generator.forward(binary_noise(10)))
            quantum_examples = quantum_generator.forward(binary_noise(10))
        queries = quantum_generator.qlayer.query_rows

    assert len(quantum_training.losses) == QUANTUM_EPOCHS * ((QUANTUM_ROWS + 63) // 64)
    assert queries == 2 * QUANTUM_ROWS * QUANTUM_EPOCHS + 20
    return QuantumRun(
        quantum_examples,
        quantum_training,
        quantum_scores,
        queries,
        queries * SHOTS,
        error,
    )
