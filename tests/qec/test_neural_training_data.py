"""Latent labels and independent circuit-sampler evidence for AI training data."""

from __future__ import annotations

import math

import pytest
import torch

from flagquantum.qec import (
    CssCode,
    CssCodeMatrices,
    DetectorErrorModel,
    PhenomenologicalNoise,
    RotatedSurfaceCode,
    build_memory_circuit,
    css_code_matrices,
    get_decoder,
    sample_memory_circuit,
)
from flagquantum.qec.circuits.memory import MemoryCircuit
from flagquantum.qec.decoders._neural import Predecoder, Trainer, evaluate

pytestmark = pytest.mark.integration


def _experiment(
    basis: str, rounds: int, probability: float
) -> tuple[MemoryCircuit, PhenomenologicalNoise, DetectorErrorModel]:
    surface = RotatedSurfaceCode(distance=3)
    matrices = css_code_matrices(surface)
    # The native surface record declares only logical Z. Use the existing CSS
    # constructor to validate the complementary logical X on the first column.
    code = CssCode(
        matrices=CssCodeMatrices(
            hz=matrices.hz,
            hx=matrices.hx,
            lz=matrices.lz,
            lx=torch.tensor([[1, 0, 0, 1, 0, 0, 1, 0, 0]]),
        ),
        distance=3,
    )
    memory = build_memory_circuit(code, rounds=rounds, readout_basis=basis)
    noise = PhenomenologicalNoise(
        data_flip=probability if basis == "z" else 0,
        phase_flip=probability if basis == "x" else 0,
        measurement_flip=probability,
    )
    dem = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    return memory, noise, dem


@pytest.mark.parametrize("basis", ["x", "z"])
@pytest.mark.parametrize("rounds", [1, 3])
def test_latent_labels_match_original_mechanism_signatures(
    basis: str, rounds: int
) -> None:
    _, _, dem = _experiment(basis, rounds, 0.02)
    batch = Predecoder(dem).sample(512, torch.Generator().manual_seed(11))
    # An independent XOR implementation; do not reuse H/L or dense multiplication.
    for shot, fired in enumerate(batch.mechanisms):
        detectors: set[int] = set()
        observables: set[int] = set()
        for index in fired.nonzero().flatten().tolist():
            detectors.symmetric_difference_update(dem.errors[index].detectors)
            observables.symmetric_difference_update(dem.errors[index].observables)
        assert (
            set(batch.sample.detectors[shot].nonzero().flatten().tolist()) == detectors
        )
        assert (
            set(batch.sample.observables[shot].nonzero().flatten().tolist())
            == observables
        )


@pytest.mark.parametrize("basis", ["x", "z"])
@pytest.mark.parametrize("rounds", [1, 3])
@pytest.mark.parametrize("probability", [0.01, 0.04])
def test_training_distribution_matches_independent_circuit_execution(
    basis: str, rounds: int, probability: float
) -> None:
    pytest.importorskip("stim")
    memory, noise, dem = _experiment(basis, rounds, probability)
    shots = 32768
    generated = Predecoder(dem).sample(shots, torch.Generator().manual_seed(17)).sample
    reference = sample_memory_circuit(memory, noise=noise, shots=shots, seed=29)
    learner_bits = torch.cat((generated.detectors, generated.observables), dim=1)
    circuit_bits = torch.cat((reference.detectors, reference.observables), dim=1)
    # Check every marginal and each detector's XOR with the logical label.
    # This detects joint-label corruption that matching marginal rates can miss.
    learner = torch.cat(
        (learner_bits, generated.detectors ^ generated.observables), dim=1
    )
    circuit = torch.cat(
        (circuit_bits, reference.detectors ^ reference.observables), dim=1
    )
    # Two independent Bernoulli means: union-bound Hoeffding across every
    # statistic and all eight profiles. Family-wise false rejection <= 1e-6.
    comparisons = learner.shape[1] * 8
    tolerance = math.sqrt(2 * math.log(4 * comparisons / 1e-6) / shots)
    difference = (learner.double().mean(0) - circuit.double().mean(0)).abs()
    assert bool((difference <= tolerance).all()), (float(difference.max()), tolerance)


def test_split_reproducibility_and_evaluation_preserves_training_stream() -> None:
    _, _, dem = _experiment("z", 1, 0.02)
    model = Predecoder(dem)
    seeds = (1, 11, 1001)
    samples = [model.sample(256, torch.Generator().manual_seed(seed)) for seed in seeds]
    for seed, batch in zip(seeds, samples, strict=True):
        replay = model.sample(256, torch.Generator().manual_seed(seed))
        assert torch.equal(replay.mechanisms, batch.mechanisms)
    assert not torch.equal(samples[0].mechanisms, samples[1].mechanisms)
    assert not torch.equal(samples[0].mechanisms, samples[2].mechanisms)
    trainer = Trainer(model, seed=seeds[0])
    before = trainer.generator.get_state().clone()
    evaluate(
        model, get_decoder("minimum_weight_matching", dem), shots=32, seed=seeds[2]
    )
    assert torch.equal(before, trainer.generator.get_state())
