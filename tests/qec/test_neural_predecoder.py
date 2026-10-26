"""Behavioral tests for the private M1 predecoder."""

from pathlib import Path

import pytest
import torch

from flagquantum.qec import DetectorErrorModel, get_decoder
from flagquantum.qec.decoders._ai.assessment import evaluate
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.decoders._ai.training import Trainer

pytestmark = pytest.mark.integration


def model() -> Predecoder:
    return Predecoder(
        DetectorErrorModel.from_stim_text(
            "error(0.1) D0 L0\nerror(0.15) D1\ndetector D0\ndetector D1\nlogical_observable L0\n"
        )
    )


def test_logical_contribution_and_residual_are_both_applied() -> None:
    net = model()
    for parameter in net.parameters():
        parameter.data.zero_()
    # Force every mechanism to fire, including the logical one. A missing Lc
    # produces the opposite answer even though the residual is decoded correctly.
    net.network[-1].bias.data.fill_(10)
    bits = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    decoder = get_decoder(
        "minimum_weight_matching", DetectorErrorModel.from_stim_text(net.dem_text)
    )
    assert torch.equal(net.decode(bits, decoder), bits[:, :1].to(torch.int8))
    assert torch.equal(
        net.decode(bits, decoder, enabled=False), bits[:, :1].to(torch.int8)
    )


def test_sampling_matches_latent_parity_and_marginals() -> None:
    net = model()
    batch = net.sample(10000, torch.Generator().manual_seed(5))
    assert torch.equal(batch.sample.detectors.float(), batch.mechanisms)
    assert torch.equal(batch.sample.observables.float(), batch.mechanisms[:, :1])
    assert torch.allclose(
        batch.mechanisms.mean(0), torch.tensor([0.1, 0.15]), atol=0.02
    )


def test_resume_restores_optimizer_and_next_batches(tmp_path: Path) -> None:
    torch.manual_seed(4)
    trainer = Trainer(model(), seed=3)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    trainer.train(3)
    assert any(
        not torch.equal(a, b)
        for a, b in zip(before, trainer.model.parameters(), strict=True)
    )
    checkpoint = tmp_path / "training.pt"
    trainer.save(checkpoint)
    expected = trainer.train(2)
    resumed = Trainer(model(), seed=99)
    resumed.restore(checkpoint)
    assert resumed.train(2) == expected
    assert resumed.step == trainer.step
    for a, b in zip(
        resumed.model.parameters(), trainer.model.parameters(), strict=True
    ):
        assert torch.equal(a, b)
    decoder = get_decoder(
        "minimum_weight_matching",
        DetectorErrorModel.from_stim_text(resumed.model.dem_text),
    )
    assert (
        evaluate(resumed.model, decoder, shots=19, seed=100, batch_size=7)["shots"]
        == 19
    )


def test_rejects_bad_inputs_and_nonfinite_output() -> None:
    net = model()
    with pytest.raises(ValueError, match="binary"):
        net(torch.tensor([[2, 0]]))
    with pytest.raises(ValueError, match="shape"):
        net(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="positive"):
        net.sample(True, torch.Generator())
    decoder = get_decoder(
        "minimum_weight_matching", DetectorErrorModel.from_stim_text(net.dem_text)
    )
    net.network[-1].bias.data.fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        net.decode(torch.zeros(1, 2), decoder)


def test_checkpoint_rejects_different_dem(tmp_path: Path) -> None:
    checkpoint = tmp_path / "training.pt"
    Trainer(model()).save(checkpoint)
    other = Predecoder(
        DetectorErrorModel.from_stim_text("error(0.2) D0\ndetector D0\n")
    )
    with pytest.raises(ValueError, match="DEM"):
        Trainer(other).restore(checkpoint)


def test_rejects_hyperedge() -> None:
    with pytest.raises(ValueError, match="graphlike"):
        Predecoder(
            DetectorErrorModel.from_stim_text(
                "error(0.1) D0 D1 D2\ndetector D0\ndetector D1\ndetector D2\n"
            )
        )


def test_overlapping_mechanisms_update_residual_before_backend() -> None:
    from dataclasses import dataclass

    @dataclass
    class Answer:
        observables: tuple[int, ...] = (0,)

    class Backend:
        def __init__(self) -> None:
            self.seen: list[list[int]] = []

        def decode(self, detection_events: object) -> Answer:
            self.seen.append(list(detection_events))
            return Answer()

    net = Predecoder(
        DetectorErrorModel.from_stim_text(
            "error(0.1) D0 D1 L0\nerror(0.2) D1\ndetector D0\ndetector D1\nlogical_observable L0\n"
        )
    )
    for parameter in net.parameters():
        parameter.data.zero_()
    net.network[-1].bias.data.fill_(10)
    backend = Backend()
    prediction = net.decode(torch.tensor([[0, 0], [1, 1]]), backend)
    assert backend.seen == [[0], [1]]
    # Local Lc=1 XOR backend=1 gives zero; XOR of shared D1 cancels too.
    assert prediction.tolist() == [[0], [0]]
