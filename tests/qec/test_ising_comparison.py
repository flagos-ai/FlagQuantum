"""Shared global decoding, complete hyperedge corrections and optional upstream smoke."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from flagquantum.qec.decoders._ai.comparison import decode, measure
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.decoders._ai.training import Trainer
from flagquantum.qec.dem.model import DetectorErrorModel

pytestmark = pytest.mark.integration


def hyperedge_model() -> Predecoder:
    dem = DetectorErrorModel.from_stim_text(
        "error(0.1) D0 D1 D2 L0\ndetector D0\ndetector D1\ndetector D2\n"
    )
    return Predecoder(dem, allow_hyperedges=True)


def test_hyperedge_is_one_label_and_one_complete_correction(tmp_path: Path) -> None:
    model = hyperedge_model()
    batch = model.sample(128, torch.Generator().manual_seed(19))
    assert batch.mechanisms.shape == (128, 1)
    assert torch.equal(batch.sample.detectors[:, 0], batch.sample.observables[:, 0])
    assert torch.equal(batch.sample.detectors[:, 0], batch.sample.detectors[:, 2])
    assert int(batch.mechanisms.sum()) > 0
    trainer = Trainer(model)
    trainer.train(2, batch_size=16)
    trainer.save(tmp_path / "hyperedge.pt")
    restored = hyperedge_model()
    Trainer(restored).restore(tmp_path / "hyperedge.pt")
    assert torch.equal(
        model.forward(batch.sample.detectors), restored.forward(batch.sample.detectors)
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        head = model.network[-1]
        assert isinstance(head, nn.Linear)
        assert head.bias is not None
        head.bias.fill_(10)
    residual, frame = model.correct(torch.ones(2, 3), threshold=0.9)
    assert torch.equal(residual, torch.zeros(2, 3, dtype=torch.int8))
    assert torch.equal(frame, torch.ones(2, 1, dtype=torch.int8))


def test_hyperedge_opt_in_does_not_admit_invisible_mechanisms() -> None:
    with pytest.raises(ValueError, match="visible"):
        Predecoder(
            DetectorErrorModel.from_stim_text("error(0.1) L0\ndetector D0\n"),
            allow_hyperedges=True,
        )


def test_shared_decoder_observes_residual_and_composes_frame() -> None:
    seen: list[torch.Tensor] = []

    def correct(bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return 1 - bits, torch.tensor([[1], [0]])

    def global_decode(bits: torch.Tensor) -> torch.Tensor:
        seen.append(bits.clone())
        return bits[:, :1]

    detectors = torch.tensor([[1, 0], [0, 1]])
    assert decode(detectors, correct, global_decode).tolist() == [[1], [1]]
    assert torch.equal(seen[0], 1 - detectors)
    timing = measure(detectors, correct, global_decode, warmup=1, repetitions=3)
    assert timing.batch_size == 2
    assert 0 < timing.p50_block_us <= timing.p95_block_us <= timing.p99_block_us
    assert timing.shots_per_second > 0


def test_bad_external_residual_is_rejected_before_global_decode() -> None:
    def bad(bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return bits + 2, torch.zeros((bits.shape[0], 1))

    def global_decode(bits: torch.Tensor) -> torch.Tensor:
        raise AssertionError("invalid external output reached matcher")

    with pytest.raises(ValueError, match="binary"):
        decode(torch.zeros(1, 2), bad, global_decode)


def test_non_smoke_requires_real_checkpoint_before_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.qec import compare_ising as study

    with pytest.raises(ValueError, match="pinned"):
        study.reference(tmp_path, None, study.Config(False))
    monkeypatch.setattr(study, "source_digest", lambda root: study.ISING_SOURCE_SHA256)
    with pytest.raises(ValueError, match="authorized"):
        study.reference(tmp_path, None, study.Config(False))


def test_external_nan_logits_are_rejected_before_thresholding() -> None:
    from examples.qec.compare_ising import check_logits

    with pytest.raises(ValueError, match="nonfinite"):
        check_logits(nn.Identity(), (), torch.tensor([float("nan")]))


class NoLocalCorrection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(-100.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.value.expand(inputs.shape[0], 4, *inputs.shape[2:])


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_real_upstream_noop_changes_only_the_declared_boundary_mask(
    basis: str,
) -> None:
    source = os.environ.get("FLAGQUANTUM_ISING_SOURCE")
    if source is None:
        pytest.skip("requires the optional pinned upstream checkout")
    from examples.qec import compare_ising as study

    config = replace(study.Config(True), basis=basis)
    ref = study.reference(Path(source), None, config)
    ref.pipeline.model = NoLocalCorrection()
    bits, _ = study.sample(ref.circuit, 128, 81)
    with torch.inference_mode():
        output = ref.pipeline(bits)
    expected = study.boundary_only(bits, config)
    assert not torch.equal(expected, bits)
    assert torch.equal(output[:, 1:].to(torch.int8), expected)
    assert not bool(output[:, 0].any())


def test_real_safetensors_loader_preserves_parameters_and_records_file_hash(
    tmp_path: Path,
) -> None:
    source = os.environ.get("FLAGQUANTUM_ISING_SOURCE")
    if source is None:
        pytest.skip("requires the optional pinned upstream checkout")
    from examples.qec import compare_ising as study

    ref = study.reference(Path(source), None, study.Config(True))
    exports = sys.modules["export.safetensors_utils"]
    checkpoint = tmp_path / "untrained-loader-fixture.safetensors"
    exports.save_safetensors(ref.pipeline.model, str(checkpoint), model_id=1)
    loaded = study.reference(Path(source), checkpoint, study.Config(False))
    assert loaded.weights["sha256"] == study.digest(checkpoint)
    assert loaded.weights["status"] == "external_checkpoint_training_history_unverified"
    for first, second in zip(
        ref.pipeline.parameters(), loaded.pipeline.parameters(), strict=True
    ):
        assert torch.equal(first, second)


def test_real_upstream_smoke_is_explicitly_not_a_ranking(tmp_path: Path) -> None:
    source = os.environ.get("FLAGQUANTUM_ISING_SOURCE")
    if source is None:
        pytest.skip(
            "set FLAGQUANTUM_ISING_SOURCE to the pinned optional upstream checkout"
        )
    output = tmp_path / "comparison"
    command = [
        sys.executable,
        "examples/qec/compare_ising.py",
        "--ising-source",
        source,
        "--output-directory",
        str(output),
        "--smoke",
    ]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=180
    )
    summary = json.loads(completed.stdout.splitlines()[-1])
    assert summary["scientific_ranking_allowed"] is False
    assert summary["distinct_test_shots"] == 32
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["weights"]["status"] == "untrained_smoke_only"
    assert manifest["training_dem_approximate_disjoint_errors"] is True
    results = json.loads((output / "results.json").read_text())
    assert len(results["records"]) == 4
    for name in summary["paths"]:
        timing = json.loads((output / f"timing-{name}.json").read_text())
        assert [item["batch_size"] for item in timing["timings"]] == [1, 32]
        assert timing["isolated_process_peak_rss_bytes"] > 0
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert repeated.returncode != 0
    assert "FileExistsError" in repeated.stderr
