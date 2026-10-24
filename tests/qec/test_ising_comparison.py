"""Shared global decoding, complete hyperedge corrections and optional upstream smoke."""

from __future__ import annotations

import argparse
import copy
import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from flagquantum.qec.decoders._ai.comparison import (
    decode,
    measure,
    on_device,
    require_parity,
)
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.decoders._ai.training import Trainer
from flagquantum.qec.dem.model import DetectorErrorModel

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("counter", ["VmHWM: 1234 kB", "VmRSS: 7 kB\nVmHWM: 1234 kB"])
def test_linux_rss_excludes_inherited_pre_exec_peak(
    monkeypatch: pytest.MonkeyPatch,
    counter: str,
) -> None:
    from examples.qec.compare_ising import peak_rss_bytes

    def status(path: Path, encoding: str = "utf-8") -> str:
        assert str(path) == "/proc/self/status"
        return counter

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(Path, "read_text", status)
    monkeypatch.setattr(
        resource, "getrusage", lambda who: SimpleNamespace(ru_maxrss=10**9)
    )
    assert peak_rss_bytes() == 1234 * 1024


def test_missing_linux_memory_counter_is_not_a_false_rss_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.qec.compare_ising import peak_rss_bytes

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(Path, "read_text", lambda path, encoding: "VmRSS: 7 kB")
    with pytest.raises(ValueError, match="requires VmHWM"):
        peak_rss_bytes()


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


def test_external_dem_merges_only_identical_complete_signatures() -> None:
    from examples.qec.compare_ising import training_dem

    stim = pytest.importorskip("stim")

    class Circuit:
        def detector_error_model(
            self, *, decompose_errors: bool, approximate_disjoint_errors: bool
        ) -> Any:
            assert not decompose_errors and approximate_disjoint_errors
            return stim.DetectorErrorModel(
                "error(0.1) D0 D1 D2 L0\nerror(0.2) D0 D1 D2 L0\n"
                "error(0.3) D0 D1 D2\n"
            )

    dem = training_dem(Circuit())
    dem.require_unique_mechanisms()
    assert dem.num_detectors == 3 and dem.num_observables == 1
    assert len(dem.errors) == 2
    by_logical = {error.observables: error for error in dem.errors}
    assert by_logical[(0,)].probability == pytest.approx(0.1 * 0.8 + 0.9 * 0.2)
    assert by_logical[()].probability == 0.3
    assert all(error.detectors == (0, 1, 2) for error in dem.errors)


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


def test_parity_refuses_cancelling_residual_and_logical_frame_errors() -> None:
    bits = torch.tensor([[0], [1]], dtype=torch.int8)

    def reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, torch.zeros_like(x)

    def broken(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return 1 - x, torch.ones_like(x)

    def matching(x: torch.Tensor) -> torch.Tensor:
        return x

    assert torch.equal(
        decode(bits, reference, matching), decode(bits, broken, matching)
    )
    with pytest.raises(ValueError, match="residual or logical-frame"):
        require_parity(
            bits, reference, on_device(broken, torch.device("cpu")), matching
        )


def test_cuda_requests_refuse_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from examples.qec.compare_ising import cuda_metadata

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="requires CUDA"):
        cuda_metadata(None)
    with pytest.raises(ValueError, match="CPU or CUDA"):
        on_device(lambda x: (x, x[:, :1]), torch.device("meta"))


def test_cuda_timing_fences_bracket_the_complete_cpu_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU contract check of synchronization order, not GPU timing evidence."""
    events: list[str] = []
    clock = iter(range(0, 6000, 1000))

    def synchronize() -> None:
        events.append("sync")

    def now() -> int:
        events.append("clock")
        return next(clock)

    def matching(x: torch.Tensor) -> torch.Tensor:
        events.append("matching")
        return x[:, :1]

    monkeypatch.setattr(time, "perf_counter_ns", now)
    bits = torch.zeros(2, 3, dtype=torch.int8)
    timing = measure(
        bits,
        lambda x: (x, x[:, :1]),
        matching,
        warmup=1,
        repetitions=3,
        synchronize=synchronize,
    )
    assert events[2:] == ["sync", "clock", "matching", "sync", "clock"] * 3
    assert timing.p50_block_us == timing.p99_block_us == 1.0


def test_a800_parity_failure_aborts_before_primary_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.qec import compare_ising as study

    model = hyperedge_model()
    output = tmp_path / "aborted"
    monkeypatch.setattr(
        study,
        "training_dem",
        lambda circuit: DetectorErrorModel.from_stim_text(model.dem_text),
    )
    monkeypatch.setattr(study, "global_decoder", lambda circuit: lambda x: x[:, :1])
    monkeypatch.setattr(study, "choices", lambda *args: {})

    def refuse(*args: object) -> None:
        raise ValueError("declared probe parity failed")

    def no_primary(*args: object) -> None:
        raise AssertionError("primary sampler reached before parity passed")

    monkeypatch.setattr(study, "device_parity", refuse)
    monkeypatch.setattr(study, "sample", no_primary)
    args = argparse.Namespace(
        output_directory=output, hardware={}, ising_source=tmp_path
    )
    ref = study.Reference(None, nn.Identity(), {}, {})
    with pytest.raises(ValueError, match="probe parity failed"):
        study.run(args, study.Config(True, distance=9, rounds=9, a800=True), ref)
    assert (output / "choices.json").is_file()
    assert not (output / "test-shots.pt").exists()


def test_a800_smoke_also_requires_the_real_pinned_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.qec import compare_ising as study

    checkpoint = tmp_path / "untrained.safetensors"
    checkpoint.write_bytes(b"not the official checkpoint")
    monkeypatch.setattr(study, "source_digest", lambda root: study.ISING_SOURCE_SHA256)
    with pytest.raises(ValueError, match="pinned official Fast"):
        study.reference(tmp_path, checkpoint, study.Config(True, a800=True))


@pytest.mark.gpu
def test_real_cuda_complete_hyperedge_correction_and_timing() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires an actual CUDA device")
    model = hyperedge_model().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        head = model.network[-1]
        assert isinstance(head, nn.Linear) and head.bias is not None
        head.bias.fill_(10)
    device = torch.device("cuda:0")
    accelerated = copy.deepcopy(model).to(device)
    transferred = on_device(accelerated.correct, device)
    bits = torch.tensor([[0, 0, 0], [1, 1, 1], [1, 0, 1]], dtype=torch.int8)

    def matching(x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]

    require_parity(bits, model.correct, transferred, matching)
    timing = measure(
        bits,
        transferred,
        matching,
        warmup=1,
        repetitions=3,
        synchronize=lambda: torch.cuda.synchronize(device),
    )
    assert timing.p99_block_us >= timing.p50_block_us > 0
    assert torch.cuda.max_memory_allocated(device) > 0


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
