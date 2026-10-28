"""Real synthetic stage composition; all host/vendor/CUDA boundaries are fakes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.qec import compare_ising as comparison
from examples.qec import diagnose_predecoder as archive
from examples.qec import ising_layout
from examples.qec import matched_checkpoints as checkpoints
from examples.qec import matched_primary as primary
from examples.qec import matched_selection as selection
from examples.qec import matched_training as training
from examples.qec import matched_validation as validation
from examples.qec import run_continued_assessment as coordinator
from examples.qec import run_matched_assessment as run
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.decoders._ai.spacetime import SpacetimePredecoder
from flagquantum.qec.decoders._ai.training import Trainer
from flagquantum.qec.dem.model import DemError, DetectorErrorModel

pytestmark = pytest.mark.integration


class SyntheticCircuit:
    num_detectors = 4
    num_observables = 1

    def __init__(self) -> None:
        self.compiled: list[int] = []
        self.calls: list[tuple[int, int]] = []

    def __str__(self) -> str:
        return "matched end-to-end synthetic circuit, never a physical source"

    def compile_detector_sampler(self, *, seed: int) -> Any:
        self.compiled.append(seed)
        owner = self
        generator = torch.Generator().manual_seed(seed)

        class Sampler:
            def sample(self, shots: int, *, separate_observables: bool) -> Any:
                assert separate_observables
                owner.calls.append((seed, shots))
                bits = torch.randint(
                    2, (shots, 4), generator=generator, dtype=torch.int8
                )
                return bits, (bits[:, 0] ^ bits[:, 2]).reshape(-1, 1)

        return Sampler()


class SyntheticPipeline(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        return torch.cat((torch.zeros(len(bits), 1, dtype=bits.dtype), bits), dim=1)


def test_actual_stages_train_to_final_and_complete_shared_stream_without_stage_mocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tf32_disabled: None,
) -> None:
    """Keep all six actual step2000 identities; reduce only synthetic sample sizes."""
    circuit = SyntheticCircuit()
    dem = DetectorErrorModel(
        4, 1, (DemError(0.1, (0, 2, 3), (0,)), DemError(0.04, (1,), ()))
    )
    template = SpacetimePredecoder.from_dem(
        dem, grid_shape=(2, 2, 2), sites=(0, 2, 5, 7), kinds=(1, 2, 1, 2)
    )
    inputs = dict(
        layout_sha256="a" * 64,
        dem_sha256=hashlib.sha256(template.dem_text.encode()).hexdigest(),
        circuit_sha256=hashlib.sha256(str(circuit).encode()).hexdigest(),
    )
    monkeypatch.setattr(ising_layout, "build_model", lambda _: (template, inputs))
    monkeypatch.setattr(training, "TRAIN_BATCH", 2)
    monkeypatch.setattr(training, "SCORE_SHOTS", 2)
    monkeypatch.setattr(training, "SCORE_CHUNK", 2)
    monkeypatch.setattr(selection, "CALIBRATION_SHOTS", 2)
    monkeypatch.setattr(selection, "CALIBRATION_CHUNK", 2)
    monkeypatch.setattr(validation, "VALIDATION_SHOTS", 8)
    monkeypatch.setattr(validation, "CHUNK", 4)
    monkeypatch.setattr(primary, "SHOTS", 64)
    monkeypatch.setattr(primary, "CHUNK", 32)
    # No stage, checkpoint, decision, path, receipt parser or statistics is mocked.
    # Explicit boundary fakes below emulate vendor source, weights, host and CUDA.
    pipeline = SyntheticPipeline()
    weights = tmp_path / "fake-weights"
    weights.write_bytes(b"synthetic weight boundary, never official weights")
    monkeypatch.setattr(comparison, "FAST_SHA256", archive.digest(weights))
    monkeypatch.setattr(
        comparison, "source_digest", lambda _: comparison.ISING_SOURCE_SHA256
    )
    monkeypatch.setattr(
        comparison,
        "reference",
        lambda *a, **k: comparison.Reference(circuit, pipeline, {}, {}),
    )
    monkeypatch.setattr(comparison, "training_dem", lambda _: dem)
    monkeypatch.setattr(
        comparison,
        "global_decoder",
        lambda _: lambda b: (b[:, 0].to(torch.int8) ^ b[:, 1].to(torch.int8)).reshape(
            -1, 1
        ),
    )
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "manifest.json").write_text(
        json.dumps({"torch": str(torch.__version__)})
    )
    (metadata / "results.json").write_text(
        json.dumps({"circuit_sha256": inputs["circuit_sha256"]})
    )
    monkeypatch.setattr(archive, "ARCHIVE", metadata)
    inspect = tmp_path / "inspect.json"
    inspect.write_text("explicit inspected-container boundary fake")
    lock = tmp_path / "requirements.lock"
    lock.write_text("synthetic dependency boundary")
    options = run.Options(
        tmp_path,
        weights,
        inspect,
        archive.digest(inspect),
        "a800-172",
        tmp_path / "study",
        lock,
    )
    host = ["a800-172"]

    def hardware(image: str) -> dict[str, Any]:
        suffix = "1" if host[0] == "a800-172" else "2"
        return dict(
            device="cuda:0",
            name="A800 software fake",
            image_id=image,
            tf32=False,
            autocast=False,
            deterministic_algorithms=True,
            nvidia_smi_uuid_name_driver=f"GPU-11111111-1111-1111-1111-{suffix * 12},A800,software-fake",
        )

    monkeypatch.setattr(comparison, "cuda_metadata", hardware)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    original_module_to, original_tensor_to = torch.nn.Module.to, torch.Tensor.to

    def cpu_args(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if (
            args
            and isinstance(args[0], (str, torch.device))
            and str(args[0]).startswith("cuda")
        ):
            args = (torch.device("cpu"), *args[1:])
        if "device" in kwargs and str(kwargs["device"]).startswith("cuda"):
            kwargs = {**kwargs, "device": torch.device("cpu")}
        return args, kwargs

    def module_to(self: torch.nn.Module, *a: Any, **k: Any) -> Any:
        a, k = cpu_args(a, k)
        return original_module_to(self, *a, **k)

    def tensor_to(self: torch.Tensor, *a: Any, **k: Any) -> Any:
        a, k = cpu_args(a, k)
        return original_tensor_to(self, *a, **k)

    monkeypatch.setattr(torch.nn.Module, "to", module_to)
    monkeypatch.setattr(torch.Tensor, "to", tensor_to)

    def preflight(opts: run.Options) -> dict[str, Any]:
        host[0] = opts.host
        return dict(
            protocols={p.name: archive.digest(p) for p in checkpoints.PROTOCOLS},
            source_files=run._fingerprints(),
            metadata_sha256={p.name: archive.digest(p) for p in metadata.iterdir()},
            dependency_lock_sha256=archive.digest(lock),
            fast_weights_sha256=archive.digest(weights),
            ising_source_sha256=comparison.ISING_SOURCE_SHA256,
            inputs=inputs,
            pipeline_state_sha256=archive.state_digest(pipeline.state_dict()),
        )

    monkeypatch.setattr(run, "preflight", preflight)
    monkeypatch.setattr(coordinator, "_container_scope", lambda *a, **k: {})
    draws: Counter[tuple[int, int]] = Counter()
    updates: Counter[str] = Counter()
    sample, train = Predecoder.sample, Trainer.train

    def observed_sample(
        self: Predecoder, shots: int, generator: torch.Generator
    ) -> Any:
        draws[(generator.initial_seed(), shots)] += 1
        return sample(self, shots, generator)

    def observed_train(self: Trainer, steps: int, **kwargs: Any) -> Any:
        updates[type(self.model).__name__] += steps
        return train(self, steps, **kwargs)

    monkeypatch.setattr(Predecoder, "sample", observed_sample)
    monkeypatch.setattr(Trainer, "train", observed_train)
    sha = run.prepare(options)
    peer = tmp_path / "peer"
    peer_sha = run.peer(replace(options, host="a800-171"), sha, peer)
    run.finish(options, sha, peer / "device/receipt.json", peer_sha)
    assert updates == {"SpacetimePredecoder": 6000, "Predecoder": 6000}
    assert draws == Counter(
        {
            **{(s + 18100000, 2): 4000 for s in (11, 22, 33)},
            **{(s + 19100000, 2): 1 for s in (11, 22, 33)},
            **{(s + 20100000, 2): 1 for s in (11, 22, 33)},
        }
    )
    assert circuit.compiled == [21000011, 23000011, 22000011]
    assert circuit.calls == [
        (21000011, 4),
        (21000011, 4),
        (23000011, 64),
        (22000011, 32),
        (22000011, 32),
    ]
    assert (options.study / "complete.json").exists()
    final_draws, final_updates = draws.copy(), updates.copy()
    models, _ = validation.restore_choices(
        tmp_path,
        options.study / "initial",
        options.study / "trained",
        options.study / "calibration",
        options.study / "choices",
        expected_sha256=archive.digest(options.study / "choices/choices.json"),
    )
    assert [(s, a, t.step) for s, a, t in models] == [
        (s, a, 2000) for s in (11, 22, 33) for a in ("cnn", "mlp")
    ]
    raw = json.loads((options.study / "summary.json").read_text())
    chunks = list(
        primary.replay_primary(
            options.study / "primary",
            expected_manifest_sha256=archive.digest(
                options.study / "primary/manifest.json"
            ),
        )
    )
    truth = torch.cat([t for t, _ in chunks]).tolist()
    for row in raw["comparisons"]:
        ref = torch.cat([p[row["reference"]] for _, p in chunks]).tolist()
        cand = torch.cat([p[row["candidate"]] for _, p in chunks]).tolist()
        pairs = [
            (
                any(x != y for x, y in zip(t, r, strict=True)),
                any(x != y for x, y in zip(t, c, strict=True)),
            )
            for t, r, c in zip(truth, ref, cand, strict=True)
        ]
        assert row["counts"] == dict(
            shots=64,
            baseline_failures=sum(r for r, c in pairs),
            hybrid_failures=sum(c for r, c in pairs),
            wins=sum(r and not c for r, c in pairs),
            losses=sum(c and not r for r, c in pairs),
        )
    assert len(raw["comparisons"]) == 32
    assert circuit.compiled == [21000011, 23000011, 22000011]
    with pytest.raises(FileExistsError):
        run.prepare(options)
    with pytest.raises(FileExistsError):
        run.finish(options, sha, peer / "device/receipt.json", peer_sha)
    assert draws == final_draws and updates == final_updates
