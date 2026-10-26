"""Optional continued-study calibration and frozen physical-validation choices.

No CLI or primary sampler is exposed. The input-only command remains in
continued_assessment; device receipts and complete execution are prerequisites
for collecting the registered study.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import itertools
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from examples.qec import continued_assessment as inputs
from examples.qec import diagnose_predecoder as archive
from examples.qec.continued_assessment import CheckpointRecord, VerifiedCheckpoint
from flagquantum.qec.decoders._ai import calibration as calibration_module
from flagquantum.qec.decoders._ai import comparison as comparison_module
from flagquantum.qec.decoders._ai import predecoder as predecoder_module
from flagquantum.qec.decoders._ai.assessment import PairedCounts, paired_counts
from flagquantum.qec.decoders._ai.calibration import (
    TemperatureFit,
    ThresholdSelection,
    ThresholdTrial,
    _select_threshold_trials,
    fit_temperature,
)
from flagquantum.qec.decoders._ai.comparison import (
    GlobalDecode,
    binary_array,
    decode,
    on_device,
)
from flagquantum.qec.decoders._ai.diagnostics import checked_scores


@dataclass(frozen=True)
class CalibrationRecord:
    seed: int
    step: int
    fit: TemperatureFit
    detectors_sha256: str
    mechanisms_sha256: str


@dataclass(frozen=True)
class CalibrationManifest:
    purpose: str
    protocol_sha256: str
    checkpoints: list[CheckpointRecord]
    score_files: dict[str, str]
    records: list[CalibrationRecord]


@dataclass(frozen=True)
class NativeChoice:
    seed: int
    step: int
    mode: str
    temperature: float
    selection: ThresholdSelection


@dataclass(frozen=True)
class Choices:
    purpose: str
    protocol_sha256: str
    circuit_sha256: str
    checkpoints: list[CheckpointRecord]
    calibration_sha256: str
    validation_sha256: str
    source_sha256: dict[str, str]
    paths: list[NativeChoice]


def _identities(models: list[VerifiedCheckpoint]) -> list[CheckpointRecord]:
    if [(m.seed, m.step) for m in models] != [
        (s, h) for s in archive.SEEDS for h in (200, 2000)
    ]:
        raise ValueError("requires all six ordered checkpoints")
    for item in models:
        if (
            item.model.training
            or any(v.device.type != "cpu" for v in item.model.state_dict().values())
            or archive.state_digest(item.model.state_dict()) != item.model_state_sha256
            or item.model.dem_text != models[0].model.dem_text
        ):
            raise ValueError("checkpoint changed or is not a frozen CPU model")
    return [
        CheckpointRecord(
            seed=m.seed,
            step=m.step,
            checkpoint_sha256=m.checkpoint_sha256,
            model_state_sha256=m.model_state_sha256,
        )
        for m in models
    ]


@torch.inference_mode()
def calibrate(models: list[VerifiedCheckpoint], output: Path) -> CalibrationManifest:
    """Save each fixed DEM stream and both horizons' scores before fitting.

    Full logits are materialized only for one seed to reuse the existing fitter.
    On d9, its double-precision diagnostics require several GiB of CPU memory.
    This internal stage is not exposed by the verification-only CLI.
    """
    identities = _identities(models)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "plan.json").open("x") as stream:
        json.dump(
            {
                "protocol_sha256": archive.digest(inputs.PROTOCOL),
                "checkpoints": identities,
            },
            stream,
            indent=2,
        )
    files: dict[str, str] = {}
    records: list[CalibrationRecord] = []
    for index, seed in enumerate(archive.SEEDS):
        pair = models[2 * index : 2 * index + 2]
        generator = torch.Generator().manual_seed(seed + 14100000)
        labels: list[torch.Tensor] = []
        scores: tuple[list[torch.Tensor], list[torch.Tensor]] = ([], [])
        detector_hash, label_hash = hashlib.sha256(), hashlib.sha256()
        for chunk in range(128):
            batch = pair[0].model.sample(32, generator)
            detectors, target = batch.sample.detectors, batch.mechanisms
            values = [m.model(detectors) for m in pair]
            # Validate before persisting or passing scores to the fitter.
            for value in values:
                checked_scores(value, target)
            path = output / f"seed-{seed}-chunk-{chunk:04d}.pt"
            with path.open("xb") as stream:
                torch.save(
                    {
                        "detectors": detectors,
                        "labels": target,
                        "step200_logits": values[0],
                        "step2000_logits": values[1],
                    },
                    stream,
                )
            files[path.name] = archive.digest(path)
            detector_hash.update(archive.tensor_bytes(detectors))
            label_hash.update(archive.tensor_bytes(target))
            labels.append(target)
            for destination, value in zip(scores, values, strict=True):
                destination.append(value)
        target = torch.cat(labels)
        for item, values in zip(pair, scores, strict=True):
            records.append(
                CalibrationRecord(
                    seed,
                    item.step,
                    fit_temperature(torch.cat(values), target),
                    detector_hash.hexdigest(),
                    label_hash.hexdigest(),
                )
            )
    if _identities(models) != identities:
        raise ValueError("calibration mutated a checkpoint")
    result = CalibrationManifest(
        "continued_temperature_calibration",
        archive.digest(inputs.PROTOCOL),
        identities,
        files,
        records,
    )
    with (output / "manifest.json").open("x") as stream:
        stream.write(json.dumps(asdict(result), indent=2, allow_nan=False) + "\n")
    return result


def _temperatures(
    directory: Path, identities: list[CheckpointRecord]
) -> dict[tuple[int, int], float]:
    raw = json.loads((directory / "manifest.json").read_text())
    if (
        raw["purpose"] != "continued_temperature_calibration"
        or raw["protocol_sha256"] != archive.digest(inputs.PROTOCOL)
        or raw["checkpoints"] != identities
    ):
        raise ValueError("calibration is not bound to these inputs")
    names = [f"seed-{s}-chunk-{i:04d}.pt" for s in archive.SEEDS for i in range(128)]
    if sorted(raw["score_files"]) != sorted(names) or sorted(
        p.name for p in directory.glob("*.pt")
    ) != sorted(names):
        raise ValueError("calibration score inventory mismatch")
    for name in names:
        if archive.digest(directory / name) != raw["score_files"][name]:
            raise ValueError("calibration score hash mismatch")
    if [(r["seed"], r["step"]) for r in raw["records"]] != [
        (r["seed"], r["step"]) for r in identities
    ]:
        raise ValueError("calibration must retain all six fits")
    result = {}
    for record in raw["records"]:
        temperature = record["fit"]["temperature"]
        if type(temperature) not in (int, float) or temperature not in (
            0.25,
            0.5,
            1,
            2,
            4,
        ):
            raise ValueError("temperature outside frozen grid")
        result[(record["seed"], record["step"])] = float(temperature)
    return result


@torch.inference_mode()
def _select_paths(
    models: list[predecoder_module.Predecoder],
    temperatures: list[float],
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    global_decode: GlobalDecode,
    device: torch.device,
    shots: int,
) -> list[ThresholdSelection]:
    if (
        not models
        or len(temperatures) != 2 * len(models)
        or device.type not in ("cpu", "cuda")
    ):
        raise ValueError(
            "selection requires two modes per model and explicit CPU or CUDA"
        )
    thresholds = (0.5, 0.7, 0.9, 0.99, 1.0)
    totals = [[[0] * 5 for _ in thresholds] for _ in range(len(temperatures))]
    working = [copy.deepcopy(m).to(device).eval() for m in models]
    for detectors, truth in batches:
        for start in range(0, detectors.shape[0], 32):
            bits, target = detectors[start : start + 32], truth[start : start + 32]
            baseline = global_decode(bits)
            binary_array(baseline, shots=bits.shape[0], width=target.shape[1])
            for index, temperature in enumerate(temperatures):
                model = working[index // 2]
                for column, threshold in enumerate(thresholds):
                    prediction = decode(
                        bits,
                        on_device(
                            functools.partial(
                                model.correct,
                                threshold=threshold,
                                temperature=temperature,
                            ),
                            device,
                        ),
                        global_decode,
                    )
                    count = paired_counts(target, baseline, prediction)
                    for field, value in enumerate(asdict(count).values()):
                        totals[index][column][field] += value
    selections = []
    for counts in totals:
        trials = tuple(
            ThresholdTrial(t, PairedCounts(*c))
            for t, c in zip(thresholds, counts, strict=True)
        )
        if any(t.counts.shots != shots for t in trials):
            raise ValueError("incomplete physical validation")
        selections.append(_select_threshold_trials(trials))
    return selections


@torch.inference_mode()
def freeze_choices(
    models: list[VerifiedCheckpoint],
    calibration: Path,
    validation: Path,
    output: Path,
    *,
    circuit_sha256: str,
    calibration_sha256: str,
    global_decode: GlobalDecode,
    device: torch.device,
) -> Choices:
    """Select from shared physical shards; create a fresh, hash-bound decision file.

    This stage constructs no circuit sampler. CPU is supported for exact software
    tests; the registered physical workflow must supply its explicit CUDA device.
    """
    identities = _identities(models)
    if archive.digest(calibration / "manifest.json") != calibration_sha256:
        raise ValueError("calibration manifest hash mismatch")
    temperatures = _temperatures(calibration, identities)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("selection requires explicit CPU or CUDA")
    # Check the entire saved inventory before creating a partial output stage.
    replay = inputs.replay_validation(validation, circuit_sha256=circuit_sha256)
    first = next(replay)
    output.mkdir(parents=True, exist_ok=False)
    calibration_hash = archive.digest(calibration / "manifest.json")
    validation_hash = archive.digest(validation / "manifest.json")
    with (output / "plan.json").open("x") as stream:
        json.dump(
            {
                "calibration_sha256": calibration_hash,
                "validation_sha256": validation_hash,
            },
            stream,
        )
    variants = [
        (m, mode, 1.0 if mode == "raw" else temperatures[(m.seed, m.step)])
        for m in models
        for mode in ("raw", "calibrated")
    ]
    selections = _select_paths(
        [m.model for m in models],
        [t for _, _, t in variants],
        itertools.chain((first,), replay),
        global_decode=global_decode,
        device=device,
        shots=inputs.VALIDATION_SHOTS,
    )
    paths = [
        NativeChoice(item.seed, item.step, mode, temperature, selected)
        for (item, mode, temperature), selected in zip(
            variants, selections, strict=True
        )
    ]
    if (
        _identities(models) != identities
        or archive.digest(calibration / "manifest.json") != calibration_hash
        or archive.digest(validation / "manifest.json") != validation_hash
    ):
        raise ValueError("selection inputs changed")
    sources = [
        Path(__file__),
        Path(inputs.__file__),
        Path(calibration_module.__file__),
        Path(comparison_module.__file__),
        Path(predecoder_module.__file__),
    ]
    result = Choices(
        "continued_frozen_choices",
        archive.digest(inputs.PROTOCOL),
        circuit_sha256,
        identities,
        calibration_hash,
        validation_hash,
        {str(p.relative_to(archive.ROOT)): archive.digest(p) for p in sources},
        paths,
    )
    with (output / "choices.json").open("x") as stream:
        stream.write(json.dumps(asdict(result), indent=2, allow_nan=False) + "\n")
    with (output / "choices.sha256").open("x") as stream:
        stream.write(archive.digest(output / "choices.json") + "\n")
    return result


def verify_choices(
    directory: Path,
    *,
    expected_sha256: str,
    models: list[VerifiedCheckpoint],
    circuit_sha256: str,
) -> None:
    """Require the externally frozen hash and input identities before later stages."""
    path = directory / "choices.json"
    if archive.digest(path) != expected_sha256:
        raise ValueError("frozen choices hash mismatch")
    raw = json.loads(path.read_text())
    if (
        raw["purpose"] != "continued_frozen_choices"
        or raw["protocol_sha256"] != archive.digest(inputs.PROTOCOL)
        or raw["circuit_sha256"] != circuit_sha256
        or raw["checkpoints"] != _identities(models)
    ):
        raise ValueError("frozen choices input binding mismatch")
    for name, digest in raw["source_sha256"].items():
        if archive.digest(archive.ROOT / name) != digest:
            raise ValueError("frozen choices source changed")
