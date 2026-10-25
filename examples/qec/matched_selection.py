"""Internal matched-model calibration; no CLI or formal launch authorization.

Physical validation, frozen choices and actual device receipts remain separate
acceptance gates. Synthetic tests never authorize formal samples or updates.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from examples.qec import diagnose_predecoder as archive
from examples.qec import matched_checkpoints as checkpoints
from examples.qec import matched_training as training
from flagquantum.qec.decoders._ai import calibration
from flagquantum.qec.decoders._ai.diagnostics import checked_scores
from flagquantum.qec.decoders._ai.training import Trainer

CALIBRATION_SHOTS = 4096
CALIBRATION_CHUNK = 32
CALIBRATION_SEED_OFFSET = 20100000


def _sources() -> dict[str, str]:
    result = training._sources()
    for path in (Path(__file__), Path(calibration.__file__)):
        result[str(path.relative_to(archive.ROOT))] = archive.digest(path)
    return result


def _calibrate_pair(
    pair: list[tuple[int, str, Trainer]], output: Path
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    seed = pair[0][0]
    generator = torch.Generator().manual_seed(seed + CALIBRATION_SEED_OFFSET)
    before = archive.state_digest({"generator": generator.get_state()})
    targets: list[torch.Tensor] = []
    scores: tuple[list[torch.Tensor], list[torch.Tensor]] = ([], [])
    files: dict[str, str] = {}
    for chunk in range(CALIBRATION_SHOTS // CALIBRATION_CHUNK):
        batch = pair[0][2].model.sample(CALIBRATION_CHUNK, generator)
        hashes = training._hashes(batch, pair[0][2].model, CALIBRATION_CHUNK)
        values = [trainer.model(batch.sample.detectors) for _, _, trainer in pair]
        for value in values:
            checked_scores(value, batch.mechanisms)
        path = output / f"seed-{seed}-chunk-{chunk:04d}.pt"
        with path.open("xb") as stream:
            torch.save(
                {
                    "detectors": batch.sample.detectors,
                    "labels": batch.mechanisms,
                    "truth": batch.sample.observables,
                    "cnn_logits": values[0],
                    "mlp_logits": values[1],
                    "batch_sha256": hashes,
                },
                stream,
            )
        files[path.name] = archive.digest(path)
        targets.append(batch.mechanisms)
        for destination, value in zip(scores, values, strict=True):
            destination.append(value)
    labels = torch.cat(targets)
    after = archive.state_digest({"generator": generator.get_state()})
    records: list[dict[str, Any]] = []
    # Materialize only this seed's two score arrays. The existing double-precision
    # fitter can require several GiB at d9; do not retain all three seeds here.
    for (_, architecture, trainer), values in zip(pair, scores, strict=True):
        fit = calibration.fit_temperature(torch.cat(values), labels)
        records.append(
            {
                "seed": seed,
                "architecture": architecture,
                "step": trainer.step,
                "fit": asdict(fit),
                "sample_seed": seed + CALIBRATION_SEED_OFFSET,
                "generator_before_sha256": before,
                "generator_after_sha256": after,
                "shared_score_files": files.copy(),
            }
        )
    return files, records


@torch.inference_mode()
def calibrate(
    source: Path,
    initial: Path,
    trained: Path,
    output: Path,
    *,
    training_sha256: str,
) -> dict[str, Any]:
    """Fit all six final models on one saved independent stream per seed.

    This internal stage is not a launch entry. Complete workflow acceptance,
    trusted locks and the total deadline must precede formal invocation.
    """
    if output.exists():
        raise FileExistsError(output)
    if (
        torch.get_num_threads() != 1
        or type(CALIBRATION_SHOTS) is not int
        or type(CALIBRATION_CHUNK) is not int
        or CALIBRATION_SHOTS <= 0
        or not 0 < CALIBRATION_CHUNK <= 32
        or CALIBRATION_SHOTS % CALIBRATION_CHUNK
    ):
        raise ValueError("requires one thread and positive bounded calibration chunks")
    models, training_files, inputs = training.restore_final(
        source, initial, trained, training_sha256=training_sha256
    )
    initial_files = {
        str(p.relative_to(initial)): archive.digest(p)
        for p in initial.rglob("*")
        if p.is_file()
    }
    states = [training._state(trainer) for _, _, trainer in models]
    sources = _sources()
    protocols = {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS}
    plan = {
        "purpose": "matched_final_temperature_calibration",
        "training_sha256": training_sha256,
        "training_files": training_files,
        "initial_files": initial_files,
        "inputs": inputs,
        "protocol_sha256": protocols,
        "source_sha256": sources,
        "models": [
            {
                "seed": seed,
                "architecture": arch,
                "step": trainer.step,
                "training_state_sha256": state,
            }
            for (seed, arch, trainer), state in zip(models, states, strict=True)
        ],
        "shots_per_seed": CALIBRATION_SHOTS,
        "chunk_size": CALIBRATION_CHUNK,
        "seed_offset": CALIBRATION_SEED_OFFSET,
        "temperature_grid": [0.25, 0.5, 1.0, 2.0, 4.0],
    }
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output / "plan.json"
    with plan_path.open("x") as stream:
        stream.write(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    files = {"plan.json": archive.digest(plan_path)}
    records: list[dict[str, Any]] = []
    for index in range(3):
        captured, fitted = _calibrate_pair(models[2 * index : 2 * index + 2], output)
        files.update(captured)
        records.extend(fitted)
        if any(
            training._state(trainer) != state or trainer.model.training
            for (_, _, trainer), state in zip(models, states, strict=True)
        ):
            raise ValueError("calibration mutated a model, Adam or training generator")
    for directory, frozen in (
        (initial, initial_files),
        (trained, training_files),
        (output, files),
    ):
        if sorted(
            str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()
        ) != sorted(frozen) or any(
            archive.digest(directory / name) != digest
            for name, digest in frozen.items()
        ):
            raise ValueError("calibration input/output inventory or hash changed")
    if (
        _sources() != sources
        or {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS} != protocols
    ):
        raise ValueError("calibration source or prospective documents changed")
    result = {
        **plan,
        "score_files": files,
        "records": records,
        "independent_calibration_draws": 3 * CALIBRATION_SHOTS,
        "optimizer_updates": 0,
        "scientific_benefit_claim_allowed": False,
    }
    with (output / "manifest.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
