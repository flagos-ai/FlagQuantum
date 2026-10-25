"""Internal shared physical validation and frozen matched-model thresholds.

No CLI or primary sampler is exposed. Complete trusted execution, source CI and
matching preflight remain prerequisites for any formal invocation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from examples.qec import continued_assessment as shards
from examples.qec import continued_selection as selection
from examples.qec import diagnose_predecoder as archive
from examples.qec import matched_checkpoints as checkpoints
from examples.qec import matched_selection as temperature_stage
from examples.qec import matched_training as training
from flagquantum.qec.decoders._ai.assessment import PairedCounts
from flagquantum.qec.decoders._ai.calibration import (
    ThresholdTrial,
    _select_threshold_trials,
)
from flagquantum.qec.decoders._ai.comparison import GlobalDecode
from flagquantum.qec.decoders._ai.training import Trainer

VALIDATION_SEED = 21000011
VALIDATION_SHOTS = 131072
CHUNK = 4096


def _files(directory: Path) -> dict[str, str]:
    return {
        str(p.relative_to(directory)): archive.digest(p)
        for p in directory.rglob("*")
        if p.is_file()
    }


def _sources() -> dict[str, str]:
    result = temperature_stage._sources()
    for path in (Path(__file__), Path(shards.__file__), Path(selection.__file__)):
        result[str(path.relative_to(archive.ROOT))] = archive.digest(path)
    return result


def _calibration(
    directory: Path,
    *,
    expected_sha256: str,
    training_sha256: str,
    models: list[dict[str, Any]],
    training_files: dict[str, str],
    initial_files: dict[str, str],
    inputs: dict[str, Any],
) -> tuple[list[float], dict[str, str]]:
    checkpoints._sha256(expected_sha256)
    if archive.digest(directory / "manifest.json") != expected_sha256:
        raise ValueError("external calibration manifest hash mismatch")
    raw = json.loads((directory / "manifest.json").read_text())
    protocols = {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS}
    if (
        raw["purpose"] != "matched_final_temperature_calibration"
        or raw["training_sha256"] != training_sha256
        or raw["training_files"] != training_files
        or raw["initial_files"] != initial_files
        or raw["models"] != models
        or raw["inputs"] != inputs
        or raw["protocol_sha256"] != protocols
        or raw["source_sha256"] != temperature_stage._sources()
        or raw["shots_per_seed"] != temperature_stage.CALIBRATION_SHOTS
        or raw["chunk_size"] != temperature_stage.CALIBRATION_CHUNK
        or raw["seed_offset"] != 20100000
        or raw["temperature_grid"] != [0.25, 0.5, 1.0, 2.0, 4.0]
        or raw["independent_calibration_draws"]
        != 3 * temperature_stage.CALIBRATION_SHOTS
        or raw["optimizer_updates"] != 0
    ):
        raise ValueError("calibration identity, recipe or training binding mismatch")
    names = [
        f"seed-{s}-chunk-{i:04d}.pt"
        for s in (11, 22, 33)
        for i in range(
            temperature_stage.CALIBRATION_SHOTS // temperature_stage.CALIBRATION_CHUNK
        )
    ]
    if sorted(raw["score_files"]) != sorted(["plan.json", *names]):
        raise ValueError("calibration score inventory mismatch")
    if _files(directory) != {**raw["score_files"], "manifest.json": expected_sha256}:
        raise ValueError("calibration file inventory or content changed")
    if len(raw["records"]) != 6:
        raise ValueError("requires every architecture/seed calibration fit")
    temperatures = []
    for model, record in zip(models, raw["records"], strict=True):
        if any(record[key] != model[key] for key in ("seed", "architecture", "step")):
            raise ValueError("calibration records are not in original model order")
        own = {
            name: raw["score_files"][name]
            for name in names
            if name.startswith(f"seed-{model['seed']}-")
        }
        fit = record["fit"]
        if (
            record["sample_seed"] != model["seed"] + 20100000
            or record["shared_score_files"] != own
            or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in fit.values()
            )
            or type(fit["temperature"]) not in (int, float)
            or fit["temperature"] not in (0.25, 0.5, 1, 2, 4)
        ):
            raise ValueError("calibration stream identity or temperature is invalid")
        temperatures.append(float(fit["temperature"]))
    return temperatures, _files(directory)


@torch.inference_mode()
def validate_and_freeze(
    source: Path,
    initial: Path,
    trained: Path,
    calibration: Path,
    output: Path,
    *,
    training_sha256: str,
    calibration_sha256: str,
    circuit: Any,
    global_decode: GlobalDecode,
    device: torch.device,
) -> dict[str, Any]:
    """Capture one shared physical stream after input checks, then freeze all60 trials.

    CPU supports synthetic correctness tests only; the registered formal workflow
    must supply its explicit CUDA device and external total-deadline controller.
    """
    if output.exists():
        raise FileExistsError(output)
    if torch.get_num_threads() != 1 or device.type not in ("cpu", "cuda"):
        raise ValueError("requires one thread and explicit CPU or CUDA")
    models, training_files, inputs = training.restore_final(
        source, initial, trained, training_sha256=training_sha256
    )
    initial_files = _files(initial)
    identities = [
        {
            "seed": seed,
            "architecture": architecture,
            "step": trainer.step,
            "training_state_sha256": training._state(trainer),
        }
        for seed, architecture, trainer in models
    ]
    temperatures, calibration_files = _calibration(
        calibration,
        expected_sha256=calibration_sha256,
        training_sha256=training_sha256,
        models=identities,
        training_files=training_files,
        initial_files=initial_files,
        inputs=inputs,
    )
    circuit_sha = hashlib.sha256(str(circuit).encode()).hexdigest()
    if (
        circuit_sha != inputs["circuit_sha256"]
        or circuit.num_detectors != models[0][2].model.h.shape[0]
        or circuit.num_observables != models[0][2].model.logical.shape[0]
    ):
        raise ValueError("validation circuit is not the verified complete cell")
    sources = _sources()
    protocols = {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS}
    plan = {
        "purpose": "matched_validation_choices",
        "inputs": inputs,
        "models": identities,
        "protocol_sha256": protocols,
        "source_sha256": sources,
        "training_sha256": training_sha256,
        "calibration_sha256": calibration_sha256,
        "validation_seed": VALIDATION_SEED,
        "validation_shots": VALIDATION_SHOTS,
        "chunk_size": CHUNK,
        "threshold_grid": [0.5, 0.7, 0.9, 0.99, 1.0],
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "plan.json").open("x") as stream:
        stream.write(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    plan_hash = archive.digest(output / "plan.json")
    validation = output / "validation"
    protocol_sha = archive.digest(checkpoints.PROTOCOLS[0])
    shards._capture_validation(
        circuit,
        validation,
        protocol_sha256=protocol_sha,
        sampler_seed=VALIDATION_SEED,
        shots=VALIDATION_SHOTS,
        chunk_size=CHUNK,
    )
    validation_files = _files(validation)
    replay = shards._replay_validation(
        validation,
        circuit_sha256=circuit_sha,
        protocol_sha256=protocol_sha,
        sampler_seed=VALIDATION_SEED,
        shots=VALIDATION_SHOTS,
        chunk_size=CHUNK,
    )
    variants = [
        (index, mode, 1.0 if mode == "raw" else temperatures[index])
        for index in range(6)
        for mode in ("raw", "calibrated")
    ]
    selections = selection._select_paths(
        [trainer.model for _, _, trainer in models],
        [t for _, _, t in variants],
        replay,
        global_decode=global_decode,
        device=device,
        shots=VALIDATION_SHOTS,
    )
    paths = [
        {
            **identities[index],
            "mode": mode,
            "temperature": temperature,
            "selection": asdict(chosen),
        }
        for (index, mode, temperature), chosen in zip(variants, selections, strict=True)
    ]
    if any(
        training._state(trainer) != row["training_state_sha256"]
        or trainer.model.training
        for (_, _, trainer), row in zip(models, identities, strict=True)
    ):
        raise ValueError("validation mutated a frozen training state")
    for directory, frozen in (
        (initial, initial_files),
        (trained, training_files),
        (calibration, calibration_files),
        (validation, validation_files),
    ):
        if _files(directory) != frozen:
            raise ValueError("validation source input or shard inventory changed")
    if (
        _sources() != sources
        or archive.digest(output / "plan.json") != plan_hash
        or {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS} != protocols
        or sorted(p.name for p in output.iterdir()) != ["plan.json", "validation"]
    ):
        raise ValueError("validation plan, source or output changed")
    result = {
        **plan,
        "initial_files": initial_files,
        "training_files": training_files,
        "calibration_files": calibration_files,
        "validation_files": validation_files,
        "paths": paths,
        "scientific_benefit_claim_allowed": False,
    }
    with (output / "choices.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (output / "choices.sha256").open("x") as stream:
        stream.write(archive.digest(output / "choices.json") + "\n")
    return result


def restore_choices(
    source: Path,
    initial: Path,
    trained: Path,
    calibration: Path,
    choices: Path,
    *,
    expected_sha256: str,
) -> tuple[list[tuple[int, str, Trainer]], dict[str, Any]]:
    """Verify the externally frozen decision and all dependencies without new draws."""
    checkpoints._sha256(expected_sha256)
    if (
        archive.digest(choices / "choices.json") != expected_sha256
        or (choices / "choices.sha256").read_text() != expected_sha256 + "\n"
    ):
        raise ValueError("external frozen choices hash mismatch")
    raw = json.loads((choices / "choices.json").read_text())
    models, training_files, inputs = training.restore_final(
        source, initial, trained, training_sha256=raw["training_sha256"]
    )
    identities = [
        {
            "seed": s,
            "architecture": a,
            "step": t.step,
            "training_state_sha256": training._state(t),
        }
        for s, a, t in models
    ]
    initial_files = _files(initial)
    temperatures, cal_files = _calibration(
        calibration,
        expected_sha256=raw["calibration_sha256"],
        training_sha256=raw["training_sha256"],
        models=identities,
        training_files=training_files,
        initial_files=initial_files,
        inputs=inputs,
    )
    if (
        raw["purpose"] != "matched_validation_choices"
        or raw["inputs"] != inputs
        or raw["models"] != identities
        or raw["source_sha256"] != _sources()
        or raw["protocol_sha256"]
        != {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS}
        or raw["validation_seed"] != VALIDATION_SEED
        or raw["validation_shots"] != VALIDATION_SHOTS
        or raw["chunk_size"] != CHUNK
        or raw["threshold_grid"] != [0.5, 0.7, 0.9, 0.99, 1.0]
        or raw["initial_files"] != initial_files
        or raw["training_files"] != training_files
        or raw["calibration_files"] != cal_files
        or _files(choices / "validation") != raw["validation_files"]
    ):
        raise ValueError("frozen choices inputs, source or recipe mismatch")
    plan = json.loads((choices / "plan.json").read_text())
    if plan != {key: raw[key] for key in plan} or set(plan) != {
        "purpose",
        "inputs",
        "models",
        "protocol_sha256",
        "source_sha256",
        "training_sha256",
        "calibration_sha256",
        "validation_seed",
        "validation_shots",
        "chunk_size",
        "threshold_grid",
    }:
        raise ValueError("frozen choices plan mismatch")
    expected = [(i, mode) for i in range(6) for mode in ("raw", "calibrated")]
    if len(raw["paths"]) != len(expected):
        raise ValueError("requires every registered native path")
    for (index, mode), row in zip(expected, raw["paths"], strict=True):
        if (
            any(row[key] != identities[index][key] for key in identities[index])
            or row["mode"] != mode
            or row["temperature"] != (1.0 if mode == "raw" else temperatures[index])
        ):
            raise ValueError("frozen native path identity or temperature mismatch")
        saved = row["selection"]
        if [t["threshold"] for t in saved["trials"]] != raw["threshold_grid"]:
            raise ValueError("requires all five registered threshold trials")
        trials = []
        for trial in saved["trials"]:
            c = trial["counts"]
            if (
                set(c)
                != {"shots", "baseline_failures", "hybrid_failures", "wins", "losses"}
                or any(
                    type(v) is not int or not 0 <= v <= VALIDATION_SHOTS
                    for v in c.values()
                )
                or c["shots"] != VALIDATION_SHOTS
                or c["wins"] > c["baseline_failures"]
                or c["losses"] > VALIDATION_SHOTS - c["baseline_failures"]
                or c["hybrid_failures"]
                != c["baseline_failures"] - c["wins"] + c["losses"]
            ):
                raise ValueError("invalid paired validation counts")
            trials.append(ThresholdTrial(trial["threshold"], PairedCounts(**c)))
        if asdict(_select_threshold_trials(tuple(trials))) != {
            **saved,
            "trials": tuple(saved["trials"]),
        }:
            raise ValueError("frozen threshold does not follow registered tie rule")
    if (
        len(
            {
                trial["counts"]["baseline_failures"]
                for path in raw["paths"]
                for trial in path["selection"]["trials"]
            }
        )
        != 1
    ):
        raise ValueError("all thresholds must use the same physical baseline")
    expected_files = {
        "plan.json": archive.digest(choices / "plan.json"),
        "choices.json": expected_sha256,
        "choices.sha256": archive.digest(choices / "choices.sha256"),
        **{"validation/" + name: sha for name, sha in raw["validation_files"].items()},
    }
    if _files(choices) != expected_files:
        raise ValueError("frozen choices inventory changed")
    return models, raw
