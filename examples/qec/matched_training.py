"""Internal matched-training stage; no CLI or formal launch authorization.

A complete trusted controller, final-source CI and matching preflight remain
prerequisites. Tests reduce constants only for explicit synthetic fixtures.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

import torch

from examples.qec import continued_primary, ising_layout
from examples.qec import diagnose_predecoder as archive
from examples.qec import matched_checkpoints as checkpoints
from examples.qec import matched_initialization as initialization
from flagquantum.qec.decoders._ai.diagnostics import LossDiagnostics, diagnose
from flagquantum.qec.decoders._ai.predecoder import Predecoder, TrainingBatch
from flagquantum.qec.decoders._ai.training import Trainer
from flagquantum.qec.dem.model import DetectorErrorModel

HORIZONS = (0, 200, 2000)
TRAIN_BATCH = 128
SCORE_SHOTS = 4096
SCORE_CHUNK = 32


def _state(trainer: Trainer) -> str:
    tensors = {"model/" + k: v for k, v in trainer.model.state_dict().items()}
    tensors["generator"] = trainer.generator.get_state()
    optimizer = trainer.optimizer.state_dict()
    for index, state in optimizer["state"].items():
        for name, value in state.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError("optimizer state is not tensor-valued")
            tensors[f"adam/{index}/{name}"] = value
    content = (
        archive.state_digest(tensors)
        + json.dumps(optimizer["param_groups"], sort_keys=True)
        + str(trainer.step)
    )
    return hashlib.sha256(content.encode()).hexdigest()


def _hashes(batch: TrainingBatch, model: Predecoder, shots: int) -> dict[str, str]:
    if (
        batch.mechanisms.shape != (shots, model.prior.numel())
        or batch.sample.detectors.shape != (shots, model.h.shape[0])
        or batch.sample.observables.shape != (shots, model.logical.shape[0])
    ):
        raise ValueError("sampled batch differs from registered shapes")
    tensors = {
        "latent": batch.mechanisms,
        "detectors": batch.sample.detectors,
        "truth": batch.sample.observables,
    }
    if any(not bool(((v == 0) | (v == 1)).all()) for v in tensors.values()):
        raise ValueError("training batch must be binary")
    return {
        name: archive.state_digest({name: tensor.to(torch.int8)})
        for name, tensor in tensors.items()
    }


def _update(
    trainer: Trainer, expected: dict[str, str] | None
) -> tuple[float, dict[str, str]]:
    """Observe the Trainer's one actual draw; never resample to inspect a batch."""
    model = trainer.model
    if "sample" in model.__dict__:
        raise ValueError("model already has an instance sampling observer")
    sampler = model.sample
    seen: list[dict[str, str]] = []
    before = trainer.step

    def observed(shots: int, generator: torch.Generator) -> TrainingBatch:
        if seen or shots != TRAIN_BATCH or generator is not trainer.generator:
            raise ValueError("Trainer did not request its one registered batch")
        state_before = archive.state_digest({"generator": generator.get_state()})
        batch = sampler(shots, generator)
        hashes = _hashes(batch, model, shots)
        hashes["generator_before"] = state_before
        hashes["generator_after"] = archive.state_digest(
            {"generator": generator.get_state()}
        )
        if expected is not None and hashes != expected:
            raise ValueError("paired training batch hash mismatch before update")
        seen.append(hashes)
        return batch

    setattr(model, "sample", observed)
    try:
        losses = trainer.train(1, batch_size=TRAIN_BATCH)
    finally:
        delattr(model, "sample")
    if len(seen) != 1 or len(losses) != 1 or trainer.step != before + 1:
        raise ValueError("Trainer update counter or observation mismatch")
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("nonfinite trained parameters")
    return losses[0], seen[0]


def train_pair(cnn: Trainer, mlp: Trainer, target: int, stream: TextIO) -> None:
    """Use two independent generators and compare every actual draw before MLP update."""
    if (
        type(target) is not int
        or target not in HORIZONS[1:]
        or cnn.step != mlp.step
        or cnn.step != HORIZONS[HORIZONS.index(target) - 1]
    ):
        raise ValueError("paired training requires an equal prior registered horizon")
    if cnn.model.dem_text != mlp.model.dem_text:
        raise ValueError("paired training requires the same canonical DEM")
    while cnn.step < target:
        loss, expected = _update(cnn, None)
        stream.write(
            json.dumps(
                {
                    "architecture": "cnn",
                    "step": cnn.step,
                    "loss": loss,
                    "batch_sha256": expected,
                },
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()
        loss, actual = _update(mlp, expected)
        stream.write(
            json.dumps(
                {
                    "architecture": "mlp",
                    "step": mlp.step,
                    "loss": loss,
                    "batch_sha256": actual,
                },
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()


def _load_initial(
    source: Path, directory: Path, expected_sha256: str
) -> tuple[list[tuple[int, str, Trainer]], dict[str, str], dict[str, Any]]:
    checkpoints._sha256(expected_sha256)
    manifest = directory / "initialization.json"
    if archive.digest(manifest) != expected_sha256:
        raise ValueError("external initialization hash mismatch")
    raw = json.loads(manifest.read_text())
    with torch.random.fork_rng(devices=[]):
        template, inputs = ising_layout.build_model(source)
    protocols = {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS}
    layout_sha = inputs["layout_sha256"]
    if not isinstance(layout_sha, str):
        raise ValueError("verified layout digest is missing")
    if (
        raw["purpose"] != "matched_initialization_preflight_only"
        or raw["inputs"] != inputs
        or raw["protocol_sha256"] != protocols
        or raw["threads"] != 1
        or raw["torch"] != str(torch.__version__)
        or raw["new_samples"] != 0
        or raw["new_optimizer_steps"] != 0
        or raw["formal_launch_authorized"] is not False
    ):
        raise ValueError(
            "initialization differs from current verified inputs/environment"
        )
    rows = raw["initial_checkpoints"]
    if [(r["seed"], r["architecture"], r["step"]) for r in rows] != [
        (s, a, 0) for s in (11, 22, 33) for a in ("cnn", "mlp")
    ]:
        raise ValueError("requires all six ordered step-zero checkpoints")
    rng_path = directory / "initialization-rng.pt"
    if archive.digest(rng_path) != raw["initialization_rng_file_sha256"]:
        raise ValueError("initialization RNG file hash mismatch")
    rng = torch.load(rng_path, map_location="cpu", weights_only=True)
    names = {
        f"{a}-{s}-{phase}"
        for s in (11, 22, 33)
        for a in ("cnn", "mlp")
        for phase in ("before", "after")
    }
    if set(rng) != names:
        raise ValueError("initialization RNG inventory mismatch")
    files = {
        "initialization.json": expected_sha256,
        "initialization-rng.pt": raw["initialization_rng_file_sha256"],
    }
    models: list[tuple[int, str, Trainer]] = []
    dem = DetectorErrorModel.from_stim_text(template.dem_text)
    for row in rows:
        seed, architecture = row["seed"], row["architecture"]
        name = f"{architecture}-{seed}"
        for phase in ("before", "after"):
            if (
                archive.state_digest({"rng": rng[name + "-" + phase]})
                != row[f"initialization_rng_{phase}_sha256"]
            ):
                raise ValueError("initialization RNG tensor hash mismatch")
        with torch.random.fork_rng(devices=[]):
            model = (
                copy.deepcopy(template)
                if architecture == "cnn"
                else Predecoder(dem, allow_hyperedges=True)
            )
        trainer = checkpoints.restore_bound(
            model,
            directory / name,
            seed=seed,
            layout_sha256=layout_sha,
            binding_sha256=row["binding_sha256"],
        )
        if (
            trainer.step != 0
            or archive.state_digest(trainer.model.state_dict())
            != row["model_state_sha256"]
            or archive.state_digest({"rng": trainer.generator.get_state()})
            != row["training_generator_sha256"]
        ):
            raise ValueError("initial model state/stream mismatch")
        models.append((seed, architecture, trainer))
        for filename in ("checkpoint.pt", "binding.json"):
            files[name + "/" + filename] = archive.digest(directory / name / filename)
    if any(archive.digest(directory / name) != value for name, value in files.items()):
        raise ValueError("initial files changed during verification")
    return models, files, inputs


def _heldout(model: Predecoder, seed: int, output: Path) -> dict[str, str]:
    output.mkdir(exist_ok=False)
    generator = torch.Generator().manual_seed(seed + 19100000)
    files: dict[str, str] = {}
    for index in range(SCORE_SHOTS // SCORE_CHUNK):
        batch = model.sample(SCORE_CHUNK, generator)
        _hashes(batch, model, SCORE_CHUNK)
        path = output / f"chunk-{index:04d}.pt"
        with path.open("xb") as stream:
            torch.save(
                {
                    "detectors": batch.sample.detectors,
                    "labels": batch.mechanisms,
                    "truth": batch.sample.observables,
                },
                stream,
            )
        files[path.name] = archive.digest(path)
    return files


@torch.inference_mode()
def _score(
    trainer: Trainer, inputs: Path, files: dict[str, str], output: Path
) -> dict[str, object]:
    before = _state(trainer)
    was_training = trainer.model.training
    output.mkdir(exist_ok=False)
    losses: LossDiagnostics | None = None
    scores: dict[str, str] = {}
    try:
        trainer.model.eval()
        if sorted(p.name for p in inputs.iterdir()) != sorted(files):
            raise ValueError("held-out input inventory mismatch")
        for name, fingerprint in files.items():
            path = inputs / name
            if archive.digest(path) != fingerprint:
                raise ValueError("held-out input file hash mismatch")
            batch = torch.load(path, map_location="cpu", weights_only=True)
            logits = trainer.model(batch["detectors"])
            losses = archive.add_loss(losses, diagnose(logits, batch["labels"]))
            destination = output / name
            with destination.open("xb") as stream:
                torch.save({"logits": logits, "input_file_sha256": fingerprint}, stream)
            scores[name] = archive.digest(destination)
    finally:
        trainer.model.train(was_training)
    if _state(trainer) != before or losses is None:
        raise ValueError("scoring mutated training state or produced no scores")
    return {
        "step": trainer.step,
        "loss": asdict(losses),
        "mean_bce": losses.mean_bce,
        "score_files": scores,
        "training_state_sha256": before,
    }


def _sources() -> dict[str, str]:
    result = continued_primary._sources()
    for path in (
        Path(__file__),
        Path(initialization.__file__),
        Path(checkpoints.__file__),
        Path(ising_layout.__file__),
    ):
        result[str(path.relative_to(archive.ROOT))] = archive.digest(path)
    return result


def run_training(
    source: Path, initial: Path, output: Path, *, initialization_sha256: str
) -> None:
    """Internal fixed-budget stage only; trusted global launch/deadline gates are external.

    This function is not exposed by the zero-sample CLI. Do not invoke it for a
    formal study before the complete workflow's acceptance gates are satisfied.
    """
    if output.exists():
        raise FileExistsError(output)
    if HORIZONS != checkpoints.HORIZONS or any(
        type(n) is not int or n <= 0 for n in (TRAIN_BATCH, SCORE_SHOTS, SCORE_CHUNK)
    ):
        raise ValueError("training/checkpoint budgets disagree or are nonpositive")
    if torch.get_num_threads() != 1 or SCORE_CHUNK > 32 or SCORE_SHOTS % SCORE_CHUNK:
        raise ValueError("requires one thread and bounded divisible scoring chunks")
    models, frozen, inputs = _load_initial(source, initial, initialization_sha256)
    sources = _sources()
    protocols = {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS}
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "purpose": "matched_training_stage",
        "initialization_sha256": initialization_sha256,
        "input_files": frozen,
        "inputs": inputs,
        "protocol_sha256": protocols,
        "source_sha256": sources,
        "seeds": [11, 22, 33],
        "architectures": ["cnn", "mlp"],
        "horizons": list(HORIZONS),
        "train_batch": TRAIN_BATCH,
        "training_seed_offset": 18100000,
        "heldout_seed_offset": 19100000,
        "score_chunk": SCORE_CHUNK,
        "heldout_shots_per_seed": SCORE_SHOTS,
        "scientific_benefit_claim_allowed": False,
    }
    with (output / "plan.json").open("x") as stream:
        stream.write(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    records: list[dict[str, object]] = []
    training_logs: dict[str, str] = {}
    heldout_files: dict[str, str] = {}
    for index, seed in enumerate((11, 22, 33)):
        cnn, mlp = (row[2] for row in models[2 * index : 2 * index + 2])
        heldout = output / f"heldout-{seed}"
        files = _heldout(cnn.model, seed, heldout)
        heldout_files.update(
            {heldout.name + "/" + name: value for name, value in files.items()}
        )
        with (output / f"seed-{seed}-training.jsonl").open("x") as log:
            for horizon in HORIZONS:
                if horizon:
                    train_pair(cnn, mlp, horizon, log)
                for architecture, trainer in (("cnn", cnn), ("mlp", mlp)):
                    name = f"{architecture}-{seed}-step-{horizon}"
                    digest = checkpoints.save_bound(
                        trainer,
                        output / name,
                        seed=seed,
                        layout_sha256=inputs["layout_sha256"],
                    )
                    score = _score(trainer, heldout, files, output / (name + "-scores"))
                    records.append(
                        {
                            "seed": seed,
                            "architecture": architecture,
                            "step": horizon,
                            "binding_sha256": digest,
                            "heldout_input_files": files,
                            "score": score,
                        }
                    )
                print(
                    json.dumps(
                        {
                            "phase": "matched_training_horizon",
                            "seed": seed,
                            "step": horizon,
                        }
                    ),
                    flush=True,
                )
        training_logs[f"seed-{seed}-training.jsonl"] = archive.digest(
            output / f"seed-{seed}-training.jsonl"
        )
    if (
        any(
            archive.digest(output / name) != value
            for name, value in heldout_files.items()
        )
        or _sources() != sources
        or {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS} != protocols
        or any(
            archive.digest(initial / name) != fingerprint
            for name, fingerprint in frozen.items()
        )
    ):
        raise ValueError(
            "frozen source/protocol/initial inputs changed during training"
        )
    result = {
        **plan,
        "records": records,
        "training_logs": training_logs,
        "heldout_files": heldout_files,
        "independent_training_draws": 3 * HORIZONS[-1] * TRAIN_BATCH,
        "training_exposures": 6 * HORIZONS[-1] * TRAIN_BATCH,
        "optimizer_updates": 6 * HORIZONS[-1],
        "independent_heldout_draws": 3 * SCORE_SHOTS,
        "new_calibration_shots": 0,
        "new_validation_shots": 0,
        "new_primary_shots": 0,
    }
    with (output / "manifest.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


def _verify_training_log(path: Path, initial_generator: str) -> str:
    previous = initial_generator
    with path.open() as stream:
        for step in range(1, HORIZONS[-1] + 1):
            expected: dict[str, str] | None = None
            for architecture in ("cnn", "mlp"):
                line = stream.readline()
                if not line:
                    raise ValueError("training loss ledger is incomplete")
                row = json.loads(line)
                loss, hashes = row["loss"], row["batch_sha256"]
                if (
                    row["architecture"] != architecture
                    or type(row["step"]) is not int
                    or row["step"] != step
                    or type(loss) not in (int, float)
                    or not math.isfinite(loss)
                    or loss < 0
                ):
                    raise ValueError("training loss/counter/order ledger mismatch")
                if set(hashes) != {
                    "latent",
                    "detectors",
                    "truth",
                    "generator_before",
                    "generator_after",
                }:
                    raise ValueError("training batch hash fields mismatch")
                for value in hashes.values():
                    checkpoints._sha256(value)
                if hashes["generator_before"] != previous or (
                    expected is not None and hashes != expected
                ):
                    raise ValueError("training pair or generator continuity mismatch")
                expected = hashes
            assert expected is not None
            previous = expected["generator_after"]
        if stream.read().strip():
            raise ValueError("training loss ledger has extra updates")
    return previous


def restore_final(
    source: Path,
    initial: Path,
    trained: Path,
    *,
    training_sha256: str,
) -> tuple[list[tuple[int, str, Trainer]], dict[str, str], dict[str, Any]]:
    """Verify the complete training inventory before returning all six final models.

    Reuse bound checkpoint restoration and retain architecture/seed/step identity.
    This gate draws no data and performs no optimizer update or model forward.
    """
    checkpoints._sha256(training_sha256)
    manifest = trained / "manifest.json"
    if archive.digest(manifest) != training_sha256:
        raise ValueError("external training manifest hash mismatch")
    raw = json.loads(manifest.read_text())
    models, frozen_initial, inputs = _load_initial(
        source, initial, raw["initialization_sha256"]
    )
    plan = json.loads((trained / "plan.json").read_text())
    declarations = {
        "purpose": "matched_training_stage",
        "initialization_sha256": raw["initialization_sha256"],
        "input_files": frozen_initial,
        "inputs": inputs,
        "protocol_sha256": {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS},
        "source_sha256": _sources(),
        "seeds": [11, 22, 33],
        "architectures": ["cnn", "mlp"],
        "horizons": list(HORIZONS),
        "train_batch": TRAIN_BATCH,
        "training_seed_offset": 18100000,
        "heldout_seed_offset": 19100000,
        "score_chunk": SCORE_CHUNK,
        "heldout_shots_per_seed": SCORE_SHOTS,
        "scientific_benefit_claim_allowed": False,
    }
    if plan != declarations or any(
        raw.get(key) != value for key, value in declarations.items()
    ):
        raise ValueError("training plan/context differs from frozen recipe")
    counts = {
        "independent_training_draws": 3 * HORIZONS[-1] * TRAIN_BATCH,
        "training_exposures": 6 * HORIZONS[-1] * TRAIN_BATCH,
        "optimizer_updates": 6 * HORIZONS[-1],
        "independent_heldout_draws": 3 * SCORE_SHOTS,
        "new_calibration_shots": 0,
        "new_validation_shots": 0,
        "new_primary_shots": 0,
    }
    if any(
        type(raw.get(key)) is not int or raw[key] != value
        for key, value in counts.items()
    ):
        raise ValueError("training draw/update counts differ from fixed budget")
    expected = [
        (seed, architecture, horizon)
        for seed in (11, 22, 33)
        for horizon in HORIZONS
        for architecture in ("cnn", "mlp")
    ]
    records = raw["records"]
    if [
        (row["seed"], row["architecture"], row["step"]) for row in records
    ] != expected or any(type(row["step"]) is not int for row in records):
        raise ValueError(
            "requires all eighteen ordered architecture/seed/horizon records"
        )
    chunks = [f"chunk-{i:04d}.pt" for i in range(SCORE_SHOTS // SCORE_CHUNK)]
    logs = {f"seed-{seed}-training.jsonl" for seed in (11, 22, 33)}
    heldout_names = {
        f"heldout-{seed}/{chunk}" for seed in (11, 22, 33) for chunk in chunks
    }
    if set(raw["training_logs"]) != logs or set(raw["heldout_files"]) != heldout_names:
        raise ValueError("training log or shared held-out inventory mismatch")
    files = {
        "manifest.json": training_sha256,
        "plan.json": archive.digest(trained / "plan.json"),
        **raw["training_logs"],
        **raw["heldout_files"],
    }
    prototypes = {
        (seed, architecture): trainer.model for seed, architecture, trainer in models
    }
    for row in records:
        seed, architecture, horizon = row["seed"], row["architecture"], row["step"]
        name = f"{architecture}-{seed}-step-{horizon}"
        shared = {
            chunk: raw["heldout_files"][f"heldout-{seed}/{chunk}"] for chunk in chunks
        }
        if (
            row["heldout_input_files"] != shared
            or set(row["score"]["score_files"]) != set(chunks)
            or row["score"]["step"] != horizon
        ):
            raise ValueError("score identity/shared held-out binding mismatch")
        binding_path = trained / name / "binding.json"
        if archive.digest(binding_path) != row["binding_sha256"]:
            raise ValueError("horizon checkpoint binding hash mismatch")
        binding = json.loads(binding_path.read_text())
        if (
            type(binding["step"]) is not int
            or binding["step"] != horizon
            or binding["identity"]
            != checkpoints._identity(
                prototypes[seed, architecture], seed, inputs["layout_sha256"]
            )
        ):
            raise ValueError("horizon checkpoint architecture/layout identity mismatch")
        files[name + "/binding.json"] = row["binding_sha256"]
        files[name + "/checkpoint.pt"] = binding["checkpoint_sha256"]
        files.update(
            {
                name + "-scores/" + chunk: value
                for chunk, value in row["score"]["score_files"].items()
            }
        )
    if {str(p.relative_to(trained)) for p in trained.rglob("*") if p.is_file()} != set(
        files
    ):
        raise ValueError("complete training artifact inventory mismatch")
    for name, fingerprint in files.items():
        checkpoints._sha256(fingerprint)
        if archive.digest(trained / name) != fingerprint:
            raise ValueError("training artifact file hash mismatch")
    stream_after = {
        seed: _verify_training_log(
            trained / f"seed-{seed}-training.jsonl",
            archive.state_digest({"generator": trainer.generator.get_state()}),
        )
        for seed, architecture, trainer in models
        if architecture == "cnn"
    }
    final: list[tuple[int, str, Trainer]] = []
    for row in records:
        if row["step"] != HORIZONS[-1]:
            continue
        seed, architecture = row["seed"], row["architecture"]
        trainer = checkpoints.restore_bound(
            prototypes[seed, architecture],
            trained / f"{architecture}-{seed}-step-{HORIZONS[-1]}",
            seed=seed,
            layout_sha256=inputs["layout_sha256"],
            binding_sha256=row["binding_sha256"],
        )
        if (
            trainer.step != HORIZONS[-1]
            or archive.state_digest({"generator": trainer.generator.get_state()})
            != stream_after[seed]
            or _state(trainer) != row["score"]["training_state_sha256"]
        ):
            raise ValueError("final restored training/scoring state mismatch")
        trainer.model.eval()
        final.append((seed, architecture, trainer))
    if [(seed, arch) for seed, arch, _ in final] != [
        (seed, arch) for seed in (11, 22, 33) for arch in ("cnn", "mlp")
    ]:
        raise ValueError("requires all six final models, without horizon selection")
    if any(
        archive.digest(trained / name) != value for name, value in files.items()
    ) or any(
        archive.digest(initial / name) != value
        for name, value in frozen_initial.items()
    ):
        raise ValueError("frozen training/initial files changed during restoration")
    return final, files, inputs
