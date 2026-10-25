"""Execute the preregistered CPU continuation study on audited checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

import torch
from torch.nn import functional

from examples.qec import diagnose_predecoder as archive
from flagquantum.qec.decoders._ai.diagnostics import (
    EquivalenceDiagnostics,
    FeatureTable,
    LossDiagnostics,
    diagnose,
    diagnose_equivalence,
)
from flagquantum.qec.decoders._ai.training import Trainer

PROTOCOL = archive.ROOT / "docs/development/QEC_AI_TRAINING_SUFFICIENCY.md"
HORIZONS = (200, 600, 2000)
SHOTS = 4096


class StudyManifest(TypedDict):
    purpose: str
    protocol_sha256: str
    source_sha256: dict[str, str]
    input_sha256: dict[str, str]
    seeds: list[int]
    horizons: list[int]
    heldout_shots_per_seed: int
    heldout_seed_offset: int
    batch_size: int
    new_training_steps_per_seed: int
    new_calibration_shots: int
    new_validation_shots: int
    new_primary_test_shots: int
    threads: int
    torch: str
    scientific_benefit_claim_allowed: bool


@dataclass(frozen=True)
class Scores:
    seed: int
    step: int
    sampler_seed: int
    shots: int
    latent_bytes_sha256: str
    model_state_sha256: str
    raw_loss: LossDiagnostics
    prior_loss: LossDiagnostics
    zero_probability_bce: float | None
    zero_probability_bce_is_infinite: bool
    minimum_logit: float
    maximum_logit: float
    active_shots: dict[str, int]
    controls: dict[str, EquivalenceDiagnostics]
    groups: list[archive.GroupReport]
    per_shot_bce: list[float]
    paired_bce_difference: list[float]
    paired_difference_mean: float
    paired_difference_std: float
    paired_difference_quantiles: list[float]


def load_trainers(directory: Path) -> tuple[list[tuple[int, Trainer]], dict[str, str]]:
    """Reuse archive verification, then validate every saved continuation state."""
    models, hashes = archive.load_inputs(directory)
    loaded: list[tuple[int, Trainer]] = []
    for archived in models:
        trainer = Trainer(archived.model)
        trainer.restore(directory / f"native-{archived.seed}.pt")
        groups = trainer.optimizer.param_groups
        if trainer.step != 200 or len(groups) != 1:
            raise ValueError("continuation requires one step-200 Adam parameter group")
        group = groups[0]
        expected = {
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0,
            "amsgrad": False,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
            "foreach": None,
            "fused": None,
        }
        if any(group.get(k) != v for k, v in expected.items()):
            raise ValueError("archived Adam settings differ from the fixed trajectory")
        parameters = list(trainer.model.parameters())
        if len(trainer.optimizer.state) != len(parameters):
            raise ValueError("archived Adam state is incomplete")
        for parameter in parameters:
            state = trainer.optimizer.state[parameter]
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("archived Adam state fields differ")
            for key in ("exp_avg", "exp_avg_sq"):
                value = state[key]
                if value.shape != parameter.shape or not bool(
                    torch.isfinite(value).all()
                ):
                    raise ValueError("archived Adam moments are malformed or nonfinite")
            if float(state["step"]) != 200 or bool((state["exp_avg_sq"] < 0).any()):
                raise ValueError("archived Adam counter or second moment is invalid")
        if not all(
            bool(torch.isfinite(v).all()) for v in trainer.model.state_dict().values()
        ):
            raise ValueError("archived model contains nonfinite values")
        loaded.append((archived.seed, trainer))
    return loaded, hashes


@torch.no_grad()
def score(
    trainer: Trainer,
    seed: int,
    *,
    shots: int = SHOTS,
    chunk: int = 32,
    reference: Scores | None = None,
) -> Scores:
    """Replay an independent held-out stream without advancing training state."""
    if shots <= 0 or chunk <= 0 or chunk > 32 or shots % chunk:
        raise ValueError("scoring requires positive shots divisible by chunk <= 32")
    model = trainer.model
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed + 12100000)
    table = FeatureTable()
    controls = {
        k: EquivalenceDiagnostics(0, 0, 0, 0, 0, 0)
        for k in ("zero", "raw_0.5", "raw_0.9")
    }
    active = dict.fromkeys(controls, 0)
    losses: list[float] = []
    raw: LossDiagnostics | None = None
    prior: LossDiagnostics | None = None
    latent_hash = hashlib.sha256()
    minimum, maximum = float("inf"), float("-inf")
    h, logical = model.h.double(), model.logical.double()
    try:
        for _ in range(shots // chunk):
            batch = model.sample(chunk, generator)
            labels, detectors = batch.mechanisms, batch.sample.detectors
            logits = model(detectors)
            raw = archive.add_loss(raw, diagnose(logits, labels))
            prior = archive.add_loss(
                prior, diagnose(torch.logit(model.prior).expand_as(logits), labels)
            )
            table.update(model._features(detectors), logits, labels)
            losses.extend(
                functional.binary_cross_entropy_with_logits(
                    logits.double(), labels.double(), reduction="none"
                )
                .mean(dim=1)
                .tolist()
            )
            minimum = min(minimum, float(logits.min()))
            maximum = max(maximum, float(logits.max()))
            latent_hash.update(archive.tensor_bytes(labels.to(torch.int8)))
            candidates = {
                "zero": torch.zeros_like(labels),
                "raw_0.5": logits.sigmoid() >= 0.5,
                "raw_0.9": logits.sigmoid() >= 0.9,
            }
            for name, candidate in candidates.items():
                active[name] += int(candidate.bool().any(dim=1).sum())
                controls[name] = archive.add_equivalence(
                    controls[name], diagnose_equivalence(h, logical, labels, candidate)
                )
    finally:
        model.train(was_training)
    assert raw is not None and prior is not None
    fingerprint = latent_hash.hexdigest()
    if reference is not None and (
        reference.seed != seed
        or reference.shots != shots
        or reference.latent_bytes_sha256 != fingerprint
    ):
        raise ValueError("paired scoring stream does not match step-200 reference")
    values = torch.tensor(losses, dtype=torch.float64)
    baseline = (
        values
        if reference is None
        else torch.tensor(reference.per_shot_bce, dtype=torch.float64)
    )
    difference = values - baseline
    return Scores(
        seed,
        trainer.step,
        seed + 12100000,
        shots,
        fingerprint,
        archive.state_digest(model.state_dict()),
        raw,
        prior,
        None if raw.positive_labels else 0.0,
        bool(raw.positive_labels),
        minimum,
        maximum,
        active,
        controls,
        archive.group_reports(table.summarize()),
        losses,
        difference.tolist(),
        float(difference.mean()),
        float(difference.std(unbiased=False)),
        torch.quantile(
            difference, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        ).tolist(),
    )


def run(directory: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    trainers, inputs = load_trainers(directory)
    output.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        Path(archive.__file__),
        archive.ROOT / "flagquantum/qec/decoders/_ai/training.py",
        archive.ROOT / "flagquantum/qec/decoders/_ai/predecoder.py",
        archive.ROOT / "flagquantum/qec/decoders/_ai/diagnostics.py",
    ]
    manifest: StudyManifest = {
        "purpose": "fixed_budget_unchanged_model_training_response",
        "protocol_sha256": archive.digest(PROTOCOL),
        "source_sha256": {
            str(p.relative_to(archive.ROOT)): archive.digest(p) for p in sources
        },
        "input_sha256": inputs,
        "seeds": list(archive.SEEDS),
        "horizons": list(HORIZONS),
        "heldout_shots_per_seed": SHOTS,
        "heldout_seed_offset": 12100000,
        "batch_size": 128,
        "new_training_steps_per_seed": 1800,
        "new_calibration_shots": 0,
        "new_validation_shots": 0,
        "new_primary_test_shots": 0,
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "scientific_benefit_claim_allowed": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    responses: dict[str, bool] = {}
    for seed, trainer in trainers:
        reference: Scores | None = None
        for step in HORIZONS:
            print(
                json.dumps({"phase": "horizon", "seed": seed, "step": step}), flush=True
            )
            if step > trainer.step:
                # Keep every update's loss; checkpoints retain the original schema.
                with (output / f"seed-{seed}-training.jsonl").open("a") as stream:
                    while trainer.step < step:
                        loss = trainer.train(1, batch_size=128)[0]
                        stream.write(
                            json.dumps({"step": trainer.step, "loss": loss}) + "\n"
                        )
                        stream.flush()
                        if trainer.step % 100 == 0:
                            print(
                                json.dumps(
                                    {
                                        "phase": "training",
                                        "seed": seed,
                                        "step": trainer.step,
                                    }
                                ),
                                flush=True,
                            )
            trainer.save(output / f"native-{seed}-step-{step}.pt")
            result = score(trainer, seed, reference=reference)
            if reference is None:
                reference = result
            report = asdict(result)
            report["bce_ratio_to_step200"] = (
                result.raw_loss.mean_bce / reference.raw_loss.mean_bce
            )
            (output / f"seed-{seed}-step-{step}.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            if step == 2000:
                responses[str(seed)] = (
                    result.raw_loss.mean_bce <= 0.8 * reference.raw_loss.mean_bce
                )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "per_seed_descriptive_response": responses,
                "consistent_descriptive_training_response": all(responses.values()),
                "logical_quality_claim_allowed": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps({"status": "completed", "output_directory": str(output)}), flush=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run(args.input_directory, args.output_directory)


if __name__ == "__main__":
    main()
