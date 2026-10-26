"""Run the frozen CPU diagnostics/calibration study; smoke is software evidence only."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import torch

from examples.qec.assess_predecoder import make_model, transfer
from flagquantum.qec import DetectorErrorModel, get_decoder
from flagquantum.qec.decoders._ai.assessment import PairedCounts, assess, summarize
from flagquantum.qec.decoders._ai.calibration import (
    TemperatureFit,
    ThresholdSelection,
    fit_temperature,
    select_threshold,
)
from flagquantum.qec.decoders._ai.diagnostics import LossDiagnostics, diagnose
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.decoders._ai.training import Trainer


@dataclass(frozen=True)
class StudyConfig:
    seeds: tuple[int, ...]
    models: tuple[str, ...]
    steps: int
    batch_size: int
    diagnostic_shots: int
    calibration_shots: int
    validation_shots: int
    test_shots: int
    train_cell: tuple[int, int, float] = (3, 3, 0.02)
    fit_cell: tuple[int, int, float] = (3, 3, 0.006)
    test_cell: tuple[int, int, float] = (5, 5, 0.003)
    temperatures: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    thresholds: tuple[float, ...] = (0.5, 0.7, 0.9, 0.99, 1.0)
    primary_alpha: float = 0.05 / 6


@dataclass(frozen=True)
class ControlResult:
    control: str
    temperature: float
    validation: ThresholdSelection
    test: PairedCounts
    summary: dict[str, object]


@dataclass(frozen=True)
class ModelResult:
    seed: int
    model: str
    checkpoint: str
    checkpoint_sha256: str
    first_loss: float
    last_loss: float
    training_diagnostics: LossDiagnostics
    calibration: TemperatureFit
    raw_validation_diagnostics: LossDiagnostics
    calibrated_validation_diagnostics: LossDiagnostics
    controls: tuple[ControlResult, ...]


@dataclass(frozen=True)
class TrainingRun:
    seed: int
    kind: str
    model: Predecoder
    checkpoint: Path
    losses: tuple[float, ...]


@dataclass(frozen=True)
class ValidationRun:
    training: TrainingRun
    training_diagnostics: LossDiagnostics
    fit: TemperatureFit
    raw_diagnostics: LossDiagnostics
    calibrated_diagnostics: LossDiagnostics
    selections: tuple[tuple[str, float, ThresholdSelection], ...]


@torch.no_grad()
def held_out_scores(
    model: Predecoder, shots: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    batch = model.sample(shots, torch.Generator().manual_seed(seed))
    return model(batch.sample.detectors), batch.mechanisms


def train_model(config: StudyConfig, output: Path, seed: int, kind: str) -> TrainingRun:
    torch.manual_seed(seed)
    trained = make_model(kind, *config.train_cell)
    trainer = Trainer(trained, seed=seed + 104729)
    losses = trainer.train(config.steps, batch_size=config.batch_size)
    checkpoint = output / f"{kind}-seed{seed}.pt"
    trainer.save(checkpoint)
    return TrainingRun(seed, kind, trained, checkpoint, tuple(losses))


def validate_model(config: StudyConfig, training: TrainingRun) -> ValidationRun:
    seed, kind, trained = training.seed, training.kind, training.model
    training_diagnostics = diagnose(
        *held_out_scores(trained, config.diagnostic_shots, seed + 3000000)
    )

    validation_model = make_model(kind, *config.fit_cell)
    transfer(trained, validation_model)
    fit = fit_temperature(
        *held_out_scores(validation_model, config.calibration_shots, seed + 4000000),
        temperatures=config.temperatures,
    )
    validation_logits, validation_labels = held_out_scores(
        validation_model, config.validation_shots, seed + 5000000
    )
    raw_diagnostics = diagnose(validation_logits, validation_labels)
    calibrated_diagnostics = diagnose(
        validation_logits / fit.temperature, validation_labels
    )
    validation_backend = get_decoder(
        "pymatching", DetectorErrorModel.from_stim_text(validation_model.dem_text)
    )
    selections = tuple(
        (
            name,
            temperature,
            select_threshold(
                validation_model,
                validation_backend,
                shots=config.validation_shots,
                seed=seed + 5000000,
                temperature=temperature,
                thresholds=config.thresholds,
            ),
        )
        for name, temperature in (("raw", 1.0), ("calibrated", fit.temperature))
    )
    return ValidationRun(
        training,
        training_diagnostics,
        fit,
        raw_diagnostics,
        calibrated_diagnostics,
        selections,
    )


def evaluate_model(config: StudyConfig, validated: ValidationRun) -> ModelResult:
    training = validated.training
    seed, kind = training.seed, training.kind
    test_model = make_model(kind, *config.test_cell)
    transfer(training.model, test_model)
    test_backend = get_decoder(
        "pymatching", DetectorErrorModel.from_stim_text(test_model.dem_text)
    )
    controls: list[ControlResult] = []
    for name, temperature, selection in validated.selections:
        counts = assess(
            test_model,
            test_backend,
            shots=config.test_shots,
            seed=seed + 6000000,
            threshold=selection.threshold,
            temperature=temperature,
        )
        controls.append(
            ControlResult(
                name,
                temperature,
                selection,
                counts,
                summarize(
                    counts, alpha=config.primary_alpha if name == "calibrated" else 0.05
                ),
            )
        )
        print(
            f"seed={seed} model={kind} control={name} temperature={temperature} "
            f"threshold={selection.threshold} baseline={counts.baseline_failures} "
            f"hybrid={counts.hybrid_failures}",
            file=sys.stderr,
            flush=True,
        )
    return ModelResult(
        seed,
        kind,
        training.checkpoint.name,
        hashlib.sha256(training.checkpoint.read_bytes()).hexdigest(),
        training.losses[0],
        training.losses[-1],
        validated.training_diagnostics,
        validated.fit,
        validated.raw_diagnostics,
        validated.calibrated_diagnostics,
        tuple(controls),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    config = StudyConfig(
        seeds=(11,) if args.smoke else (11, 22, 33),
        models=("mlp", "cnn"),
        steps=2 if args.smoke else 200,
        batch_size=16 if args.smoke else 128,
        diagnostic_shots=32 if args.smoke else 4096,
        calibration_shots=32 if args.smoke else 4096,
        validation_shots=32 if args.smoke else 4096,
        test_shots=32 if args.smoke else 65536,
    )
    root = Path(__file__).resolve().parents[2]
    protocol = root / "docs/development/QEC_AI_CALIBRATION.md"
    sources = [
        Path(__file__).resolve(),
        root / "examples/qec/assess_predecoder.py",
        *sorted((root / "flagquantum/qec/decoders/_ai").glob("*.py")),
    ]
    manifest: dict[str, object] = {
        "schema": "qec_predecoder_calibration_v1",
        "mode": "smoke" if args.smoke else "full",
        "claim": "diagnostic_research_only",
        "configuration": asdict(config),
        "configuration_sha256": hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True).encode()
        ).hexdigest(),
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pymatching": version("pymatching"),
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "distinct_test_draws": len(config.seeds) * config.test_shots,
    }
    args.output_directory.mkdir(parents=True, exist_ok=False)
    with (args.output_directory / "manifest.json").open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
        file.write("\n")
    # Global stage barriers freeze every checkpoint, then every validation choice,
    # before any model's primary test is sampled.
    trained = [
        train_model(config, args.output_directory, seed, kind)
        for seed in config.seeds
        for kind in config.models
    ]
    validated = [validate_model(config, training) for training in trained]
    rows = [asdict(evaluate_model(config, validation)) for validation in validated]
    with (args.output_directory / "results.json").open("x", encoding="utf-8") as file:
        json.dump({**manifest, "rows": rows}, file, indent=2, allow_nan=False)
        file.write("\n")
    print(
        json.dumps(
            {"output_directory": str(args.output_directory), "models": len(rows)}
        )
    )


if __name__ == "__main__":
    main()
