"""Prospective CPU study: validation-selected thresholds and paired held-out tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch

from flagquantum.qec import (
    DetectorErrorModel,
    PhenomenologicalNoise,
    RotatedSurfaceCode,
    build_memory_circuit,
    get_decoder,
)
from flagquantum.qec.decoders._ai.assessment import assess, block_timings, summarize
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.decoders._ai.spacetime import SpacetimePredecoder
from flagquantum.qec.decoders._ai.training import Trainer


def make_model(kind: str, distance: int, rounds: int, probability: float) -> Predecoder:
    memory = build_memory_circuit(RotatedSurfaceCode(distance=distance), rounds=rounds)
    noise = PhenomenologicalNoise(data_flip=probability, measurement_flip=probability)
    if kind == "mlp":
        return Predecoder(DetectorErrorModel.from_memory_circuit(memory, noise=noise))
    model = SpacetimePredecoder(memory, noise)
    if kind == "cnn_no_types":
        model.masks[1:] = 0
    return model


def transfer(source: Predecoder, target: Predecoder) -> None:
    """Copy trained feature weights only; preserve target DEM, priors and geometry."""
    target.network.load_state_dict(source.network.state_dict())
    if isinstance(source, SpacetimePredecoder) and isinstance(
        target, SpacetimePredecoder
    ):
        target.encoder.load_state_dict(source.encoder.state_dict())


def run(smoke: bool) -> dict[str, object]:
    seeds = (11,) if smoke else (11, 22, 33)
    grid = (
        ((3, 3, 0.003),)
        if smoke
        else (
            (3, 3, 0.001),
            (3, 3, 0.003),
            (3, 3, 0.006),
            (5, 5, 0.003),
            (7, 7, 0.003),
            (5, 15, 0.003),
        )
    )
    shots, steps = (32, 2) if smoke else (16384, 200)
    config = {
        "mode": "smoke" if smoke else "prospective",
        "seeds": seeds,
        "grid": grid,
        "shots_per_cell": shots,
        "steps": steps,
        "train": (3, 3, 0.02),
        "validation": (3, 3, 0.006),
        "thresholds": (0.5, 0.9, 1.0),
        "primary": (5, 5, 0.003),
        "variants": ("mlp", "cnn", "cnn_no_types"),
        "basis": "z",
        "batch_size": 128,
        "primary_alpha_per_model_seed": 0.05 / 9,
    }
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for kind in config["variants"]:
            torch.manual_seed(seed)
            trained = make_model(kind, 3, 3, 0.02)
            trainer = Trainer(trained, seed=seed + 104729)
            losses = trainer.train(steps)
            validation = make_model(kind, 3, 3, 0.006)
            transfer(trained, validation)
            dem = DetectorErrorModel.from_stim_text(validation.dem_text)
            decoder = get_decoder("pymatching", dem)
            candidates = []
            for threshold in (0.5, 0.9, 1.0):
                score = assess(
                    validation,
                    decoder,
                    shots=shots,
                    seed=seed + 1000000,
                    threshold=threshold,
                )
                candidates.append((score.hybrid_failures, -threshold))
            threshold = -min(candidates)[1]
            for cell, (distance, rounds, probability) in enumerate(grid):
                model = make_model(kind, distance, rounds, probability)
                transfer(trained, model)
                backend = get_decoder(
                    "pymatching", DetectorErrorModel.from_stim_text(model.dem_text)
                )
                test_seed = seed + 2000000 + cell * 10000
                counts = assess(
                    model, backend, shots=shots, seed=test_seed, threshold=threshold
                )
                primary = (distance, rounds, probability) == (5, 5, 0.003)
                row: dict[str, object] = {
                    "seed": seed,
                    "kind": kind,
                    "distance": distance,
                    "rounds": rounds,
                    "probability": probability,
                    "test_seed": test_seed,
                    "threshold": threshold,
                    "first_loss": losses[0],
                    "last_loss": losses[-1],
                    "primary": primary,
                    "counts": asdict(counts),
                    "statistics": summarize(
                        counts, alpha=0.05 / 9 if primary else 0.05
                    ),
                }
                if primary or smoke:
                    row["timing"] = block_timings(
                        model,
                        backend,
                        seed=test_seed + 5000000,
                        threshold=threshold,
                        repetitions=8 if smoke else 128,
                    )
                rows.append(row)
                print(
                    f"seed={seed} model={kind} cell={cell} baseline={counts.baseline_failures} hybrid={counts.hybrid_failures}",
                    file=sys.stderr,
                    flush=True,
                )
    source_root = Path(__file__).resolve().parents[2]
    source_hashes = {
        name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in (
            "examples/qec/assess_predecoder.py",
            "flagquantum/qec/decoders/_ai/assessment.py",
            "flagquantum/qec/decoders/_ai/predecoder.py",
            "flagquantum/qec/decoders/_ai/training.py",
            "flagquantum/qec/decoders/_ai/spacetime.py",
        )
    }
    configuration_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    return {
        "schema": "qec_predecoder_assessment_v1",
        "source_sha256": source_hashes,
        "configuration": config,
        "configuration_sha256": configuration_hash,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
            "pymatching": version("pymatching"),
        },
        "claim": (
            "smoke_only"
            if smoke
            else "inspect_primary_intervals_no_automatic_promotion"
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Exclusive creation prevents silently replacing a frozen study record.
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(run(args.smoke), output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
