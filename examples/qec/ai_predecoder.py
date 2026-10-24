"""Train, checkpoint/resume and evaluate a CPU surface-code predecoder.

Run with PYTHONPATH=. python examples/qec/ai_predecoder.py --steps 20 --shots 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flagquantum.qec import (
    DetectorErrorModel,
    PhenomenologicalNoise,
    RotatedSurfaceCode,
    build_memory_circuit,
    get_decoder,
)
from flagquantum.qec.decoders._neural import Predecoder, Trainer, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--checkpoint", type=Path, default=Path("predecoder.pt"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument(
        "--backend",
        choices=("minimum_weight_matching", "pymatching"),
        default="minimum_weight_matching",
    )
    args = parser.parse_args()
    torch.manual_seed(0)
    dem = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(RotatedSurfaceCode(distance=3), rounds=3),
        noise=PhenomenologicalNoise(data_flip=0.02, measurement_flip=0.02),
    )
    model = Predecoder(dem)
    trainer = Trainer(model, seed=1)
    if args.resume or args.evaluate_only:
        trainer.restore(args.checkpoint)
    losses = [] if args.evaluate_only else trainer.train(args.steps)
    if not args.evaluate_only:
        trainer.save(args.checkpoint)
    report = evaluate(
        model,
        get_decoder(args.backend, dem),
        shots=args.shots,
        seed=1001,
        threshold=args.threshold,
    )
    print(
        json.dumps(
            {
                "training_step": trainer.step,
                "first_loss": losses[0] if losses else None,
                "last_loss": losses[-1] if losses else None,
                "evaluation_seed": 1001,
                **report,
            }
        )
    )


if __name__ == "__main__":
    main()
