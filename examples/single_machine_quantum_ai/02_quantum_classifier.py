"""Tiny quantum classifier trained locally with PyTorch autograd."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.nn import functional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import (  # noqa: E402
    print_section,
    print_training_summary,
    time_forward,
)

import flagquantum as fq  # noqa: E402

TEACHER_WEIGHTS = torch.tensor([0.7, -0.4, 1.1, -0.8], dtype=torch.float32)


def make_dataset(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.linspace(-1.0, 1.0, steps=7, device=device)
    points = torch.cartesian_prod(values, values)
    teacher = TEACHER_WEIGHTS.to(device=device)
    with torch.no_grad():
        teacher_logits = batch_logits(points, teacher)
        labels = (teacher_logits > 0).to(torch.float32)
    return points, labels, teacher


def model_logit(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    circuit = fq.Circuit(2, device=features.device)
    circuit.ry(0, theta=1.4 * features[0])
    circuit.ry(1, theta=1.4 * features[1])
    circuit.rz(0, theta=weights[0])
    circuit.rz(1, theta=weights[1])
    circuit.cx(0, 1)
    circuit.ry(0, theta=weights[2])
    circuit.ry(1, theta=weights[3])
    circuit.cx(1, 0)
    return 3.0 * circuit.expectation_z(0).reshape(())


def batch_logits(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.stack([model_logit(row, weights) for row in features])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bench-iters", type=int, default=5)
    parser.add_argument("--compare-torch", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(11)
    x, y, teacher_weights = make_dataset(args.device)
    weights = (0.1 * torch.randn(4, device=args.device)).requires_grad_(True)
    optimizer = torch.optim.Adam([weights], lr=args.lr)
    teacher_logits = batch_logits(x, teacher_weights)
    teacher_loss = functional.binary_cross_entropy_with_logits(teacher_logits, y)
    for step in range(args.steps):
        optimizer.zero_grad()
        logits = batch_logits(x, weights)
        loss = functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()
        if (
            step == 0
            or step == args.steps - 1
            or (step + 1) % max(1, args.steps // 5) == 0
        ):
            accuracy = ((torch.sigmoid(logits) > 0.5) == y.bool()).float().mean()
            print(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "accuracy": float(accuracy.detach()),
                }
            )

    final_logits = batch_logits(x, weights.detach())
    final_accuracy = ((torch.sigmoid(final_logits) > 0.5) == y.bool()).float().mean()
    native_speed = None
    if args.compare_torch:
        native_speed = time_forward(
            lambda theta: batch_logits(x, theta),
            weights.detach(),
            iters=args.bench_iters,
            device=args.device,
        )
    print_training_summary(
        title="Single-Machine Quantum Classifier",
        example="single_machine_quantum_classifier",
        metrics={
            "theoretical_solution": "teacher quantum circuit",
            "training_backend": "pytorch",
            "teacher_loss": float(teacher_loss.detach()),
            "teacher_accuracy": 1.0,
            "final_loss": float(
                functional.binary_cross_entropy_with_logits(final_logits, y).detach()
            ),
            "final_accuracy": float(final_accuracy),
            "trained_weight_norm": float(weights.detach().norm()),
        },
        speed_compare={
            "pytorch_native_forward": native_speed
            if native_speed is not None
            else {"status": "unavailable", "reason": "run with --compare-torch"},
            "jax_kernel_forward": {
                "status": "see examples/single_machine_quantum_ai/04_jax_kernel_torch_layer.py"
            },
        },
    )
    print_section("Known Teacher Parameters")
    print(
        "  teacher_weights :",
        [round(float(value), 6) for value in teacher_weights.detach().cpu()],
    )


if __name__ == "__main__":
    main()
