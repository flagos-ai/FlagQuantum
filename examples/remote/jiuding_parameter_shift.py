"""Run one batched parameter-shift update on a warm Jiuding executor."""

import argparse
import time

import torch

import flagquantum as fq
from flagquantum.gradients import batched_parameter_shift_gradient
from flagquantum.remote.compute import JiudingClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    args = parser.parse_args()
    parameters = torch.tensor([0.4], dtype=torch.float32)
    output = fq.expectation(fq.Z(0))
    batch_calls = 0
    batch_remote_seconds = 0.0

    def build(values: torch.Tensor) -> fq.Circuit:
        return fq.Circuit(1).ry(0, theta=values[0])

    with JiudingClient(workspace=args.workspace) as client:

        def evaluate_batch(
            circuits: tuple[fq.Circuit, ...],
        ) -> tuple[torch.Tensor, ...]:
            nonlocal batch_calls, batch_remote_seconds
            batch_calls += 1
            results = client.run_batch(
                circuits,
                target=args.target,
                outputs=output,
            )
            batch_remote_seconds = float(results[0].runtime["batch_elapsed_seconds"])
            return tuple(result.expectation().sum() for result in results)

        before = client.run(
            build(parameters), target=args.target, outputs=output
        ).expectation()
        gradient_started = time.perf_counter()
        gradient = batched_parameter_shift_gradient(
            build,
            parameters,
            evaluate_batch,
        )
        gradient_seconds = time.perf_counter() - gradient_started
        updated = parameters - args.learning_rate * gradient
        after = client.run(
            build(updated), target=args.target, outputs=output
        ).expectation()

    print(
        {
            "energy_before": float(before),
            "gradient": gradient.tolist(),
            "gradient_batch_calls": batch_calls,
            "gradient_batch_size": 2 * parameters.numel(),
            "gradient_remote_seconds": batch_remote_seconds,
            "gradient_total_seconds": gradient_seconds,
            "parameters_after": updated.tolist(),
            "energy_after": float(after),
        }
    )


if __name__ == "__main__":
    main()
