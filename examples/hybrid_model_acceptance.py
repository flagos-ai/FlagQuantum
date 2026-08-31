"""Maintained tiny classifier/energy workflows for Phase-2 acceptance."""

from __future__ import annotations

import argparse
import json
import time

import torch

import flagquantum as fq


def acceptance_summary(
    *,
    model: str,
    policy: dict[str, object],
    correctness: dict[str, object],
    accuracy: dict[str, object],
    performance: dict[str, object],
    deployment: dict[str, object],
) -> dict[str, object]:
    """Build the public, fail-closed acceptance payload used by this example."""
    return {
        "model": model,
        "policy": policy,
        "correctness": correctness,
        "accuracy": accuracy,
        "performance": performance,
        "deployment": deployment,
        "accuracy_is_performance_claim": False,
        "scalability_claim_allowed": False,
    }


def classifier_run(steps: int) -> dict[str, object]:
    fq.seed_everything(480)
    model = fq.HybridQuantumClassifier(
        deployment_binding={"provider": "local", "target": "simulator"}
    )
    inputs = torch.tensor([[-1.0, -0.5], [-0.7, 0.8], [0.6, -0.9], [0.9, 0.7]])
    targets = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    losses = []
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    elapsed = time.perf_counter() - started
    accuracy = float((torch.sign(model(inputs).detach()) == targets).float().mean())
    return acceptance_summary(
        model="HybridQuantumClassifier",
        policy=model.quantum.policy.to_dict(),
        correctness={"finite": bool(torch.isfinite(torch.tensor(losses)).all())},
        accuracy={"classification_accuracy": accuracy, "losses": losses},
        performance={"elapsed_seconds": elapsed, "steps": steps},
        deployment=model.deployment_parameters()["binding"],
    )


def energy_run(steps: int, backend: str) -> dict[str, object]:
    fq.seed_everything(481)
    policy = fq.RuntimePolicy(
        execution_options=fq.ExecutionOptions(
            backend=backend, allow_backend_fallback=False
        ),
        observable="hamiltonian",
        observable_wires=(0, 1),
    )
    model = fq.VariationalEnergyModel(policy=policy)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    energies = []
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad()
        energy = model()
        energy.backward()
        optimizer.step()
        energies.append(float(energy.detach()))
    elapsed = time.perf_counter() - started
    runtime = model.quantum.execute().runtime
    return acceptance_summary(
        model="VariationalEnergyModel",
        policy=policy.to_dict(),
        correctness={"finite": bool(torch.isfinite(torch.tensor(energies)).all())},
        accuracy={"energies": energies},
        performance={"elapsed_seconds": elapsed, "steps": steps},
        deployment={"runtime_backend": runtime.get("backend", backend)},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=("classifier", "energy"), default="classifier"
    )
    parser.add_argument("--backend", choices=("pytorch", "jax"), default="pytorch")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    report = (
        classifier_run(args.steps)
        if args.model == "classifier"
        else energy_run(args.steps, args.backend)
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
