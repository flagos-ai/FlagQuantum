"""One VQE training loop, switchable statevector/MPS/tensor-network execution.

Examples
--------
Run all backends and certify trajectory parity::

    python examples/vqe_switch_sv_mps_tn.py --backend all --plot

Compare advanced backend-native execution without changing the training loop::

    FQ_VQE_BACKEND=mps python examples/vqe_switch_sv_mps_tn.py
    FQ_VQE_BACKEND=tn  python examples/vqe_switch_sv_mps_tn.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

import flagquantum as fq
import flagquantum.backends as fqb

BACKEND_ALIASES = {
    "sv": "statevector",
    "statevector": "statevector",
    "mps": "mps",
    "tn": "tensor_network",
    "tensor_network": "tensor_network",
}


def build_ansatz(parameters: torch.Tensor) -> fq.Circuit:
    """Backend-independent VQE ansatz."""
    depth, n_wires, _ = parameters.shape
    circuit = fq.Circuit(n_wires, device=parameters.device)
    for wire in range(n_wires):
        circuit.h(wire)
    for layer in range(depth):
        for wire in range(n_wires):
            circuit.ry(wire, parameters[layer, wire, 0])
            circuit.rz(wire, parameters[layer, wire, 1])
        for wire in range(n_wires - 1):
            circuit.cx(wire, wire + 1)
    return circuit


def execute(circuit: fq.Circuit, backend: str, *, max_bond: int) -> Any:
    """Return a backend-native object for this explicit parity experiment.

    This advanced example intentionally uses ``flagquantum.backends.run_native`` because it
    compares backend-specific expectation methods. Normal applications should
    use ``fq.run`` and the stable ``fq.ExecutionResult`` contract.
    """
    options = {"max_bond": max_bond} if backend == "mps" else {}
    return fqb.run_native(circuit, mode=backend, **options)


def expectation(
    target: Any, n_wires: int, *, x: tuple[int, ...] = (), z: tuple[int, ...] = ()
) -> torch.Tensor:
    if hasattr(target, "expectation_ps"):
        return target.expectation_ps(x=x, z=z).sum()
    state_circuit = fq.Circuit(n_wires, device=target.device, inputs=target)
    return state_circuit.expectation_ps(x=x, z=z).sum()


def vqe_energy(
    parameters: torch.Tensor, *, backend: str, max_bond: int
) -> torch.Tensor:
    """Open-boundary transverse-field Ising energy."""
    target = execute(build_ansatz(parameters), backend, max_bond=max_bond)
    n_wires = parameters.shape[1]
    energy = sum(
        -expectation(target, n_wires, z=(wire, wire + 1)) for wire in range(n_wires - 1)
    )
    energy = energy + sum(
        -0.7 * expectation(target, n_wires, x=(wire,)) for wire in range(n_wires)
    )
    return energy


def train_vqe(
    initial: torch.Tensor, *, backend: str, steps: int, lr: float, max_bond: int
) -> dict[str, Any]:
    """This training loop is identical for SV, MPS, and TN."""
    parameters = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([parameters], lr=lr)
    energies = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = vqe_energy(parameters, backend=backend, max_bond=max_bond)
        loss.backward()
        optimizer.step()
        energies.append(float(loss.detach().cpu()))
    return {
        "backend": backend,
        "energies": energies,
        "final_energy": energies[-1],
        "parameters": parameters.detach().cpu().tolist(),
    }


def plot_results(
    results: list[dict[str, Any]],
    output: Path,
    *,
    n_wires: int,
    depth: int,
    steps: int,
    max_bond: int,
) -> None:
    import matplotlib.pyplot as plt

    labels = {"statevector": "SV", "mps": "MPS", "tensor_network": "TN"}
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
    styles = {
        "statevector": {"marker": "o", "markevery": (0, 6), "linestyle": "-"},
        "mps": {"marker": "s", "markevery": (2, 6), "linestyle": "--"},
        "tensor_network": {"marker": "^", "markevery": (4, 6), "linestyle": ":"},
    }
    for result in results:
        left.plot(
            range(1, len(result["energies"]) + 1),
            result["energies"],
            label=labels[result["backend"]],
            linewidth=2,
            markersize=5,
            **styles[result["backend"]],
        )
    reference = next(result for result in results if result["backend"] == "statevector")
    names, errors = [], []
    for result in results:
        names.append(labels[result["backend"]])
        errors.append(
            max(abs(a - b) for a, b in zip(reference["energies"], result["energies"]))
        )
    left.set_xlabel("Optimizer step")
    left.set_ylabel("VQE energy")
    left.set_title("One training loop, three simulation methods")
    left.grid(alpha=0.25)
    left.legend()
    right.bar(
        names,
        [max(value, 1e-12) for value in errors],
        color=["#2468b4", "#d24b40", "#238b45"],
    )
    right.set_yscale("log")
    right.set_ylabel("Max trajectory error vs SV")
    right.set_title("Training precision alignment")
    right.grid(alpha=0.25, axis="y", which="both")
    fig.suptitle(
        f"VQE backend switching · N={n_wires} qubits · depth={depth} · {steps} Adam steps\n"
        f"SV exact · MPS max_bond={max_bond}, no truncation · TN greedy, unsliced · complex64",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default=os.getenv("FQ_VQE_BACKEND", "all"),
        choices=("all", *BACKEND_ALIASES),
    )
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--max-bond", type=int, default=16)
    parser.add_argument("--seed", type=int, default=260720)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("vqe_backend_switch.json"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    initial = 0.05 * torch.randn(args.depth, args.n_qubits, 2, device=args.device)
    backends = (
        ("statevector", "mps", "tensor_network")
        if args.backend == "all"
        else (BACKEND_ALIASES[args.backend],)
    )
    results = [
        train_vqe(
            initial,
            backend=backend,
            steps=args.steps,
            lr=args.lr,
            max_bond=args.max_bond,
        )
        for backend in backends
    ]
    payload = {
        "schema": "flagquantum.vqe_backend_switch.v1",
        "training_code_changed": False,
        "n_qubits": args.n_qubits,
        "depth": args.depth,
        "steps": args.steps,
        "seed": args.seed,
        "device": args.device,
        "max_bond": args.max_bond,
        "results": results,
    }
    if len(results) == 3:
        reference = results[0]["energies"]
        payload["precision_alignment"] = {
            result["backend"]: {
                "max_energy_trajectory_abs_error_vs_sv": max(
                    abs(a - b) for a, b in zip(reference, result["energies"])
                ),
                "final_energy_abs_error_vs_sv": abs(
                    reference[-1] - result["final_energy"]
                ),
            }
            for result in results
        }
        payload["precision_aligned"] = all(
            item["max_energy_trajectory_abs_error_vs_sv"] <= 1e-5
            for item in payload["precision_alignment"].values()
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.plot and len(results) == 3:
        plot_results(
            results,
            args.output.with_suffix(""),
            n_wires=args.n_qubits,
            depth=args.depth,
            steps=args.steps,
            max_bond=args.max_bond,
        )
    print(
        json.dumps(
            {
                "final_energy": {r["backend"]: r["final_energy"] for r in results},
                "precision_aligned": payload.get("precision_aligned"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
