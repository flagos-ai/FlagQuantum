"""Compare a recorded distributed-MPS Heisenberg gradient with statevector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import flagquantum.algorithms as fqa
from flagquantum.ops import set_global_precision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    set_global_precision(torch.complex128)

    parameter_count = int(payload["parameter_count"])
    values: list[float | None] = [None] * parameter_count
    gradients: list[float | None] = [None] * parameter_count
    for record in payload["rank_records"]:
        step = record["training"]["step_metrics"][0]
        for index, value in step["parameter_values"]:
            values[int(index)] = float(value)
        for index, value in step["parameter_gradients"]:
            gradients[int(index)] = float(value)
    if any(value is None for value in values + gradients):
        raise RuntimeError(
            "artifact does not contain complete owner-recorded gradients"
        )

    parameters = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    hamiltonian = fqa.heisenberg_chain_hamiltonian(int(payload["n_sites"]))
    circuit = fqa.heisenberg_hva(
        int(payload["n_sites"]),
        int(payload["ansatz_depth"]),
        parameters,
        parameterization=str(payload["parameterization"]),
        initial_state="dimer_singlet",
    )
    energy = hamiltonian.expectation(circuit).sum()
    reference = torch.autograd.grad(energy, parameters)[0]
    distributed = torch.tensor(gradients, dtype=torch.float64)
    difference = distributed - reference
    per_layer = parameter_count // int(payload["ansatz_depth"])
    family_summaries = {}
    family_ranges = {
        "rxx": (0, int(payload["n_sites"]) - 1),
        "ryy": (int(payload["n_sites"]) - 1, 2 * (int(payload["n_sites"]) - 1)),
        "rzz": (2 * (int(payload["n_sites"]) - 1), 3 * (int(payload["n_sites"]) - 1)),
        "rz_phase": (3 * (int(payload["n_sites"]) - 1), per_layer),
    }
    for family, (start, stop) in family_ranges.items():
        indices = torch.tensor(
            [
                layer * per_layer + offset
                for layer in range(int(payload["ansatz_depth"]))
                for offset in range(start, stop)
            ]
        )
        actual = distributed[indices]
        expected = reference[indices]
        family_summaries[family] = {
            "relative_l2_difference": float(
                torch.linalg.vector_norm(actual - expected)
                / torch.linalg.vector_norm(expected).clamp_min(1e-30)
            ),
            "cosine_similarity": float(
                torch.dot(actual, expected)
                / (
                    torch.linalg.vector_norm(actual)
                    * torch.linalg.vector_norm(expected)
                ).clamp_min(1e-30)
            ),
        }
    print(
        json.dumps(
            {
                "distributed_energy": payload["optimization_trace"][0]["energy"],
                "statevector_energy": float(energy.detach()),
                "energy_absolute_difference": abs(
                    float(payload["optimization_trace"][0]["energy"])
                    - float(energy.detach())
                ),
                "gradient_max_absolute_difference": float(difference.abs().max()),
                "gradient_relative_l2_difference": float(
                    torch.linalg.vector_norm(difference)
                    / torch.linalg.vector_norm(reference).clamp_min(1e-30)
                ),
                "gradient_cosine_similarity": float(
                    torch.dot(distributed, reference)
                    / (
                        torch.linalg.vector_norm(distributed)
                        * torch.linalg.vector_norm(reference)
                    ).clamp_min(1e-30)
                ),
                "gradient_family_summaries": family_summaries,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
