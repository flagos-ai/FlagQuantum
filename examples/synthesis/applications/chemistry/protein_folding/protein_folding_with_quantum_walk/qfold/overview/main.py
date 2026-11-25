"""Run the alanylalanine QFold walkthrough and print the observed counts."""

from collections.abc import Mapping
from typing import cast

import torch

import flagquantum as fq
from flagquantum.synth import Program

from . import reference, workflow


def describe(data: workflow.Dataset) -> None:
    print(f"Protein: {data.protein}")
    print(f"Rotation bits per angle: {data.rotation_bits}")
    print(f"Move-id bits: {data.move_bits}")
    print(f"Logical and environment bits: {data.encoding_width}")
    print(f"Phi and psi starting angles: {data.phi_angles}, {data.psi_angles}")
    print(f"Archived Psi4 reference energy: {data.psi4_min_energy}")


def verify(program: Program, angles: tuple[float, float]) -> None:
    result = fq.run(program)
    actual = cast(torch.Tensor, result.statevector()).reshape(16, -1)
    expected = reference.complete_state(angles, program.circuit.n_wires)
    print(f"Resources: {program.report['resources']}")
    print(f"Complete state error: {float((actual - expected).abs().max()):.3g}")
    print(
        f"Predicate workspace probability: {float(actual[:, 1:].abs().square().sum()):.3g}"
    )
    print(
        f"Retained nonzero coin/move probability: {float(actual.reshape(4, 4, -1)[:, 1:].abs().square().sum()):.9f}"
    )
    probabilities = dict(program.probabilities(result, registers=("phi", "psi")))
    print(f"Exact phi/psi probabilities: {probabilities}")


def print_counts(counts: Mapping[tuple[int, int], int]) -> None:
    total = sum(counts.values())
    print(f"Observed phi/psi counts ({total} draws): {dict(sorted(counts.items()))}")


def main() -> None:
    data = workflow.dataset()
    print(f"Energy differences: {data.energy_differences}")
    describe(data)
    angles = workflow.coin_angles(data, beta=1.0)
    print(f"Coin rotations by move (radians): {angles}")
    program = workflow.build(angles)
    verify(program, angles)
    print_counts(workflow.sample(program))
    print(f"Source 01-prefix energy lookup: {workflow.energy_lookup(data)}")


if __name__ == "__main__":
    main()
