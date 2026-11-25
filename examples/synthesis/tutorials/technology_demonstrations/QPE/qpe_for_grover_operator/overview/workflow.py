"""Compare complete controlled Grover states and measure actual target resources."""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

import torch

import flagquantum as fq
from examples.synthesis.qpe_reference import dense_qpe
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.classiq_qsvt.overview.exact_basis import (
    exact_rules,
)
from examples.synthesis.tutorials.technology_demonstrations.QPE.qpe_for_grover_operator.comparisons.circuit import (
    operators,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate, estimate_resources
from flagquantum.compiler.basis_conversion import convert_basis


@dataclass(frozen=True)
class StudyPoint:
    precision: int
    objective: Literal["width", "depth"]
    maximum_state_error: float
    converted_state_error: float
    named_probability_error: float
    retained_data_nonzero: float
    resources: ResourceEstimate


def verify_program(
    program: synth.Program, precision: int, objective: Literal["width", "depth"]
) -> StudyPoint:
    matrix, preparation, _ = operators()
    expected = fq.run(dense_qpe(matrix, preparation, precision)).statevector()
    result = fq.run(program)
    actual = result.statevector()
    assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
    error = float((actual.flatten() - expected.flatten()).abs().max())
    executable = convert_basis(
        program.circuit,
        gates=("cx", "u3"),
        max_added_operations=500000,
        rules=exact_rules(),
    ).program
    converted = fq.run(executable).statevector()
    assert isinstance(converted, torch.Tensor)
    converted_error = float((converted.flatten() - expected.flatten()).abs().max())
    probabilities = expected.reshape(1 << precision, 32).abs().square().sum(dim=1)
    named = program.numeric_probabilities(result, registers=("phase_result",))
    named_error = max(
        abs(
            float(probabilities[word])
            - named.get((Fraction(word, 1 << precision),), 0.0)
        )
        for word in range(1 << precision)
    )
    retained = float(actual.reshape(1 << precision, 32)[:, 1:].abs().square().sum())
    resources = estimate_resources(executable, gates=("cx", "u3"))
    assert error < 2e-11 and converted_error < 2e-10 and named_error < 2e-11
    assert resources.n_qubits == precision + 5
    return StudyPoint(
        precision, objective, error, converted_error, named_error, retained, resources
    )


def plot_study(points: tuple[StudyPoint, ...], directory: Path) -> Path:
    import matplotlib.pyplot as plt

    if len(points) != 10 or {p.precision for p in points} != set(range(1, 6)):
        raise ValueError(
            "retain both objectives and every precision from one through five"
        )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    historical_depths = (3094, 9262, 21596, 46254, 95560)
    historical_widths = (6, 7, 8, 9, 10)
    figure, axis = plt.subplots(figsize=(9, 5))
    for objective, marker in (("width", "o"), ("depth", "x")):
        subset = tuple(point for point in points if point.objective == objective)
        axis.semilogy(
            [p.precision for p in subset],
            [p.resources.depth for p in subset],
            "-" + marker,
            label="Native " + objective + " objective (measured)",
        )
        if objective == "width":
            for point in subset:
                axis.annotate(
                    str(point.resources.n_qubits),
                    (point.precision, point.resources.depth),
                    xytext=(-7, 7),
                    textcoords="offset points",
                    ha="right",
                    fontsize=8,
                )
    axis.semilogy(
        range(1, 6),
        historical_depths,
        "--s",
        label="Qiskit depth (source historical data)",
    )
    for precision, depth, width in zip(
        range(1, 6), historical_depths, historical_widths, strict=True
    ):
        axis.annotate(
            str(width),
            (precision, depth),
            xytext=(7, -10),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("QPE precision (bits)")
    axis.set_ylabel("CX/U3 depth")
    axis.set_xticks(range(1, 6))
    axis.legend(fontsize=9)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    path = directory / "precision_resources.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
