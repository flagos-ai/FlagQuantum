"""Run the full original-degree native plasma inversion workflow."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from . import plots, workflow


def report(result: workflow.Result) -> None:
    value = result.model
    print(
        f"Original grid=8x8, data amplitudes=128, degree=251, phases={len(value.phases)}"
    )
    print(
        f"Raw inverse minimum singular value kappa={value.kappa}, block alpha={value.normalization}, inverse alpha={value.inverse_normalization}"
    )
    print(
        f"Actual width={result.resources.n_qubits}, operations={result.resources.n_operations}, basis={result.resources.basis}"
    )
    print(
        f"Fresh phase fitting residual={value.fitted.residual}, iterations={value.fitted.iterations}"
    )
    print(
        f"Full complex state error={result.full_state_error}, polynomial projection error={result.polynomial_error}"
    )
    print(
        f"Actual projection probability={result.projection_probability}, original polynomial linear-solution L2 error={result.linear_solution_error}"
    )


def write(result: workflow.Result, destination: Path) -> None:
    value = result.model
    data = dict(
        degree=251,
        phase_count=len(value.phases),
        kappa=value.kappa,
        alpha=value.normalization,
        inverse_alpha=value.inverse_normalization,
        fit_residual=value.fitted.residual,
        full_state_error=result.full_state_error,
        polynomial_error=result.polynomial_error,
        linear_solution_error=result.linear_solution_error,
        projection_probability=result.projection_probability,
        resources=asdict(result.resources),
        quantum_field=[
            [complex(v).real, complex(v).imag] for v in result.quantum_field
        ],
        classical_field=[
            [complex(v).real, complex(v).imag] for v in result.classical_field
        ],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("vlasov-inverse-figures"))
    parser.add_argument(
        "--results", type=Path, default=Path("vlasov-inverse-results.json")
    )
    args = parser.parse_args()
    result = workflow.run()
    report(result)
    plots.save(result, args.figures)
    write(result, args.results)


if __name__ == "__main__":
    main()
