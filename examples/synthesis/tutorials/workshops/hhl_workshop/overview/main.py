"""Run the original four-phase-bit HHL workshop and solution diagnostics."""

from pathlib import Path

import flagquantum as fq
from flagquantum import synth

from .ansatz import build_program, load_b
from .physics import AMPLITUDES, MATRIX, B
from .workflow import extract_solution, plot_solution, validate_problem, verify_program


def run_example() -> None:
    validate_problem(MATRIX, B)
    preparation = synth.synthesize(
        load_b, classical_inputs={"amplitudes": list(AMPLITUDES)}
    )
    print("Preparation:", verify_program(preparation, preparation_only=True))
    print("Prepared amplitudes:", fq.run(preparation).statevector())
    program = build_program()
    print("HHL:", verify_program(program))
    print(program.report["resources"])
    solution = extract_solution(program)
    print("Rounded quantum solution:", solution.rounded_quantum)
    print("Classical solution:", solution.classical)
    print("Normalized solution fidelity:", solution.fidelity)
    print("Postselection probability:", solution.postselection_probability)
    print(
        "Retained phase leakage probability:",
        solution.retained_phase_leakage_probability,
    )
    print(
        "Source magnitude relative distance (%):",
        solution.magnitude_relative_distance_percent,
    )
    print("Complex relative distance:", solution.complex_relative_distance)
    print(plot_solution(solution, Path("artifacts/development/hhl_workshop")))


if __name__ == "__main__":
    run_example()
