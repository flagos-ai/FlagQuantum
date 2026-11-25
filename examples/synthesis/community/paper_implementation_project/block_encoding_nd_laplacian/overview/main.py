"""Run both source periodic-Laplacian workflows without Qmod input."""

from flagquantum.drawer import draw

from .workflow import run_case


def run_example() -> None:
    for multidimensional in (False, True):
        result = run_case(multidimensional=multidimensional)
        print("Grid:", (2, 4) if multidimensional else (32,))
        print("Resources:", result.resources)
        print(draw(result.program.circuit, show_all_qubits=True))
        print("Unnormalized complex block:", result.projected)
        print("Source phase-aligned display:", result.source_display)
        print("Expected Laplacian column:", result.expected)
        print("Projection probability:", result.projection_probability)
        print("Absolute full-state error:", result.full_state_error)
        print("Absolute block error:", result.block_error)
        print("Normalized source fidelity:", result.fidelity)


if __name__ == "__main__":
    run_example()
