"""Check the flexible source's native program against exact Bell-pair labels."""

from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.qpe.comparisons.circuit import reference_circuit
from examples.synthesis.functions.qpe.qpe_flexible.main import build_program


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("state", "phase"))


if __name__ == "__main__":
    run_example()
