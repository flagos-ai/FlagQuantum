"""Workshop counterpart of the same independent two-asset reference."""

from examples.synthesis.applications.finance.rainbow_options.rainbow_options_bruteforce_method.comparisons.circuit import (
    preparation_matrix,
    reference_circuit,
)

__all__ = ["preparation_matrix", "reference_circuit"]


def run_example() -> None:
    import torch

    import flagquantum as fq

    from ..main import build_program

    for k in (0, 1, 2):
        program = build_program(k)
        torch.testing.assert_close(
            fq.run(program).statevector().flatten(),
            fq.run(reference_circuit(k)).statevector().flatten(),
            atol=1e-11,
            rtol=0,
        )
        print(k, program.report["resources"])


if __name__ == "__main__":
    run_example()
