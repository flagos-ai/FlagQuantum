"""Compare native FlagQuantum and independent Circuit amplitudes."""

import torch

import flagquantum as fq
from examples.synthesis.community.paper_implementation_project.classical_shadows.main import (
    main,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth


def reference_circuit() -> fq.Circuit:
    return fq.Circuit(2, dtype=torch.complex128).h(0).z(0).cx(0, 1)


def run_example() -> None:
    compare_states(synth.synthesize(main), reference_circuit(), ("qarr",))


if __name__ == "__main__":
    run_example()
