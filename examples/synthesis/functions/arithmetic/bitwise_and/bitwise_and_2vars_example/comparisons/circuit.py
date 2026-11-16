"""Compare native two-register AND against independent basis preparation."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.arithmetic.bitwise_and.bitwise_and_2vars_example.main import (
    main,
)
from flagquantum.synth import synthesize


def reference_circuit() -> fq.Circuit:
    return fq.Circuit(9, dtype=torch.complex128).x(0).x(3).x(5).x(6)


def run_example() -> None:
    compare_states(synthesize(main), reference_circuit(), ("a", "b", "res"))


if __name__ == "__main__":
    run_example()
