"""Compare native arithmetic with an independently constructed Circuit."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.arithmetic.bitwise_and.bitwise_and_integer_example.main import (
    main,
)
from flagquantum.synth import synthesize


def reference_circuit() -> fq.Circuit:
    return fq.Circuit(5, dtype=torch.complex128).x(0).x(2).x(4)


def run_example() -> None:
    compare_states(synthesize(main), reference_circuit(), ("a", "res"))


if __name__ == "__main__":
    run_example()
