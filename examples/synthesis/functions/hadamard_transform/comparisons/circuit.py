"""Compare native FlagQuantum and independent Circuit amplitudes."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.hadamard_transform.main import main
from flagquantum import synth


def reference_circuit() -> fq.Circuit:
    return fq.Circuit(3, dtype=torch.complex128).h(0).h(1).h(2)


def run_example() -> None:
    compare_states(synth.synthesize(main), reference_circuit(), ("x",))


if __name__ == "__main__":
    run_example()
