"""Compare native FlagQuantum and independent Circuit amplitudes."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.basics.entanglement.main import main
from flagquantum import synth


def reference_circuit() -> fq.Circuit:
    circuit = fq.Circuit(6, dtype=torch.complex128).h(5)
    for target in range(4, -1, -1):
        circuit.cx(target + 1, target)
    return circuit


def run_example() -> None:
    compare_states(synth.synthesize(main), reference_circuit(), ("reg",))


if __name__ == "__main__":
    run_example()
