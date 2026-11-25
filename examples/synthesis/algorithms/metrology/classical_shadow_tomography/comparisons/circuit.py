"""Compare native FlagQuantum and independent Circuit amplitudes."""

import torch

import flagquantum as fq
from examples.synthesis.algorithms.metrology.classical_shadow_tomography.main import (
    main,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth


def reference_circuit() -> fq.Circuit:
    circuit = fq.Circuit(2, dtype=torch.complex128).h(0).z(0).cx(0, 1)
    circuit.s(1).h(1).s(0).s(0).s(1).h(0)
    circuit.cx(1, 0).cx(1, 0).s(1).h(0).s(1).h(1).s(1).s(1)
    circuit.cx(0, 1).h(0).h(0).s(1).s(1).s(1)
    return circuit


def run_example() -> None:
    compare_states(synth.synthesize(main), reference_circuit(), ("qarr",))


if __name__ == "__main__":
    run_example()
