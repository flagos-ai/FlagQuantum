"""Compare native FlagQuantum and independent Circuit amplitudes."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.state_preparation.prepare_ghz_state.main import main
from flagquantum import synth


def reference_circuit() -> fq.Circuit:
    circuit = fq.Circuit(5, dtype=torch.complex128).h(0)
    for target in range(1, 5):
        circuit.cx(0, target)
    return circuit


def run_example() -> None:
    compare_states(
        synth.synthesize(main, memory_limit_bytes=512 * 1024**2),
        reference_circuit(),
        ("x",),
    )


if __name__ == "__main__":
    run_example()
