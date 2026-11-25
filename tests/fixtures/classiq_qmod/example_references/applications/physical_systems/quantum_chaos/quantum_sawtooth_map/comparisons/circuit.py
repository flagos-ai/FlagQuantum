"""Compare the frozen source with independently specified unitary matrices."""

from pathlib import Path

from examples.synthesis.applications.physical_systems.quantum_chaos.quantum_sawtooth_map.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth


def run_example() -> None:
    compare_states(
        synth.synthesize(
            Path(__file__).with_name("source.qmod"),
            classical_inputs={"kick_mag": 0.43, "num_kicks": 2},
        ),
        reference_circuit(),
        ("p",),
    )


if __name__ == "__main__":
    run_example()
