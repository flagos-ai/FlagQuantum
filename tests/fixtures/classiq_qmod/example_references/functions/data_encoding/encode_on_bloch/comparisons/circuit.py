"""Check the pinned original encoding against independent rotations."""

from pathlib import Path

from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.data_encoding.encode_on_bloch.comparisons.circuit import (
    reference_circuit,
)
from flagquantum.synth import synthesize


def run_example() -> None:
    data = [0.2, -0.3, 0.4, 0.5, 0.6, -0.7, 0.8]
    program = synthesize(
        Path(__file__).with_name("source.qmod"), classical_inputs={"data": data}
    )
    compare_states(program, reference_circuit(data), ("x",))


if __name__ == "__main__":
    run_example()
