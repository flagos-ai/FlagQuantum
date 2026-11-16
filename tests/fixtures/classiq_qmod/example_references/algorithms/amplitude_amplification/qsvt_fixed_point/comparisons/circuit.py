"""Validate the untouched fixed-point source against independent projected phases."""

from pathlib import Path

from examples.synthesis.algorithms.amplitude_amplification.qsvt_fixed_point.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth

if __name__ == "__main__":
    compare_states(
        synth.synthesize(Path(__file__).with_name("source.qmod")),
        reference_circuit(),
        ("state.a", "state.b", "state.c", "aux"),
    )
